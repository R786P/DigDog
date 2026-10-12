import os
import io
import json
import uuid
import requests
from urllib.parse import quote
from flask import Flask, render_template, request, jsonify, send_from_directory

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

PALETTES = {
    "warm":   {"name": "Warm & Earthy",        "hex": "#B45309"},
    "pastel": {"name": "Pastel",                "hex": "#DB2777"},
    "bold":   {"name": "Bold & Vibrant",        "hex": "#DC2626"},
    "bw":     {"name": "Black & White (Coloring Book)", "hex": "#111111"},
    "dark":   {"name": "Dark Premium",          "hex": "#6D28D9"},
}

IMAGE_STYLES = {
    "flat":      "flat illustration style, vibrant colors, clean simple background",
    "realistic": "photorealistic, soft natural lighting, high detail, professional photography",
    "3d":        "3D render style, soft shadows, glossy finish, vibrant colors",
    "watercolor": "watercolor painting style, soft brush strokes, artistic, light paper texture",
    "lineart":   "black and white line art, coloring book style, bold clean outlines, no shading, no color",
    "cartoon":   "cartoon illustration style, bold outlines, vibrant flat colors, playful",
}

LANGUAGES = {
    "hinglish": "Hinglish (a natural mix of Hindi and English, written in Roman/English script)",
    "hindi":    "Hindi (written in Devanagari script)",
    "english":  "English",
}


# ---------- Gemini helpers ----------

def call_gemini(prompt, max_tokens=2048):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.9},
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def generate_outline(topic, language):
    prompt = f"""You are an expert eBook writer. Create a chapter outline for an eBook on the topic: "{topic}".
Write the chapter titles and short descriptions in {LANGUAGES[language]}.
Return ONLY a valid JSON array, no markdown, no extra text, in this exact format:
[{{"title": "Chapter Title", "description": "one line description"}}]
Give between 5 and 7 chapters."""
    text = call_gemini(prompt, max_tokens=1024)
    text = text.strip().strip("```json").strip("```").strip()
    try:
        chapters = json.loads(text)
    except Exception:
        chapters = [{"title": "Chapter 1", "description": text[:200]}]
    return chapters


def generate_chapter_content(topic, title, description, language):
    prompt = f"""Write one chapter of an eBook on the overall topic "{topic}".
Chapter title: "{title}"
Chapter focus: {description}
Write in {LANGUAGES[language]}. Length: 350-500 words. Practical, engaging, easy to read.
Return ONLY the chapter body text, no title, no markdown headers."""
    return call_gemini(prompt, max_tokens=1024).strip()


def generate_image_prompt(topic, title, description, style_key):
    # Image prompts are always generated in English (AI image tools respond best in English).
    prompt = f"""Based on an eBook titled "{topic}", write ONE short AI image generation prompt (max 25 words, in English)
for the chapter "{title}" ({description}). Describe a visual scene only, no text/words in the image.
Return ONLY the prompt text, nothing else."""
    try:
        base = call_gemini(prompt, max_tokens=100).strip().strip('"')
    except Exception:
        base = f"illustration representing {title}"
    return f"{base}, {IMAGE_STYLES[style_key]}"


# ---------- Image helpers ----------

def pollinations_url(image_prompt, seed=1):
    encoded = quote(image_prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&seed={seed}&nologo=true"


def fetch_image_bytes(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


# ---------- PDF builder ----------

def build_pdf(topic, chapters, palette_key):
    buf = io.BytesIO()
    accent_hex = PALETTES[palette_key]["hex"]
    accent_color = colors.HexColor(accent_hex)

    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], textColor=accent_color, fontSize=28
    )
    chapter_style = ParagraphStyle(
        "ChapterHeading", parent=styles["Heading1"], textColor=accent_color, spaceAfter=12
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=10
    )

    story = [Spacer(1, 6*cm), Paragraph(topic, title_style), PageBreak()]

    for ch in chapters:
        story.append(Paragraph(ch["title"], chapter_style))
        try:
            img_bytes = fetch_image_bytes(pollinations_url(ch["image_prompt"]))
        except Exception:
            img_bytes = None
        if img_bytes:
            story.append(RLImage(io.BytesIO(img_bytes), width=12*cm, height=12*cm))
            story.append(Spacer(1, 0.5*cm))
        for para in ch["content"].split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
        story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ---------- Routes ----------

@app.route("/sw.js")
def service_worker():
    response = send_from_directory(app.static_folder, "sw.js")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Content-Type"] = "application/javascript"
    return response


@app.route("/")
def index():
    return render_template(
        "index.html", palettes=PALETTES, styles=IMAGE_STYLES, languages=LANGUAGES
    )


@app.route("/generate", methods=["POST"])
def generate():
    topic = request.form.get("topic", "").strip()
    language = request.form.get("language", "hinglish")
    palette = request.form.get("palette", "warm")
    image_style = request.form.get("image_style", "flat")

    if not topic:
        return jsonify({"status": "error", "message": "Topic likhna zaroori hai."}), 400

    try:
        outline = generate_outline(topic, language)
        chapters = []
        for ch in outline:
            content = generate_chapter_content(topic, ch["title"], ch.get("description", ""), language)
            img_prompt = generate_image_prompt(topic, ch["title"], ch.get("description", ""), image_style)
            chapters.append({"title": ch["title"], "content": content, "image_prompt": img_prompt})

        pdf_bytes = build_pdf(topic, chapters, palette)

        filename = f"{uuid.uuid4().hex}.pdf"
        with open(os.path.join(GENERATED_DIR, filename), "wb") as f:
            f.write(pdf_bytes)

        return jsonify({
            "status": "ok",
            "title": topic,
            "chapters": len(chapters),
            "download_url": f"/static/generated/{filename}",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
