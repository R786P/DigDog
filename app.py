import os
import io
import json
import uuid
import requests
from urllib.parse import quote
from flask import Flask, render_template, request, redirect, url_for, session, send_file, send_from_directory

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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# In-memory draft storage -> { draft_id: {...} }
# Simple approach for a personal single-user tool (no DB needed).
DRAFTS = {}

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
    """Calls Gemini REST API and returns plain text response."""
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
Give between 5 and 8 chapters."""
    text = call_gemini(prompt, max_tokens=1024)
    text = text.strip().strip("```json").strip("```").strip()
    try:
        chapters = json.loads(text)
    except Exception:
        # fallback: wrap raw text as single chapter so the flow doesn't break
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


# ---------- Routes ----------

@app.route("/sw.js")
def service_worker():
    # Served from root (not /static/) so its scope covers the whole app — required for PWA installability.
    response = send_from_directory(app.static_folder, "sw.js")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Content-Type"] = "application/javascript"
    return response


@app.route("/")
def index():
    return render_template(
        "index.html", palettes=PALETTES, styles=IMAGE_STYLES, languages=LANGUAGES
    )


@app.route("/create", methods=["POST"])
def create():
    topic = request.form.get("topic", "").strip()
    template = request.form.get("template", "minimal")
    palette = request.form.get("palette", "warm")
    image_style = request.form.get("image_style", "flat")
    language = request.form.get("language", "hinglish")

    if not topic:
        return redirect(url_for("index"))

    draft_id = uuid.uuid4().hex
    chapters = generate_outline(topic, language)

    DRAFTS[draft_id] = {
        "topic": topic,
        "template": template,
        "palette": palette,
        "image_style": image_style,
        "language": language,
        "chapters": chapters,  # list of {title, description}
    }
    session["draft_id"] = draft_id
    return redirect(url_for("outline"))


def get_draft():
    draft_id = session.get("draft_id")
    return draft_id, DRAFTS.get(draft_id)


@app.route("/outline", methods=["GET", "POST"])
def outline():
    draft_id, draft = get_draft()
    if not draft:
        return redirect(url_for("index"))

    if request.method == "POST":
        raw = request.form.get("outline_text", "")
        chapters = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                title, desc = line.split(":", 1)
            else:
                title, desc = line, ""
            chapters.append({"title": title.strip(), "description": desc.strip()})
        draft["chapters"] = chapters

        # Generate full content + image prompt for every chapter
        enriched = []
        for ch in chapters:
            content = generate_chapter_content(
                draft["topic"], ch["title"], ch["description"], draft["language"]
            )
            img_prompt = generate_image_prompt(
                draft["topic"], ch["title"], ch["description"], draft["image_style"]
            )
            enriched.append({
                "title": ch["title"],
                "description": ch["description"],
                "content": content,
                "image_prompt": img_prompt,
                "seed": 1,
                "uploaded_image": None,
            })
        draft["chapters"] = enriched
        DRAFTS[draft_id] = draft
        return redirect(url_for("review"))

    outline_text = "\n".join(f"{c['title']}: {c.get('description','')}" for c in draft["chapters"])
    return render_template("outline.html", draft=draft, outline_text=outline_text)


@app.route("/review")
def review():
    draft_id, draft = get_draft()
    if not draft or not draft["chapters"] or "content" not in draft["chapters"][0]:
        return redirect(url_for("index"))

    for ch in draft["chapters"]:
        if ch.get("uploaded_image"):
            ch["image_url"] = url_for("static", filename=f"uploads/{ch['uploaded_image']}")
        else:
            ch["image_url"] = pollinations_url(ch["image_prompt"], ch["seed"])

    return render_template("review.html", draft=draft, palette=PALETTES[draft["palette"]])


@app.route("/regenerate_image/<int:idx>", methods=["POST"])
def regenerate_image(idx):
    draft_id, draft = get_draft()
    if draft and 0 <= idx < len(draft["chapters"]):
        draft["chapters"][idx]["seed"] += 1
        draft["chapters"][idx]["uploaded_image"] = None
    return redirect(url_for("review"))


@app.route("/update_prompt/<int:idx>", methods=["POST"])
def update_prompt(idx):
    draft_id, draft = get_draft()
    if draft and 0 <= idx < len(draft["chapters"]):
        new_prompt = request.form.get("image_prompt", "").strip()
        if new_prompt:
            draft["chapters"][idx]["image_prompt"] = new_prompt
            draft["chapters"][idx]["seed"] += 1
            draft["chapters"][idx]["uploaded_image"] = None
    return redirect(url_for("review"))


@app.route("/update_text/<int:idx>", methods=["POST"])
def update_text(idx):
    draft_id, draft = get_draft()
    if draft and 0 <= idx < len(draft["chapters"]):
        draft["chapters"][idx]["content"] = request.form.get("content", "").strip()
    return redirect(url_for("review"))


@app.route("/upload_image/<int:idx>", methods=["POST"])
def upload_image(idx):
    draft_id, draft = get_draft()
    if not draft or not (0 <= idx < len(draft["chapters"])):
        return redirect(url_for("review"))

    file = request.files.get("image_file")
    if file and file.filename:
        os.makedirs(os.path.join(app.static_folder, "uploads"), exist_ok=True)
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{draft_id}_{idx}{ext}"
        file.save(os.path.join(app.static_folder, "uploads", filename))
        draft["chapters"][idx]["uploaded_image"] = filename
    return redirect(url_for("review"))


@app.route("/download")
def download():
    draft_id, draft = get_draft()
    if not draft:
        return redirect(url_for("index"))

    buf = io.BytesIO()
    accent_hex = PALETTES[draft["palette"]]["hex"]
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

    story = [
        Spacer(1, 6*cm),
        Paragraph(draft["topic"], title_style),
        PageBreak(),
    ]

    for ch in draft["chapters"]:
        story.append(Paragraph(ch["title"], chapter_style))

        # image: uploaded file takes priority over generated image
        img_bytes = None
        try:
            if ch.get("uploaded_image"):
                path = os.path.join(app.static_folder, "uploads", ch["uploaded_image"])
                with open(path, "rb") as f:
                    img_bytes = f.read()
            else:
                img_bytes = fetch_image_bytes(pollinations_url(ch["image_prompt"], ch["seed"]))
        except Exception:
            img_bytes = None

        if img_bytes:
            img_stream = io.BytesIO(img_bytes)
            story.append(RLImage(img_stream, width=12*cm, height=12*cm))
            story.append(Spacer(1, 0.5*cm))

        for para in ch["content"].split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))

        story.append(PageBreak())

    doc.build(story)
    buf.seek(0)

    safe_name = "".join(c for c in draft["topic"] if c.isalnum() or c in (" ", "_")).strip()
    safe_name = safe_name.replace(" ", "_") or "ebook"

    return send_file(
        buf,
        as_attachment=True,
        download_name=f"{safe_name}.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
