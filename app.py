import os
import io
import json
import time
import uuid
import requests
from urllib.parse import quote
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# ---------- Groq API config ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "static", "generated")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# In-memory draft storage: draft_id -> full editable eBook data.
# This is what powers the Customize Studio — the PDF alone can't be "edited",
# so we keep the underlying text/image data around after generation too.
DRAFTS = {}

PALETTES = {
    "warm":    {"name": "Warm & Earthy",                "hex": "#B45309"},
    "pastel":  {"name": "Pastel",                       "hex": "#DB2777"},
    "bold":    {"name": "Bold & Vibrant",               "hex": "#DC2626"},
    "bw":      {"name": "Black & White (Coloring Book)","hex": "#111111"},
    "dark":    {"name": "Dark Premium",                 "hex": "#6D28D9"},
}

IMAGE_STYLES = {
    "flat":       "flat illustration style, vibrant colors, clean simple background",
    "realistic":  "photorealistic, soft natural lighting, high detail, professional photography",
    "3d":         "3D render style, soft shadows, glossy finish, vibrant colors",
    "watercolor": "watercolor painting style, soft brush strokes, artistic, light paper texture",
    "lineart":    "black and white line art, coloring book style, bold clean outlines, no shading, no color",
    "cartoon":    "cartoon illustration style, bold outlines, vibrant flat colors, playful",
}

TEMPLATES = ["minimal", "magazine", "storybook", "coloring", "report"]

LANGUAGES = {
    "hinglish": "Hinglish (a natural mix of Hindi and English, written in Roman/English script)",
    "hindi":    "Hindi (written in Devanagari script)",
    "english":  "English",
}


# ---------- Groq helpers ----------
def call_groq(prompt, max_tokens=2048):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.9,
    }

    # Groq is usually very fast, but we still keep a small retry loop for
    # transient 429/5xx errors so a single bad moment doesn't kill the eBook.
    last_error = None
    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code in (429, 500, 503) and attempt < 3:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_error


def generate_outline(topic, language):
    prompt = f"""You are an expert eBook writer. Create a chapter outline for an eBook on the topic: "{topic}".
Write the chapter titles and short descriptions in {LANGUAGES[language]}.
Return ONLY a valid JSON array, no markdown, no extra text, in this exact format:
[{{ "title": "Chapter Title", "description": "one line description" }}]
Give between 5 and 7 chapters."""
    text = call_groq(prompt, max_tokens=1024)
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
    return call_groq(prompt, max_tokens=1024).strip()


def generate_image_prompt(topic, title, description, style_key):
    # Image prompts are always generated in English (AI image tools respond best in English).
    prompt = f"""Based on an eBook titled "{topic}", write ONE short AI image generation prompt (max 25 words, in English)
for the chapter "{title}" ({description}). Describe a visual scene only, no text/words in the image.
Return ONLY the prompt text, nothing else."""
    try:
        base = call_groq(prompt, max_tokens=100).strip().strip('"')
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


def strip_pollinations_mark(img_bytes):
    """Covers the small 'pollinations.ai' attribution mark Pollinations stamps in the
    bottom-right corner of generated images, by patching that corner with a sampled
    nearby color. Only used for AI-generated images, never for the user's own uploads."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        patch_w, patch_h = int(w * 0.34), int(h * 0.10)
        x0, y0 = w - patch_w, h - patch_h
        sample_x = max(0, x0 - 8)
        sample_y = max(0, y0 - 8)
        sample_color = img.getpixel((sample_x, sample_y))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, w, h], fill=sample_color)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return img_bytes


def get_chapter_image_bytes(ch, hide_watermark):
    """Returns the raw bytes to embed for one chapter, or None if no image should be shown."""
    if ch.get("no_image"):
        return None
    if ch.get("uploaded_image"):
        path = os.path.join(UPLOADS_DIR, ch["uploaded_image"])
        try:
            with open(path, "rb") as f:
                return f.read()  # user's own image — never watermark-stripped
        except Exception:
            return None
    try:
        img_bytes = fetch_image_bytes(pollinations_url(ch["image_prompt"], ch.get("seed", 1)))
        if hide_watermark:
            img_bytes = strip_pollinations_mark(img_bytes)
        return img_bytes
    except Exception:
        return None


# ---------- PDF builder ----------
def build_pdf(draft):
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
    watermark_style = ParagraphStyle(
        "Watermark", parent=styles["BodyText"], fontSize=8, textColor=colors.grey, spaceBefore=18
    )

    story = [Spacer(1, 6*cm), Paragraph(draft["topic"], title_style), PageBreak()]

    hide_watermark = draft.get("hide_pollinations_watermark", True)
    custom_watermark = (draft.get("watermark_text") or "").strip()

    for ch in draft["chapters"]:
        story.append(Paragraph(ch["title"], chapter_style))
        img_bytes = get_chapter_image_bytes(ch, hide_watermark)
        if img_bytes:
            story.append(RLImage(io.BytesIO(img_bytes), width=12*cm, height=12*cm))
            story.append(Spacer(1, 0.5*cm))
        for para in ch["content"].split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
        if custom_watermark:
            story.append(Paragraph(custom_watermark, watermark_style))
        story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf.read()


def save_pdf_for_draft(draft_id, draft):
    pdf_bytes = build_pdf(draft)
    filename = f"{draft_id}.pdf"
    with open(os.path.join(GENERATED_DIR, filename), "wb") as f:
        f.write(pdf_bytes)
    return f"/static/generated/{filename}"


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
    template = request.form.get("template", "minimal")
    image_style = request.form.get("image_style", "flat")
    image_style_locked = request.form.get("image_style_locked") == "1"

    if not topic:
        return jsonify({"status": "error", "message": "Topic likhna zaroori hai."}), 400

    # Auto-detect "coloring book" intent from the topic text itself — if the user typed
    # it directly in the chat box without opening the "+" options panel, we still want
    # outline-only, colorable artwork instead of already-colored illustrations.
    COLORING_KEYWORDS = ["coloring book", "colouring book", "color book", "colour book", "rang bharo", "rang bharne"]
    if not image_style_locked and any(k in topic.lower() for k in COLORING_KEYWORDS):
        image_style = "lineart"

    try:
        outline = generate_outline(topic, language)
        chapters = []
        for ch in outline:
            content = generate_chapter_content(topic, ch["title"], ch.get("description", ""), language)
            img_prompt = generate_image_prompt(topic, ch["title"], ch.get("description", ""), image_style)
            chapters.append({
                "title": ch["title"],
                "content": content,
                "image_prompt": img_prompt,
                "seed": 1,
                "uploaded_image": None,
                "no_image": False,
            })

        draft_id = uuid.uuid4().hex
        draft = {
            "topic": topic,
            "language": language,
            "palette": palette,
            "template": template,
            "image_style": image_style,
            "hide_pollinations_watermark": True,
            "watermark_text": "",
            "chapters": chapters,
        }
        DRAFTS[draft_id] = draft

        download_url = save_pdf_for_draft(draft_id, draft)
        return jsonify({
            "status": "ok",
            "title": topic,
            "chapters": len(chapters),
            "download_url": download_url,
            "edit_url": url_for("studio", draft_id=draft_id),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------- Customize Studio ----------
@app.route("/studio/<draft_id>")
def studio(draft_id):
    draft = DRAFTS.get(draft_id)
    if not draft:
        return redirect(url_for("index"))

    for ch in draft["chapters"]:
        if ch.get("no_image"):
            ch["image_url"] = None
        elif ch.get("uploaded_image"):
            ch["image_url"] = url_for("static", filename=f"uploads/{ch['uploaded_image']}")
        else:
            ch["image_url"] = pollinations_url(ch["image_prompt"], ch.get("seed", 1))

    return render_template(
        "studio.html",
        draft=draft,
        draft_id=draft_id,
        palettes=PALETTES,
        styles=IMAGE_STYLES,
        templates=TEMPLATES,
        pdf_url=f"/static/generated/{draft_id}.pdf",
    )


@app.route("/studio/<draft_id>/regenerate_image/<int:idx>", methods=["POST"])
def studio_regenerate_image(draft_id, idx):
    draft = DRAFTS.get(draft_id)
    if draft and 0 <= idx < len(draft["chapters"]):
        draft["chapters"][idx]["seed"] += 1
        draft["chapters"][idx]["uploaded_image"] = None
        draft["chapters"][idx]["no_image"] = False
    return redirect(url_for("studio", draft_id=draft_id))


@app.route("/studio/<draft_id>/update_prompt/<int:idx>", methods=["POST"])
def studio_update_prompt(draft_id, idx):
    draft = DRAFTS.get(draft_id)
    if draft and 0 <= idx < len(draft["chapters"]):
        new_prompt = request.form.get("image_prompt", "").strip()
        if new_prompt:
            draft["chapters"][idx]["image_prompt"] = new_prompt
            draft["chapters"][idx]["seed"] += 1
            draft["chapters"][idx]["uploaded_image"] = None
            draft["chapters"][idx]["no_image"] = False
    return redirect(url_for("studio", draft_id=draft_id))


@app.route("/studio/<draft_id>/upload_image/<int:idx>", methods=["POST"])
def studio_upload_image(draft_id, idx):
    draft = DRAFTS.get(draft_id)
    if not draft or not (0 <= idx < len(draft["chapters"])):
        return redirect(url_for("index"))

    file = request.files.get("image_file")
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        filename = f"{draft_id}_{idx}{ext}"
        file.save(os.path.join(UPLOADS_DIR, filename))
        draft["chapters"][idx]["uploaded_image"] = filename
        draft["chapters"][idx]["no_image"] = False
    return redirect(url_for("studio", draft_id=draft_id))


@app.route("/studio/<draft_id>/remove_image/<int:idx>", methods=["POST"])
def studio_remove_image(draft_id, idx):
    draft = DRAFTS.get(draft_id)
    if draft and 0 <= idx < len(draft["chapters"]):
        draft["chapters"][idx]["no_image"] = True
        draft["chapters"][idx]["uploaded_image"] = None
    return redirect(url_for("studio", draft_id=draft_id))


@app.route("/studio/<draft_id>/save", methods=["POST"])
def studio_save(draft_id):
    draft = DRAFTS.get(draft_id)
    if not draft:
        return jsonify({"status": "error", "message": "Draft not found."}), 404

    draft["topic"] = request.form.get("topic", draft["topic"]).strip() or draft["topic"]
    draft["template"] = request.form.get("template", draft["template"])
    draft["palette"] = request.form.get("palette", draft["palette"])
    draft["hide_pollinations_watermark"] = request.form.get("hide_pollinations_watermark") == "1"
    draft["watermark_text"] = request.form.get("watermark_text", "").strip()

    for i, ch in enumerate(draft["chapters"]):
        new_title = request.form.get(f"title_{i}")
        new_content = request.form.get(f"content_{i}")
        if new_title is not None:
            ch["title"] = new_title.strip() or ch["title"]
        if new_content is not None:
            ch["content"] = new_content.strip() or ch["content"]

    try:
        download_url = save_pdf_for_draft(draft_id, draft)
        return jsonify({"status": "ok", "download_url": download_url})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
