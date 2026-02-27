from __future__ import annotations

import io
import os
import subprocess
import uuid
from typing import Optional

import fitz  # PyMuPDF
import httpx
import pdfplumber
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
# In Docker the Dockerfile copies frontend/ alongside main.py; locally it's ../frontend/
FRONTEND_INDEX = os.path.join(_HERE, "frontend", "index.html")
if not os.path.exists(FRONTEND_INDEX):
    FRONTEND_INDEX = os.path.join(_HERE, "..", "frontend", "index.html")

app = FastAPI(title="Image Region OCR / PDF Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory metadata stores (maps id -> file path + metadata)
image_store: dict[str, dict] = {}
pdf_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ImageLoadURL(BaseModel):
    url: str


class Rect(BaseModel):
    x: float
    y: float
    w: float
    h: float


class OCRRequest(BaseModel):
    image_id: str
    rect: Rect
    lang: str = "eng"
    psm: int = 6
    oem: int = 3


class PDFRect(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ExtractPageRequest(BaseModel):
    pdf_id: str
    page_index: int
    include_rect: Optional[PDFRect] = None


class PageRectEntry(BaseModel):
    page_index: int
    rect: PDFRect


class ExtractAllRequest(BaseModel):
    pdf_id: str
    page_rects: list[PageRectEntry]  # only listed pages are extracted; others skipped


class PdftotextRegionRequest(BaseModel):
    pdf_id: str
    page_first: int = 1
    page_last: int = 1
    rect: Rect  # x,y,w,h in PDF points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _crop(img: Image.Image, rect: Rect) -> Image.Image:
    x, y, w, h = int(rect.x), int(rect.y), int(rect.w), int(rect.h)
    return img.crop((x, y, x + w, y + h))


def _fitz_rect(r: PDFRect) -> fitz.Rect:
    return fitz.Rect(r.x0, r.y0, r.x1, r.y1)


def _extract_page_text(
    page: fitz.Page,
    include_rect: Optional[PDFRect] = None,
) -> str:
    """Extract text from a PyMuPDF page, clipped to include_rect if provided."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_idx)
    result_words: list[str] = []
    ir = _fitz_rect(include_rect) if include_rect else None
    for w in words:
        wx0, wy0, wx1, wy1, word = w[0], w[1], w[2], w[3], w[4]
        if ir is not None and not fitz.Rect(wx0, wy0, wx1, wy1).intersects(ir):
            continue
        result_words.append(word)
    return " ".join(result_words)


# ---------------------------------------------------------------------------
# Image endpoints
# ---------------------------------------------------------------------------


@app.post("/api/image/load")
async def image_load_file(file: UploadFile = File(...)):
    """Upload an image file."""
    data = await file.read()
    ext = os.path.splitext(file.filename or "img")[1] or ".jpg"
    img_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{img_id}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    img = _load_image(path)
    w, h = img.size
    image_store[img_id] = {"path": path, "width": w, "height": h}
    return {"image_id": img_id, "width": w, "height": h}


@app.post("/api/image/load_url")
async def image_load_url(body: ImageLoadURL):
    """Fetch an image from a URL (backend-side, bypasses CORS)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        try:
            resp = await client.get(body.url)
            resp.raise_for_status()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    ext = ".jpg"
    for ct in ["png", "webp", "gif", "jpg", "jpeg"]:
        if ct in resp.headers.get("content-type", ""):
            ext = f".{ct}"
    img_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{img_id}{ext}")
    with open(path, "wb") as f:
        f.write(resp.content)
    img = _load_image(path)
    w, h = img.size
    image_store[img_id] = {"path": path, "width": w, "height": h}
    return {"image_id": img_id, "width": w, "height": h}


@app.get("/api/image/preview/{image_id}")
async def image_preview(image_id: str):
    """Return image bytes for preview."""
    if image_id not in image_store:
        raise HTTPException(404, "Not found")
    path = image_store[image_id]["path"]
    img = _load_image(path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.post("/api/image/ocr")
async def image_ocr(req: OCRRequest):
    """Crop a region and run Tesseract OCR on it."""
    if req.image_id not in image_store:
        raise HTTPException(404, "image_id not found")
    meta = image_store[req.image_id]
    img = _load_image(meta["path"])
    img_w, img_h = img.size

    crop = _crop(img, req.rect)
    config = f"--psm {req.psm} --oem {req.oem}"

    text = pytesseract.image_to_string(crop, lang=req.lang, config=config).strip()

    data = pytesseract.image_to_data(
        crop,
        lang=req.lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words = []
    for i, w in enumerate(data["text"]):
        if w.strip():
            words.append(
                {
                    "word": w,
                    "conf": data["conf"][i],
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                }
            )

    # Tesseract bottom-left convention for image_to_boxes
    rect = req.rect
    tess_box = {
        "left": int(rect.x),
        "top": int(rect.y),
        "width": int(rect.w),
        "height": int(rect.h),
        # bottom-left origin variant
        "x1": int(rect.x),
        "y1": img_h - int(rect.y + rect.h),
        "x2": int(rect.x + rect.w),
        "y2": img_h - int(rect.y),
    }

    return {
        "text": text,
        "words": words,
        "tess_box": tess_box,
        "image_width": img_w,
        "image_height": img_h,
        "rect": {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h},
    }


# ---------------------------------------------------------------------------
# PDF endpoints
# ---------------------------------------------------------------------------


@app.post("/api/pdf/load")
async def pdf_load(file: UploadFile = File(...)):
    """Upload a PDF file."""
    data = await file.read()
    pdf_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{pdf_id}.pdf")
    with open(path, "wb") as f:
        f.write(data)

    doc = fitz.open(path)
    page_sizes = []
    for page in doc:
        r = page.rect
        page_sizes.append({"width": r.width, "height": r.height})
    doc.close()

    pdf_store[pdf_id] = {"path": path, "page_count": len(page_sizes)}
    return {"pdf_id": pdf_id, "page_count": len(page_sizes), "page_sizes": page_sizes}


@app.get("/api/pdf/page_render")
async def pdf_page_render(pdf_id: str, page: int = 0, scale: float = 1.5):
    """Render a PDF page to JPEG (for optional backend rendering)."""
    if pdf_id not in pdf_store:
        raise HTTPException(404, "Not found")
    doc = fitz.open(pdf_store[pdf_id]["path"])
    if page < 0 or page >= len(doc):
        raise HTTPException(400, "Page out of range")
    p = doc[page]
    mat = fitz.Matrix(scale, scale)
    pix = p.get_pixmap(matrix=mat, alpha=False)
    doc.close()
    return Response(content=pix.tobytes("jpeg"), media_type="image/jpeg")


@app.post("/api/pdf/extract_page")
async def pdf_extract_page(req: ExtractPageRequest):
    """Extract text from one page, clipped to include_rect if given."""
    if req.pdf_id not in pdf_store:
        raise HTTPException(404, "pdf_id not found")
    doc = fitz.open(pdf_store[req.pdf_id]["path"])
    if req.page_index < 0 or req.page_index >= len(doc):
        raise HTTPException(400, "page_index out of range")
    page = doc[req.page_index]
    text = _extract_page_text(page, req.include_rect)
    doc.close()
    return {"page_index": req.page_index, "text": text}


@app.post("/api/pdf/extract_all")
async def pdf_extract_all(req: ExtractAllRequest):
    """Extract text from pages that have a rect; skip the rest."""
    if req.pdf_id not in pdf_store:
        raise HTTPException(404, "pdf_id not found")
    doc = fitz.open(pdf_store[req.pdf_id]["path"])
    rect_map: dict[int, PDFRect] = {e.page_index: e.rect for e in req.page_rects}
    extracted: list[dict] = []
    for i, page in enumerate(doc):
        if i not in rect_map:
            continue
        text = _extract_page_text(page, rect_map[i])
        extracted.append({"page_index": i, "text": text})
    doc.close()
    combined = "\n\f\n".join(e["text"] for e in extracted)
    return {"pages": extracted, "combined": combined}


@app.post("/api/pdf/pdftotext_region")
async def pdf_pdftotext_region(req: PdftotextRegionRequest):
    """Use Poppler pdftotext with -x -y -W -H for a region (debug/fallback)."""
    if req.pdf_id not in pdf_store:
        raise HTTPException(404, "pdf_id not found")
    path = pdf_store[req.pdf_id]["path"]
    cmd = [
        "pdftotext",
        "-f", str(req.page_first),
        "-l", str(req.page_last),
        "-x", str(int(req.rect.x)),
        "-y", str(int(req.rect.y)),
        "-W", str(int(req.rect.w)),
        "-H", str(int(req.rect.h)),
        path,
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"text": result.stdout, "stderr": result.stderr}
    except FileNotFoundError:
        raise HTTPException(500, "pdftotext (Poppler) not found on PATH")
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "pdftotext timed out")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_frontend():
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {"message": "API running. No bundled frontend found."}
