# image-region-ocr

A local web tool for selecting rectangular regions on images or PDFs and extracting text from them.

## What it does

**Image mode**
- Upload an image (PNG/JPG/WebP) or load from a URL (fetched backend-side, bypasses CORS)
- Draw a rectangle on the image with click-drag; handles let you resize/move it
- Coords tab shows the selection in multiple formats with one-click copy:
  - space-separated `x y w h` (for `bbox_text.split(" ")` style usage)
  - JSON array `[x, y, w, h]`
  - JSON object `{x, y, w, h}`
  - Tesseract `image_to_boxes` bottom-left convention
- OCR Result tab: full text extracted from the region
- Words tab: per-word confidence table (`pytesseract.image_to_data`)
- Language selector: enter any Tesseract language code (`gle`, `deu`, `fra`, …); defaults to `eng`

**PDF mode**
- Upload a PDF; PDF.js renders each page in-browser
- Draw exclusion rectangles (headers/footers) on pages
- Templates per page parity: **Odd**, **Even**, optional **First** and **Last** page overrides
- "Extract This Page" / "Export All Pages" removes excluded regions and extracts text via PyMuPDF word-level filtering
- Save/load templates as JSON config; save extracted text as `.txt`

## Stack

- **Backend**: Python, FastAPI, pytesseract, Pillow, PyMuPDF (`fitz`), pdfplumber, httpx
- **Frontend**: single `frontend/index.html` — vanilla JS + Canvas API + PDF.js from CDN; no build step
- When served from the backend (Docker/uvicorn) the frontend uses relative API URLs; when opened as `file://` it falls back to `http://localhost:8000`

## Running

### Docker (recommended)

```bash
docker compose up --build
# open http://localhost:8000
```

Uploaded files are persisted in `./uploads/` via a bind mount.

To add more Tesseract language packs, add `tesseract-ocr-XXX` to the `apt-get` line in `Dockerfile` and rebuild. See `/usr/share/tesseract-ocr/4.00/tessdata/` inside the container for installed langs.

### Conda (local dev)

```bash
conda create -n image-region-ocr python=3.11
conda run -n image-region-ocr pip install -r backend/requirements.txt
./run.sh
# open http://localhost:8000  (or open frontend/index.html directly)
```

`./run.sh` starts uvicorn with `--reload` for live backend reloads. The frontend can also be opened as a plain `file://` without the backend serving it.

## Project structure

```
image-region-ocr/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── run.sh                    # local dev launcher (conda)
├── backend/
│   ├── main.py               # FastAPI app + all endpoints
│   ├── requirements.txt
│   └── uploads/              # temp storage (gitignored except .gitkeep)
└── frontend/
    └── index.html            # entire UI — no build step
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/image/load` | Upload image file |
| POST | `/api/image/load_url` | Fetch image from URL |
| GET  | `/api/image/preview/{id}` | Return JPEG preview |
| POST | `/api/image/ocr` | Crop region and run Tesseract |
| POST | `/api/pdf/load` | Upload PDF |
| GET  | `/api/pdf/page_render` | Render page to JPEG (optional) |
| POST | `/api/pdf/extract_page` | Extract text from one page with exclude rects |
| POST | `/api/pdf/extract_all` | Extract all pages with odd/even/first/last templates |
| POST | `/api/pdf/pdftotext_region` | Poppler pdftotext region fallback |
| GET  | `/` | Serves `frontend/index.html` |

## Coordinate conventions

- Image mode stores selections as `{x, y, w, h}` in **image pixels, top-left origin**
- PDF mode stores rectangles as `{x0, y0, x1, y1}` in **PDF points** (PyMuPDF top-left origin, y increases downward), converted from canvas coordinates by dividing by the render scale
- Tesseract `image_to_boxes` uses bottom-left origin: `y1 = H - (y + h)`, `y2 = H - y`
