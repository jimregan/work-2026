# image-region-ocr

A local web tool for selecting regions on images or PDFs and extracting text from them.

## Modes

### Image mode

Upload an image (or load from a URL) and drag a rectangle over any region. The sidebar shows the selection in several copyable formats:

- space-separated `x y w h` — for use with `bbox_text.split(" ")`
- JSON array `[x, y, w, h]`
- JSON object `{"x": …, "y": …, "w": …, "h": …}`
- Tesseract `image_to_boxes` bottom-left convention

Click **Run OCR** to extract text from the selected region via Tesseract. Set the language code (e.g. `gle`, `deu`, `fra`) before running. The Words tab shows per-word confidence.

### PDF mode

Upload a PDF. Pages are rendered in-browser via PDF.js. Draw a rectangle on each page to define the body region to extract. Pages with no rectangle are skipped.

- Navigate pages with ← →; each page stores its own region independently
- **Get Page Text** — extract text from the current page's region
- **Export All Pages** — extract all pages that have regions, concatenated with page breaks
- **All Odd / All Even / All Pages** — copy the current page's region to other pages
- The page overview panel on the right lists every page and its region (or "skipped")
- Save/load region config as JSON; save extracted text as `.txt`

Text extraction uses PyMuPDF with line breaks and paragraph structure preserved.

## Running

### Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>.

### Conda (local dev)

```bash
conda create -n image-region-ocr python=3.11
conda run -n image-region-ocr pip install -r backend/requirements.txt
./run.sh
```

Open <http://localhost:8000> (or open `frontend/index.html` directly — it falls back to `localhost:8000` for API calls).

## Adding Tesseract languages

In Docker, add the package name to the `apt-get` line in `Dockerfile` and rebuild:

```dockerfile
tesseract-ocr-gle \   # Irish (already included)
tesseract-ocr-deu \   # German
tesseract-ocr-fra \   # French
```

For the conda setup, install via Homebrew or your system package manager:

```bash
brew install tesseract-lang
```
