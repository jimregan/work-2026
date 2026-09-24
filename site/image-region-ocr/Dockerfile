FROM python:3.11-slim

# ── System packages ──────────────────────────────────────────────────────────
# tesseract-ocr-gle  : Irish
# tesseract-ocr-osd  : orientation/script detection
# poppler-utils      : pdftotext fallback
# libgomp1           : OpenMP (required by some PyMuPDF builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-gle \
        tesseract-ocr-osd \
        poppler-utils \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY backend/ .
COPY frontend/ ./frontend/

RUN mkdir -p uploads

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
