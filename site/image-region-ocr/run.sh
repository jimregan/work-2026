#!/usr/bin/env bash
# Run the backend locally (conda). Open http://localhost:8000 in your browser.
set -e
cd "$(dirname "$0")/backend"
exec conda run -n image-region-ocr uvicorn main:app --reload --port 8000
