#!/bin/bash
set -e
here="$(cd "$(dirname "$0")" && pwd)"

docker run --rm \
  -v chroma:/data \
  -v "$here":/app \
  --entrypoint bash \
  ghcr.io/open-webui/pipelines:main \
  -c "pip install -q chromadb==1.5.9 && python /app/debug_collection.py"
