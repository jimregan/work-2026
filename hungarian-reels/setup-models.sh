#!/usr/bin/env bash
# Create the three Hungarian Reel models on the Ollama server.
# Usage: OLLAMA_HOST=http://your-server:11434 ./setup-models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "Target: $OLLAMA_HOST"

for model in ear eye brain; do
    echo "Creating hu-$model from $SCRIPT_DIR/modelfiles/Modelfile.$model ..."
    ollama create "hu-$model" -f "$SCRIPT_DIR/modelfiles/Modelfile.$model"
done

echo "Done. Models available: hu-ear, hu-eye, hu-brain"
