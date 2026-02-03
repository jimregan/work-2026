"""Flask app for reviewing VibeVoice-to-etext alignment mismatches."""

import difflib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_file

# Add parent directory so librivox_matching is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from librivox_matching.vibevoice import Chunk

app = Flask(__name__)

ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


def _normalize_word(word: str) -> str:
    """Normalize a single word for comparison: lowercase, strip punctuation."""
    w = word.lower()
    w = unicodedata.normalize("NFKC", w)
    w = w.replace("-", " ")
    w = re.sub(r"[^\w\s]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    return w


def _align_chunks_to_etext(chunks: list[Chunk], etext: str) -> list[dict]:
    """Align VibeVoice chunks to etext using SequenceMatcher on normalised words.

    Concatenates all chunk texts, aligns the whole thing against the etext
    at the word level (using normalised forms for comparison), then splits
    the aligned etext back into per-chunk passages using SequenceMatcher's
    matching blocks.
    """
    et_words = etext.split()
    et_norm = [_normalize_word(w) for w in et_words]

    # Build a flat list of all VV words with chunk boundaries
    vv_words = []
    chunk_boundaries = []  # (start_idx, end_idx) into vv_words for each chunk
    for chunk in chunks:
        start = len(vv_words)
        words = chunk.content.split()
        vv_words.extend(words)
        chunk_boundaries.append((start, len(vv_words)))

    vv_norm = [_normalize_word(w) for w in vv_words]

    # Single SequenceMatcher on the full normalised word lists
    matcher = difflib.SequenceMatcher(None, et_norm, vv_norm)
    opcodes = matcher.get_opcodes()

    # For each chunk, find which etext words correspond to it
    # by looking at the opcodes that overlap the chunk's VV word range.
    segments = []
    for ci, chunk in enumerate(chunks):
        vv_start, vv_end = chunk_boundaries[ci]

        # Find the etext word range that maps to this chunk's VV words
        et_start = None
        et_end = None

        for tag, i1, i2, j1, j2 in opcodes:
            # Does this opcode overlap with our chunk's VV range?
            if j2 <= vv_start:
                continue  # entirely before our chunk
            if j1 >= vv_end:
                break  # entirely after our chunk

            # This opcode overlaps with our chunk
            if et_start is None:
                et_start = i1
            et_end = i2

        if et_start is not None:
            passage = " ".join(et_words[et_start:et_end])
            is_boilerplate = False
        else:
            passage = ""
            is_boilerplate = True

        segments.append({
            "index": ci,
            "start": chunk.start,
            "end": chunk.end,
            "vibevoice": chunk.content,
            "etext": passage,
            "boilerplate": is_boilerplate,
        })

    return segments


def compute_word_diff(etext: str, vibevoice: str) -> list[dict]:
    """Compute word-level diff ops between etext and vibevoice text.

    Comparison is done on normalised words (lowercase, no punctuation)
    so that differences in case and punctuation are treated as equal.
    The original words are preserved in the output.
    """
    et_words = etext.split()
    vv_words = vibevoice.split()

    et_norm = [_normalize_word(w) for w in et_words]
    vv_norm = [_normalize_word(w) for w in vv_words]

    if et_norm == vv_norm:
        return [{"op": "equal", "vv_words": vv_words, "et_words": et_words}]

    matcher = difflib.SequenceMatcher(None, et_norm, vv_norm)
    diff_ops = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        diff_ops.append({
            "op": tag,
            "et_words": et_words[i1:i2],
            "vv_words": vv_words[j1:j2],
        })
    return diff_ops


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/align", methods=["POST"])
def align():
    # Get VibeVoice JSON
    if "vibevoice_file" in request.files and request.files["vibevoice_file"].filename:
        vv_text = request.files["vibevoice_file"].read().decode("utf-8")
    else:
        vv_text = request.form.get("vibevoice_text", "")

    # Get etext
    if "etext_file" in request.files and request.files["etext_file"].filename:
        etext = request.files["etext_file"].read().decode("utf-8")
    else:
        etext = request.form.get("etext_text", "")

    if not vv_text or not etext:
        return jsonify({"error": "Both VibeVoice JSON and etext are required."
                        f" Got vv={len(vv_text)} chars, etext={len(etext)} chars."}), 400

    try:
        vv_data = json.loads(vv_text)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid VibeVoice JSON: {e}"}), 400

    # Parse chunks
    try:
        def _get(item, *keys):
            for k in keys:
                if k in item:
                    return item[k]
            raise KeyError(f"Missing key (tried {keys}) in {list(item.keys())}")

        chunks = [
            Chunk(
                start=_get(item, "Start", "start"),
                end=_get(item, "End", "end"),
                speaker=item.get("Speaker", item.get("speaker", item.get("speaker_id", 0))),
                content=_get(item, "Content", "content", "text"),
            )
            for item in vv_data
        ]
    except Exception as e:
        return jsonify({"error": f"Failed to parse VibeVoice data: {e}"}), 400

    # Align chunks to etext
    try:
        segments = _align_chunks_to_etext(chunks, etext)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Alignment failed: {e}"}), 500

    # Add word-level diffs
    for seg in segments:
        if seg["boilerplate"] or not seg["etext"]:
            seg["diff_ops"] = []
        else:
            seg["diff_ops"] = compute_word_diff(seg["etext"], seg["vibevoice"])

    return jsonify({"segments": segments})


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"annotations_{timestamp}.json"
    filepath = os.path.join(ANNOTATIONS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved", "filename": filename})


@app.route("/export")
def export():
    # Find the most recent annotations file
    files = sorted(
        [f for f in os.listdir(ANNOTATIONS_DIR) if f.startswith("annotations_")],
        reverse=True,
    )
    if not files:
        return jsonify({"error": "No annotations saved yet"}), 404

    filepath = os.path.join(ANNOTATIONS_DIR, files[0])
    return send_file(filepath, as_attachment=True, download_name=files[0])


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
