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

    Does a single global alignment of all concatenated VV words against the
    etext, then walks through the opcodes splitting them at chunk boundaries
    to produce per-chunk diff ops directly. Each chunk gets its own list of
    diff ops with no overlap.
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
    opcodes = list(matcher.get_opcodes())

    # Split opcodes at chunk boundaries to produce per-chunk diff ops.
    # Each opcode covers et_words[i1:i2] and vv_words[j1:j2].
    # We walk through opcodes and chunk boundaries together.
    chunk_ops: list[list[dict]] = [[] for _ in chunks]
    op_idx = 0

    for ci in range(len(chunks)):
        vv_start, vv_end = chunk_boundaries[ci]

        while op_idx < len(opcodes):
            tag, i1, i2, j1, j2 = opcodes[op_idx]

            if j1 >= vv_end:
                # This opcode starts at or after our chunk ends.
                # Any "delete" ops (no VV words) right before a chunk
                # boundary should be assigned to the next chunk.
                break

            if j2 <= vv_start:
                # Entirely before our chunk — skip
                op_idx += 1
                continue

            # This opcode overlaps with our chunk.
            if j1 >= vv_start and j2 <= vv_end:
                # Entirely within our chunk — take it whole
                chunk_ops[ci].append({
                    "op": tag,
                    "et_words": et_words[i1:i2],
                    "vv_words": vv_words[j1:j2],
                })
                op_idx += 1
            elif j1 < vv_start:
                # Opcode starts before our chunk — split it
                # The part before vv_start belongs to previous chunk(s)
                # Take only the part from vv_start onward
                vv_offset = vv_start - j1
                vv_portion = j2 - vv_start
                total_vv = j2 - j1
                total_et = i2 - i1

                if tag == "equal":
                    et_offset = vv_offset
                    chunk_ops[ci].append({
                        "op": "equal",
                        "et_words": et_words[i1 + et_offset:i2],
                        "vv_words": vv_words[vv_start:j2],
                    })
                elif tag in ("replace", "insert"):
                    # For replace/insert, proportionally split etext words
                    if total_vv > 0:
                        et_split = int(total_et * vv_offset / total_vv)
                    else:
                        et_split = total_et
                    chunk_ops[ci].append({
                        "op": tag,
                        "et_words": et_words[i1 + et_split:i2],
                        "vv_words": vv_words[vv_start:j2],
                    })
                # delete ops have no VV words, so they can't span chunks
                op_idx += 1
            elif j2 > vv_end:
                # Opcode extends past our chunk — split it
                vv_portion = vv_end - j1
                total_vv = j2 - j1
                total_et = i2 - i1

                if tag == "equal":
                    chunk_ops[ci].append({
                        "op": "equal",
                        "et_words": et_words[i1:i1 + vv_portion],
                        "vv_words": vv_words[j1:vv_end],
                    })
                    # Update opcode for next chunk
                    opcodes[op_idx] = (tag, i1 + vv_portion, i2,
                                       vv_end, j2)
                elif tag in ("replace", "insert"):
                    if total_vv > 0:
                        et_split = int(total_et * vv_portion / total_vv)
                    else:
                        et_split = 0
                    chunk_ops[ci].append({
                        "op": tag,
                        "et_words": et_words[i1:i1 + et_split],
                        "vv_words": vv_words[j1:vv_end],
                    })
                    opcodes[op_idx] = (tag, i1 + et_split, i2,
                                       vv_end, j2)
                # Don't advance op_idx — remainder belongs to next chunk
                break
            else:
                op_idx += 1

    # Build segments with their diff ops
    segments = []
    for ci, chunk in enumerate(chunks):
        ops = chunk_ops[ci]

        # Determine etext passage from the ops
        et_passage_words = []
        for op in ops:
            et_passage_words.extend(op["et_words"])
        passage = " ".join(et_passage_words)

        # A chunk is boilerplate if it has no matching etext words
        has_match = any(
            op["op"] == "equal" and op["et_words"]
            for op in ops
        )

        segments.append({
            "index": ci,
            "start": chunk.start,
            "end": chunk.end,
            "vibevoice": chunk.content,
            "etext": passage,
            "boilerplate": not has_match,
            "diff_ops": ops,
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

    # diff_ops are already computed by _align_chunks_to_etext

    return jsonify({"segments": segments})


@app.route("/load", methods=["POST"])
def load():
    """Reconstruct segments from a previously exported annotations JSON."""
    data = request.get_json()
    if not data or not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of annotations"}), 400

    # Group annotations by segment_index and extract segment metadata
    seg_map = {}
    for entry in data:
        si = entry.get("segment_index")
        if si is None:
            continue
        if si not in seg_map:
            seg_map[si] = {
                "start": entry.get("start", 0),
                "end": entry.get("end", 0),
                "vibevoice": entry.get("vibevoice", ""),
                "etext": entry.get("etext", ""),
            }

    # Build segments with recomputed diff_ops, re-indexed from 0
    # Keep a mapping from original index to new index for the client
    sorted_keys = sorted(seg_map.keys())
    segments = []
    index_map = {}  # original_index → new_index
    for new_idx, si in enumerate(sorted_keys):
        info = seg_map[si]
        diff_ops = compute_word_diff(info["etext"], info["vibevoice"])
        segments.append({
            "index": new_idx,
            "start": info["start"],
            "end": info["end"],
            "vibevoice": info["vibevoice"],
            "etext": info["etext"],
            "boilerplate": False,
            "diff_ops": diff_ops,
        })
        index_map[si] = new_idx

    return jsonify({"segments": segments, "index_map": {str(k): v for k, v in index_map.items()}})


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
