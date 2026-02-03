"""Flask app for reviewing VibVoice-to-etext alignment mismatches."""

import difflib
import json
import os
import sys
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_file

# Add parent directory so librivox_matching is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from librivox_matching.normalize import normalize_for_matching
from librivox_matching.vibevoice import Chunk

app = Flask(__name__)

ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


def _build_norm_to_orig_map(original: str, normalised: str) -> list[int]:
    """Build a mapping from each character index in normalised text
    back to the corresponding character index in the original text.

    Uses SequenceMatcher to align the two strings character by character.
    Returns a list where map[norm_idx] = orig_idx.
    """
    orig_lower = original.lower()
    mapping = [0] * len(normalised)

    # Walk through both strings matching characters greedily
    oi = 0
    for ni, nc in enumerate(normalised):
        # Find nc in original starting from oi
        while oi < len(orig_lower) and orig_lower[oi] != nc:
            oi += 1
        if oi < len(orig_lower):
            mapping[ni] = oi
            oi += 1
        else:
            # Past end of original; map to end
            mapping[ni] = len(original)
    return mapping


def _find_passage(
    norm_chunk: str, norm_etext: str, orig_etext: str
) -> tuple[str, float]:
    """Find the passage in orig_etext that best matches norm_chunk.

    Works entirely in normalised character space to avoid word-count
    mismatches from number verbalisation, then maps back to the original
    etext to extract the passage with its original formatting.

    Returns (passage, score) where score is 0..1.
    """
    if not norm_chunk or not norm_etext:
        return "", 0.0

    # Use rapidfuzz for fast substring matching if available
    try:
        from rapidfuzz.fuzz import partial_ratio_alignment
        result = partial_ratio_alignment(norm_chunk, norm_etext)
        if result is None or result.score < 40:
            return "", 0.0

        # result.dest_start/dest_end are positions in norm_etext
        norm_start = result.dest_start
        norm_end = result.dest_end
        score = result.score / 100.0
    except ImportError:
        # Fallback to difflib
        matcher = difflib.SequenceMatcher(None, norm_chunk, norm_etext)
        blocks = matcher.get_matching_blocks()
        if not blocks or (len(blocks) == 1 and blocks[0].size == 0):
            return "", 0.0
        # Find the extent in norm_etext covered by matching blocks
        norm_start = min(b.b for b in blocks if b.size > 0)
        norm_end = max(b.b + b.size for b in blocks if b.size > 0)
        matched_chars = sum(b.size for b in blocks)
        score = matched_chars / max(len(norm_chunk), 1)

    # Map normalised positions back to original etext
    mapping = _build_norm_to_orig_map(orig_etext, norm_etext)

    orig_start = mapping[norm_start] if norm_start < len(mapping) else len(orig_etext)
    # For end, we want to include the full original word
    orig_end = mapping[norm_end - 1] + 1 if norm_end - 1 < len(mapping) else len(orig_etext)

    # Extend to word boundaries in original text
    while orig_start > 0 and orig_etext[orig_start - 1] not in " \n\r\t":
        orig_start -= 1
    while orig_end < len(orig_etext) and orig_etext[orig_end] not in " \n\r\t":
        orig_end += 1

    passage = orig_etext[orig_start:orig_end].strip()
    return passage, score


def _normalize_word(word: str) -> str:
    """Normalize a single word for comparison: lowercase, strip punctuation."""
    import re
    import unicodedata
    w = word.lower()
    w = unicodedata.normalize("NFKC", w)
    w = w.replace("-", " ")
    w = re.sub(r"[^\w\s]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    return w


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
    # Get vibevoice JSON
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
        return jsonify({"error": "Both vibevoice JSON and etext are required."
                        f" Got vv={len(vv_text)} chars, etext={len(etext)} chars."}), 400

    try:
        vv_data = json.loads(vv_text)
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid vibevoice JSON: {e}"}), 400

    # Parse and merge chunks
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
        # Don't merge: VibeVoice chunks are already well-segmented,
        # and merging creates chunks too large for the matcher.
    except Exception as e:
        return jsonify({"error": f"Failed to parse VibVoice data: {e}"}), 400

    # Build a normalised-char to original-char mapping for the etext.
    # This lets us find the original etext passage after matching in
    # normalised space, without the word-count mismatch bug.
    norm_etext = normalize_for_matching(etext)

    result_segments = []
    for i, chunk in enumerate(chunks):
        norm_chunk = normalize_for_matching(chunk.content)

        # Use SequenceMatcher on normalised text (character level) to
        # find the best matching region in the etext.
        passage, score = _find_passage(norm_chunk, norm_etext, etext)

        is_boilerplate = score < 0.4

        seg_out = {
            "index": i,
            "start": chunk.start,
            "end": chunk.end,
            "vibevoice": chunk.content,
            "etext": passage,
            "boilerplate": is_boilerplate,
        }

        if is_boilerplate:
            seg_out["diff_ops"] = []
        else:
            seg_out["diff_ops"] = compute_word_diff(passage, chunk.content)

        result_segments.append(seg_out)

    return jsonify({"segments": result_segments})


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
