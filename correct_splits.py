#!/usr/bin/env python3
"""WhisperX segment split corrector.

Presents segments as a plain-text editor (one line per segment, words joined
by spaces).  Use normal editing to fix sentence boundaries:

  Backspace at start of line  → merge with previous segment
  Enter anywhere              → split segment at cursor

On save, segments are reconstructed from the flat word list: text is
" ".join(word["word"] for word in words), start/end come from words[0]/words[-1].

Usage:
    python correct_splits.py [path/to/file.json]
    # then open http://127.0.0.1:5000 in a browser
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WhisperX Segment Editor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #1e1e1e;
    color: #d4d4d4;
  }

  #toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #2d2d2d;
    border-bottom: 1px solid #404040;
    flex-shrink: 0;
  }
  #toolbar input[type="text"] {
    flex: 1;
    background: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555;
    padding: 4px 8px;
    font-size: 13px;
    border-radius: 3px;
  }
  #toolbar button {
    padding: 4px 14px;
    cursor: pointer;
    background: #0e639c;
    color: white;
    border: none;
    border-radius: 3px;
    font-size: 13px;
    white-space: nowrap;
  }
  #toolbar button:hover { background: #1177bb; }
  #status { font-size: 12px; color: #aaa; white-space: nowrap; }

  #editor-wrap {
    flex: 1;
    display: flex;
    overflow: hidden;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.6;
  }

  #line-numbers {
    min-width: 52px;
    text-align: right;
    padding: 8px 6px 8px 0;
    background: #252526;
    color: #555;
    border-right: 1px solid #404040;
    overflow: hidden;
    user-select: none;
    flex-shrink: 0;
    white-space: pre;
  }

  #editor {
    flex: 1;
    background: #1e1e1e;
    color: #d4d4d4;
    border: none;
    outline: none;
    padding: 8px 12px;
    resize: none;
    line-height: 1.6;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    overflow-y: scroll;
    white-space: pre;
    overflow-x: auto;
  }

  #statusbar {
    padding: 3px 12px;
    background: #007acc;
    color: white;
    font-size: 12px;
    display: flex;
    gap: 20px;
    flex-shrink: 0;
  }
</style>
</head>
<body>

<div id="toolbar">
  <input type="text" id="filepath" placeholder="Path to WhisperX JSON file" value="{{ filepath }}" />
  <button onclick="loadFile()">Load</button>
  <button onclick="saveFile()">Save (Ctrl+S)</button>
  <span id="status">{{ status_msg }}</span>
</div>

<div id="editor-wrap">
  <div id="line-numbers"></div>
  <textarea id="editor" spellcheck="false" wrap="off"
    placeholder="Load a WhisperX JSON file to begin editing...">{{ content }}</textarea>
</div>

<div id="statusbar">
  <span id="sb-segments">Segments: –</span>
  <span id="sb-words">Words: –</span>
  <span id="sb-cursor">Ln 1, Col 1</span>
</div>

<script>
const editor = document.getElementById('editor');
const lineNumbers = document.getElementById('line-numbers');
const status = document.getElementById('status');
const sbSegments = document.getElementById('sb-segments');
const sbWords = document.getElementById('sb-words');
const sbCursor = document.getElementById('sb-cursor');

function updateLineNumbers() {
  const lines = editor.value.split('\n');
  lineNumbers.textContent = lines.map((_, i) => i + 1).join('\n');
  lineNumbers.scrollTop = editor.scrollTop;

  const nonEmpty = lines.filter(l => l.trim().length > 0);
  sbSegments.textContent = 'Segments: ' + nonEmpty.length;
  const words = nonEmpty.reduce(
    (acc, l) => acc + l.trim().split(/\s+/).filter(Boolean).length, 0
  );
  sbWords.textContent = 'Words: ' + words;
}

function updateCursor() {
  const val = editor.value;
  const pos = editor.selectionStart;
  const before = val.slice(0, pos);
  const ln = (before.match(/\n/g) || []).length + 1;
  const col = pos - before.lastIndexOf('\n');
  sbCursor.textContent = `Ln ${ln}, Col ${col}`;
}

editor.addEventListener('input', updateLineNumbers);
editor.addEventListener('scroll', () => { lineNumbers.scrollTop = editor.scrollTop; });
editor.addEventListener('keyup', updateCursor);
editor.addEventListener('click', updateCursor);

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveFile();
  }
});

async function loadFile() {
  const fp = document.getElementById('filepath').value.trim();
  if (!fp) return;
  status.textContent = 'Loading…';
  try {
    const resp = await fetch('/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filepath: fp})
    });
    const data = await resp.json();
    if (data.error) { status.textContent = 'Error: ' + data.error; return; }
    editor.value = data.content;
    updateLineNumbers();
    status.textContent = `Loaded ${data.segments} segments, ${data.words} words`;
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
}

async function saveFile() {
  const fp = document.getElementById('filepath').value.trim();
  if (!fp) { status.textContent = 'No file path'; return; }
  status.textContent = 'Saving…';
  try {
    const resp = await fetch('/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filepath: fp, text: editor.value})
    });
    const data = await resp.json();
    if (data.error) { status.textContent = 'Error: ' + data.error; return; }
    status.textContent = `Saved ${data.segments} segments → ${data.saved_to}`;
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
}

// Initialise
if (editor.value.trim()) updateLineNumbers();
</script>
</body>
</html>
"""


def load_whisperx(filepath):
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def segments_to_text(segments):
    lines = []
    for seg in segments:
        words = seg.get("words", [])
        if words:
            lines.append(" ".join(w["word"] for w in words))
        else:
            # Fallback: use text field if no word-level data
            lines.append(seg.get("text", "").strip())
    return "\n".join(lines)


def reconstruct_segments(original_data, new_text):
    """Rebuild segments from edited text, preserving word-level timing data."""
    all_words = [w for seg in original_data["segments"] for w in seg.get("words", [])]
    total_words = len(all_words)

    new_lines = [line for line in new_text.split("\n") if line.strip()]

    # Validate word count matches
    new_word_count = sum(len(line.split()) for line in new_lines)
    if new_word_count != total_words:
        raise ValueError(
            f"Word count mismatch: original has {total_words} words, "
            f"edited text has {new_word_count}. "
            f"Only line breaks should be changed, not the words themselves."
        )

    new_segments = []
    word_idx = 0
    for line in new_lines:
        line_words = line.split()
        count = len(line_words)
        seg_words = all_words[word_idx : word_idx + count]
        word_idx += count
        new_segments.append(
            {
                "start": seg_words[0]["start"],
                "end": seg_words[-1]["end"],
                "text": " ".join(w["word"] for w in seg_words),
                "words": seg_words,
            }
        )

    return new_segments


@app.route("/")
def index():
    filepath = sys.argv[1] if len(sys.argv) > 1 else ""
    content = ""
    status_msg = ""
    if filepath and os.path.exists(filepath):
        try:
            data = load_whisperx(filepath)
            content = segments_to_text(data["segments"])
            n_seg = len(data["segments"])
            n_words = sum(len(s.get("words", [])) for s in data["segments"])
            status_msg = f"Loaded {n_seg} segments, {n_words} words"
        except Exception as e:
            status_msg = f"Error loading {filepath}: {e}"
    return render_template_string(
        HTML, filepath=filepath, content=content, status_msg=status_msg
    )


@app.route("/load", methods=["POST"])
def load():
    req = request.json
    filepath = req.get("filepath", "").strip()
    if not os.path.exists(filepath):
        return jsonify({"error": f"File not found: {filepath}"})
    try:
        data = load_whisperx(filepath)
        segments = data["segments"]
        content = segments_to_text(segments)
        n_words = sum(len(s.get("words", [])) for s in segments)
        return jsonify(
            {"content": content, "segments": len(segments), "words": n_words}
        )
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/save", methods=["POST"])
def save():
    req = request.json
    filepath = req.get("filepath", "").strip()
    new_text = req.get("text", "")

    if not os.path.exists(filepath):
        return jsonify({"error": f"File not found: {filepath}"})

    try:
        original = load_whisperx(filepath)
        new_segments = reconstruct_segments(original, new_text)
        original["segments"] = new_segments

        base, ext = os.path.splitext(filepath)
        out_path = base + ".corrected" + ext
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)

        return jsonify(
            {"status": "ok", "saved_to": out_path, "segments": len(new_segments)}
        )
    except ValueError as e:
        return jsonify({"error": str(e)})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Opening http://127.0.0.1:{port}")
    app.run(debug=False, port=port)
