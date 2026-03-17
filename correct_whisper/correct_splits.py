#!/usr/bin/env python3
"""WhisperX segment split corrector.

Each word is a <span> carrying its timing data.  Segments are <div> lines.
Edit exactly like a text editor:

  Backspace across a space    → merge the two adjacent words
  Backspace at line start     → merge this segment with the previous one
  Space inside a word         → split the word at cursor (timing interpolated)
  Enter                       → split segment at the nearest word boundary

On save, segments are rebuilt:
  text  = " ".join(w["word"] for w in words)
  start = words[0]["start"]
  end   = words[-1]["end"]

Usage:
    python correct_splits.py [path/to/file.json]
    # open http://127.0.0.1:5000
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
  background: #1e1e1e;
  color: #d4d4d4;
  font-family: sans-serif;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

/* ── toolbar ── */
#toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #2d2d2d;
  border-bottom: 1px solid #444;
  flex-shrink: 0;
}
#filepath {
  flex: 1;
  background: #3c3c3c;
  color: #d4d4d4;
  border: 1px solid #555;
  padding: 4px 8px;
  font-size: 13px;
  border-radius: 3px;
}
button {
  padding: 4px 14px;
  cursor: pointer;
  background: #0e639c;
  color: white;
  border: none;
  border-radius: 3px;
  font-size: 13px;
  white-space: nowrap;
}
button:hover { background: #1177bb; }
#status { font-size: 12px; color: #aaa; }

/* ── editor ── */
#editor-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px 40px;
}

#editor {
  outline: none;
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 16px;
  line-height: 2;
  max-width: 820px;
  margin: 0 auto;
  caret-color: #aeafad;
}

/* Each segment is a block with a subtle left rule */
.seg {
  display: block;
  padding: 1px 8px;
  border-left: 3px solid transparent;
  margin-bottom: 2px;
  min-height: 1.2em;
}
.seg:focus-within { border-left-color: #007acc; }

/* Each word carries timing as data attributes */
.w {
  display: inline;
  border-radius: 2px;
  padding: 0 1px;
}
.w:hover { background: rgba(255,255,255,0.08); }

/* ── tooltip ── */
#tip {
  position: fixed;
  background: #252526;
  border: 1px solid #454545;
  color: #9cdcfe;
  font-size: 11px;
  font-family: monospace;
  padding: 3px 7px;
  border-radius: 3px;
  pointer-events: none;
  display: none;
  z-index: 1000;
}

/* ── status bar ── */
#statusbar {
  padding: 3px 12px;
  background: #007acc;
  color: white;
  font-size: 12px;
  display: flex;
  gap: 24px;
  flex-shrink: 0;
}
</style>
</head>
<body>

<div id="toolbar">
  <input type="text" id="filepath" placeholder="Path to WhisperX JSON file" />
  <button onclick="loadFile()">Load</button>
  <button onclick="saveFile()">Save (Ctrl+S)</button>
  <span id="status"></span>
</div>

<div id="editor-container">
  <div id="editor" contenteditable="true" spellcheck="false"></div>
</div>

<div id="tip"></div>

<div id="statusbar">
  <span id="sb-segs">Segments: –</span>
  <span id="sb-words">Words: –</span>
</div>

<script>
'use strict';

const editor    = document.getElementById('editor');
const statusEl  = document.getElementById('status');
const tip       = document.getElementById('tip');
const sbSegs    = document.getElementById('sb-segs');
const sbWords   = document.getElementById('sb-words');

// ── Rendering ──────────────────────────────────────────────────────────────

function makeWordSpan(w) {
  const s = document.createElement('span');
  s.className = 'w';
  s.dataset.start = w.start;
  s.dataset.end   = w.end;
  if (w.score != null) s.dataset.score = w.score;
  s.textContent = w.word;
  return s;
}

function makeSegDiv(words) {
  const div = document.createElement('div');
  div.className = 'seg';
  words.forEach((w, i) => {
    if (i > 0) div.appendChild(document.createTextNode(' '));
    div.appendChild(makeWordSpan(w));
  });
  return div;
}

function renderAll(segments) {
  editor.innerHTML = '';
  segments.forEach(words => editor.appendChild(makeSegDiv(words)));
  updateStatusBar();
}

// ── Tooltip ────────────────────────────────────────────────────────────────

editor.addEventListener('mouseover', e => {
  const w = e.target.closest('.w');
  if (!w) { tip.style.display = 'none'; return; }
  const s = (+w.dataset.start).toFixed(3);
  const en = (+w.dataset.end).toFixed(3);
  tip.textContent = `${s} – ${en}`;
  tip.style.display = 'block';
});
editor.addEventListener('mousemove', e => {
  tip.style.left = (e.clientX + 14) + 'px';
  tip.style.top  = (e.clientY - 28) + 'px';
});
editor.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

// ── DOM helpers ────────────────────────────────────────────────────────────

function containingSeg(node) {
  let el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  while (el && el !== editor) {
    if (el.classList?.contains('seg')) return el;
    el = el.parentElement;
  }
  return null;
}

function containingWord(node) {
  let el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
  while (el && el !== editor) {
    if (el.classList?.contains('w')) return el;
    el = el.parentElement;
  }
  return null;
}

/** First text node in el, or null. */
function firstText(el) {
  const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  return w.nextNode();
}

/** Last text node in el, or null. */
function lastText(el) {
  const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let n, last = null;
  while ((n = w.nextNode())) last = n;
  return last;
}

function setCursor(node, offset) {
  const sel = window.getSelection();
  const r = document.createRange();
  r.setStart(node, offset);
  r.collapse(true);
  sel.removeAllRanges();
  sel.addRange(r);
}

/** True when the caret is at the very beginning of seg. */
function atSegStart(seg, anchor, offset) {
  if (offset !== 0) return false;
  const ft = firstText(seg);
  return !ft || anchor === ft || anchor === seg;
}

/** Trim whitespace-only text nodes from both ends of el. */
function trimWhitespace(el) {
  while (el.firstChild?.nodeType === Node.TEXT_NODE &&
         el.firstChild.textContent.trim() === '')
    el.firstChild.remove();
  while (el.lastChild?.nodeType === Node.TEXT_NODE &&
         el.lastChild.textContent.trim() === '')
    el.lastChild.remove();
}

// ── Enter: split segment ───────────────────────────────────────────────────

function handleEnter(anchor, offset) {
  const seg = containingSeg(anchor);
  if (!seg) return;

  const wordSpans = Array.from(seg.querySelectorAll('.w'));
  if (!wordSpans.length) return;

  // Find how many word spans come entirely before the cursor.
  const sel = window.getSelection();
  const curRange = sel.getRangeAt(0);
  let splitIdx = wordSpans.length; // default: cursor after last word

  for (let i = 0; i < wordSpans.length; i++) {
    const wr = document.createRange();
    wr.selectNodeContents(wordSpans[i]);
    // If cursor is before the start of this word, split here
    if (curRange.compareBoundaryPoints(Range.START_TO_START, wr) <= 0) {
      splitIdx = i;
      break;
    }
  }

  // Build new segment from wordSpans[splitIdx:]
  const newSeg = document.createElement('div');
  newSeg.className = 'seg';
  for (let i = splitIdx; i < wordSpans.length; i++) {
    if (newSeg.lastChild) newSeg.appendChild(document.createTextNode(' '));
    newSeg.appendChild(wordSpans[i]);
  }

  // Clean up trailing whitespace in current seg
  trimWhitespace(seg);

  seg.parentNode.insertBefore(newSeg, seg.nextSibling);

  // Place cursor at start of new segment
  const ft = firstText(newSeg);
  if (ft) setCursor(ft, 0);
  else setCursor(newSeg, 0);

  updateStatusBar();
}

// ── Backspace at seg start: merge with previous segment ───────────────────

function mergeWithPrev(seg) {
  const prev = seg.previousElementSibling;
  if (!prev?.classList.contains('seg')) return;

  // Remember where to leave the cursor (end of prev content)
  const lt = lastText(prev);
  const cursorNode   = lt ?? prev;
  const cursorOffset = lt ? lt.textContent.length : prev.childNodes.length;

  // Join with a space
  if (prev.lastChild && seg.firstChild)
    prev.appendChild(document.createTextNode(' '));

  while (seg.firstChild) prev.appendChild(seg.firstChild);
  seg.remove();

  setCursor(cursorNode, cursorOffset);
  updateStatusBar();
}

// ── Space inside a word: split word at cursor ─────────────────────────────

function splitWordAtCursor(wordSpan, anchor, offset) {
  const full  = anchor.textContent;          // text node inside wordSpan
  const left  = full.slice(0, offset);
  const right = full.slice(offset);

  const start = +wordSpan.dataset.start;
  const end   = +wordSpan.dataset.end;
  const score = wordSpan.dataset.score != null ? +wordSpan.dataset.score : null;
  const frac  = offset / full.length;
  const mid   = +(start + frac * (end - start)).toFixed(3);

  // Update left span
  wordSpan.textContent = left;
  wordSpan.dataset.end = mid;

  // Build right span
  const r = document.createElement('span');
  r.className = 'w';
  r.dataset.start = mid;
  r.dataset.end   = end;
  if (score != null) r.dataset.score = score;
  r.textContent = right;

  wordSpan.after(document.createTextNode(' '), r);

  // Place cursor at start of right word
  if (r.firstChild) setCursor(r.firstChild, 0);
  else setCursor(r, 0);
}

// ── input: merge adjacent .w spans (space between them was deleted) ────────

editor.addEventListener('input', () => {
  // Look for two .w siblings with no text node between them → merge
  let again = true;
  while (again) {
    again = false;
    for (const seg of editor.querySelectorAll('.seg')) {
      for (const w of seg.querySelectorAll('.w')) {
        const nxt = w.nextSibling;
        if (nxt?.nodeType === Node.ELEMENT_NODE && nxt.classList.contains('w')) {
          // Adjacent word spans — merge timing and text
          w.dataset.end = nxt.dataset.end;
          w.textContent += nxt.textContent;
          nxt.remove();
          again = true;
          break;
        }
      }
      if (again) break;
    }
  }
  updateStatusBar();
});

// ── keydown: intercept structural operations ───────────────────────────────

editor.addEventListener('keydown', e => {
  const sel = window.getSelection();
  if (!sel?.isCollapsed) return;

  const anchor = sel.anchorNode;
  const offset = sel.anchorOffset;

  if (e.key === 'Enter') {
    e.preventDefault();
    handleEnter(anchor, offset);

  } else if (e.key === 'Backspace') {
    const seg = containingSeg(anchor);
    if (seg && atSegStart(seg, anchor, offset)) {
      e.preventDefault();
      mergeWithPrev(seg);
    }
    // else: browser handles backspace (char delete or space delete → 'input' fires)

  } else if (e.key === ' ') {
    const wSpan = containingWord(anchor);
    if (wSpan && anchor.nodeType === Node.TEXT_NODE &&
        offset > 0 && offset < anchor.textContent.length) {
      // Cursor is inside word text, not at an edge
      e.preventDefault();
      splitWordAtCursor(wSpan, anchor, offset);
    }
    // else: browser inserts a space (between words → fine, or at edge → fine)
  }
});

// ── Save ───────────────────────────────────────────────────────────────────

function extractSegments() {
  const result = [];
  for (const seg of editor.querySelectorAll('.seg')) {
    const words = [];
    for (const w of seg.querySelectorAll('.w')) {
      const text = w.textContent;
      if (!text) continue;
      const obj = { word: text, start: +w.dataset.start, end: +w.dataset.end };
      if (w.dataset.score != null) obj.score = +w.dataset.score;
      words.push(obj);
    }
    if (words.length) result.push(words);
  }
  return result;
}

async function saveFile() {
  const fp = document.getElementById('filepath').value.trim();
  if (!fp) { statusEl.textContent = 'No file loaded'; return; }
  statusEl.textContent = 'Saving…';
  try {
    const resp = await fetch('/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filepath: fp, segments: extractSegments() })
    });
    const data = await resp.json();
    if (data.error) { statusEl.textContent = 'Error: ' + data.error; return; }
    statusEl.textContent = `Saved ${data.segments} segments → ${data.saved_to}`;
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

// ── Load ───────────────────────────────────────────────────────────────────

async function loadFile() {
  const fp = document.getElementById('filepath').value.trim();
  if (!fp) return;
  statusEl.textContent = 'Loading…';
  try {
    const resp = await fetch('/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filepath: fp })
    });
    const data = await resp.json();
    if (data.error) { statusEl.textContent = 'Error: ' + data.error; return; }
    renderAll(data.segments);
    statusEl.textContent =
      `Loaded ${data.segments.length} segments, ${data.words} words`;
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
}

// ── Status bar ─────────────────────────────────────────────────────────────

function updateStatusBar() {
  sbSegs.textContent  = 'Segments: ' + editor.querySelectorAll('.seg').length;
  sbWords.textContent = 'Words: '    + editor.querySelectorAll('.w').length;
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveFile();
  }
});

// ── Init ───────────────────────────────────────────────────────────────────

(function init() {
  const fp = INITIAL_FILEPATH;
  if (fp) {
    document.getElementById('filepath').value = fp;
    loadFile();
  }
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    filepath = sys.argv[1] if len(sys.argv) > 1 else ""
    # Inject filepath as a JS literal so quotes/backslashes are safe
    html = HTML.replace("INITIAL_FILEPATH", json.dumps(filepath))
    return html


@app.route("/load", methods=["POST"])
def load():
    fp = request.json.get("filepath", "").strip()
    if not os.path.exists(fp):
        return jsonify({"error": f"File not found: {fp}"})
    try:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        segments = [s.get("words", []) for s in data["segments"] if s.get("words")]
        n_words = sum(len(s) for s in segments)
        return jsonify({"segments": segments, "words": n_words})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/save", methods=["POST"])
def save():
    req = request.json
    fp = req.get("filepath", "").strip()
    new_segs_data = req.get("segments", [])

    if not os.path.exists(fp):
        return jsonify({"error": f"File not found: {fp}"})
    try:
        with open(fp, encoding="utf-8") as f:
            original = json.load(f)

        new_segments = []
        for words in new_segs_data:
            if not words:
                continue
            new_segments.append(
                {
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "text": " ".join(w["word"] for w in words),
                    "words": words,
                }
            )

        original["segments"] = new_segments

        base, ext = os.path.splitext(fp)
        out = base + ".corrected" + ext
        with open(out, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2, ensure_ascii=False)

        return jsonify({"status": "ok", "saved_to": out, "segments": len(new_segments)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)
