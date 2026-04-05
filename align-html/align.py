"""
align.py
Aligns WhisperX word-level JSON output to a clean reference text,
producing sentence/chunk-level segments with timestamps.

The alignment strategy:
  1. Tokenise both the WhisperX transcript and the reference text into words.
  2. Use a sliding-window fuzzy match (difflib SequenceMatcher) to find the
     best correspondence between the ASR stream and the reference.
  3. Split the reference into sentence chunks (spaCy or simple regex fallback).
  4. For each sentence chunk, find the first/last ASR word that matches and
     record start/end timestamps.

Output (JSON per chapter):
  [
    {
      "chunk_id": 0,
      "text": "The quick brown fox ...",
      "start": 1.24,
      "end": 4.80,
      "words": [{"word": "The", "start": 1.24, "end": 1.40}, ...]
    },
    ...
  ]

Usage:
    # Single chapter
    python align.py \\
        --whisperx chapter1.json \\
        --text     chapter1.txt  \\
        --out      chapter1_aligned.json

    # Whole book via config
    python align.py --config book_config.yaml --outdir aligned/
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import yaml

# Optional: spaCy for better sentence splitting
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


# ── Text utilities ────────────────────────────────────────────────────────────

def normalise(word: str) -> str:
    """Lowercase, strip punctuation for comparison."""
    return re.sub(r"[^\w']", "", word).lower()


def tokenise(text: str) -> list[str]:
    return text.split()


def sentence_split_regex(text: str) -> list[str]:
    """Simple regex sentence splitter fallback."""
    chunks = re.split(r'(?<=[.!?])\s+', text.strip())
    return [c.strip() for c in chunks if c.strip()]


def sentence_split_spacy(text: str, nlp) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def get_sentence_splitter():
    if _SPACY_AVAILABLE:
        # Try to load a small model; fall back to regex if unavailable
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                nlp = spacy.load(model, disable=["ner", "parser"])
                nlp.add_pipe("sentencizer")
                print(f"  Using spaCy model: {model}", file=sys.stderr)
                return lambda t: sentence_split_spacy(t, nlp)
            except OSError:
                continue
    print("  spaCy not available or no model found — using regex splitter.", file=sys.stderr)
    return sentence_split_regex


# ── WhisperX JSON loading ─────────────────────────────────────────────────────

def load_whisperx(path: Path) -> list[dict]:
    """
    Load WhisperX JSON. Returns a flat list of word dicts:
      {"word": str, "start": float, "end": float}
    WhisperX stores words under segments[].words[].
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    words = []
    segments = data if isinstance(data, list) else data.get("segments", [])
    for seg in segments:
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            if not word:
                continue
            words.append({
                "word": word,
                "start": w.get("start"),
                "end": w.get("end"),
            })
    return words


# ── Fuzzy alignment ───────────────────────────────────────────────────────────

def build_norm_index(words: list[dict]) -> list[str]:
    return [normalise(w["word"]) for w in words]


def align_sequences(ref_tokens: list[str], asr_norm: list[str]) -> list[int | None]:
    """
    For each ref_token, return the index into asr_norm that best matches it,
    or None if no match found.

    Uses SequenceMatcher for global alignment; maps ref positions → asr positions.
    """
    ref_norm = [normalise(t) for t in ref_tokens]
    sm = difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False)
    # Build asr_idx → ref_idx mapping
    asr_to_ref = {}
    for block in sm.get_matching_blocks():
        asr_start, ref_start, size = block
        for k in range(size):
            asr_to_ref[asr_start + k] = ref_start + k

    # Invert: ref_idx → list of asr_idx
    ref_to_asr: dict[int, list[int]] = {}
    for ai, ri in asr_to_ref.items():
        ref_to_asr.setdefault(ri, []).append(ai)

    # For each ref token pick the best (earliest) asr match
    mapping: list[int | None] = []
    for ri in range(len(ref_norm)):
        candidates = ref_to_asr.get(ri)
        mapping.append(min(candidates) if candidates else None)

    return mapping


# ── Chunk timestamping ────────────────────────────────────────────────────────

def timestamp_chunks(
    chunks: list[str],
    asr_words: list[dict],
    mapping: list[int | None],
) -> list[dict]:
    """
    Given sentence chunks, a flat list of ref tokens (words), and their
    mapping to ASR word indices, produce timestamped chunk records.
    """
    # Build the ref token list with chunk membership
    ref_tokens = []
    chunk_ids = []
    for ci, chunk in enumerate(chunks):
        toks = tokenise(chunk)
        ref_tokens.extend(toks)
        chunk_ids.extend([ci] * len(toks))

    assert len(ref_tokens) == len(mapping)

    results = []
    for ci, chunk in enumerate(chunks):
        # Collect ASR indices for this chunk
        asr_indices = [
            mapping[ri]
            for ri, cid in enumerate(chunk_ids)
            if cid == ci and mapping[ri] is not None
        ]

        if not asr_indices:
            results.append({
                "chunk_id": ci,
                "text": chunk,
                "start": None,
                "end": None,
                "words": [],
                "warning": "no_asr_match",
            })
            continue

        first_ai = min(asr_indices)
        last_ai = max(asr_indices)

        chunk_asr_words = [
            asr_words[i]
            for i in range(first_ai, last_ai + 1)
            if asr_words[i]["start"] is not None
        ]

        start_t = asr_words[first_ai].get("start")
        end_t = asr_words[last_ai].get("end")

        results.append({
            "chunk_id": ci,
            "text": chunk,
            "start": start_t,
            "end": end_t,
            "words": chunk_asr_words,
        })

    return results


# ── Main alignment function ───────────────────────────────────────────────────

def align_chapter(
    whisperx_path: Path,
    text_path: Path,
    split_fn,
) -> list[dict]:
    print(f"  Loading ASR: {whisperx_path}", file=sys.stderr)
    asr_words = load_whisperx(whisperx_path)

    print(f"  Loading text: {text_path}", file=sys.stderr)
    ref_text = text_path.read_text(encoding="utf-8")

    print(f"  Splitting into sentences…", file=sys.stderr)
    chunks = split_fn(ref_text)
    print(f"  {len(chunks)} chunks, {len(asr_words)} ASR words", file=sys.stderr)

    ref_tokens = tokenise(ref_text)
    asr_norm = build_norm_index(asr_words)

    print(f"  Running sequence alignment…", file=sys.stderr)
    mapping = align_sequences(ref_tokens, asr_norm)

    matched = sum(1 for m in mapping if m is not None)
    print(f"  Matched {matched}/{len(ref_tokens)} ref tokens ({100*matched//max(len(ref_tokens),1)}%)",
          file=sys.stderr)

    return timestamp_chunks(chunks, asr_words, mapping)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Align WhisperX JSON to reference text.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--whisperx", help="Path to WhisperX JSON (single chapter)")
    group.add_argument("--config", help="book_config.yaml (whole book)")
    parser.add_argument("--text", help="Reference text file (single chapter mode)")
    parser.add_argument("--out", help="Output JSON path (single chapter mode)")
    parser.add_argument("--outdir", default="aligned", help="Output dir (config mode)")
    args = parser.parse_args()

    split_fn = get_sentence_splitter()

    if args.whisperx:
        if not args.text:
            print("ERROR: --text required in single-chapter mode", file=sys.stderr)
            sys.exit(1)
        result = align_chapter(
            Path(args.whisperx),
            Path(args.text),
            split_fn,
        )
        out_path = Path(args.out) if args.out else Path("aligned.json")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Written: {out_path}  ({len(result)} chunks)")

    else:
        config_path = Path(args.config)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        chapters = config.get("chapters", [])
        for i, ch in enumerate(chapters):
            label = ch.get("chapter", f"chapter_{i+1}")
            wx_path = Path(ch.get("whisperx_json", ""))
            txt_path = Path(ch.get("text_file", ""))

            missing = []
            if not wx_path.exists():
                missing.append(f"whisperx_json={wx_path}")
            if not txt_path.exists():
                missing.append(f"text_file={txt_path}")
            if missing:
                print(f"[{i+1}/{len(chapters)}] Skipping '{label}' — missing: {', '.join(missing)}")
                continue

            print(f"\n[{i+1}/{len(chapters)}] Aligning '{label}'")
            slug = re.sub(r"[^\w]+", "_", label)[:50]
            out_path = outdir / f"{slug}_aligned.json"
            try:
                result = align_chapter(wx_path, txt_path, split_fn)
                out_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                ch["aligned_json"] = str(out_path)
                print(f"  → {out_path}  ({len(result)} chunks)")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)

        config_path.write_text(
            yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"\nConfig updated: {config_path}")


if __name__ == "__main__":
    main()
