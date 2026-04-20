"""
align.py

Book-oriented wrapper around the shared alignment engine in align_whisper.
This keeps the old align-html workflow intact, but removes the separate
alignment implementation that had diverged from align_whisper.

Usage:
    # Single chapter
    python align.py \
        --whisperx chapter1.json \
        --text     chapter1.txt  \
        --out      chapter1_aligned.json

    # Whole book via config
    python align.py --config book_config.yaml --outdir aligned/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from align_to_json import align_file_to_sentences

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


def sentence_split_regex(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def sentence_split_spacy(text: str, nlp) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def get_sentence_splitter():
    if _SPACY_AVAILABLE:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                nlp = spacy.load(model, disable=["ner", "parser"])
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                print(f"  Using spaCy model: {model}", file=sys.stderr)
                return lambda t: sentence_split_spacy(t, nlp)
            except OSError:
                continue
    print("  spaCy not available or no model found — using regex splitter.", file=sys.stderr)
    return sentence_split_regex


def load_sentence_list(text_path: Path, split_fn) -> tuple[list[list[str]], list[str]]:
    text = text_path.read_text(encoding="utf-8")
    chunks = split_fn(text)
    sentence_list = [chunk.split() for chunk in chunks if chunk.split()]
    sentence_nums = [str(i + 1) for i in range(len(sentence_list))]
    return sentence_list, sentence_nums


def align_chapter(
    whisperx_path: Path,
    text_path: Path,
    split_fn,
    *,
    hyp_format: str = "auto",
    hyp2_path: str | Path | None = None,
    hyp2_format: str = "auto",
    speaker: str | None = None,
    gender: str | None = None,
    validator: str | None = None,
    secondary_threshold: float = 1.0,
    normalizations: str | None = None,
    eps_symbol: str = "<eps>",
    correct_score: int = 1,
    substitution_penalty: int = 1,
    deletion_penalty: int = 1,
    insertion_penalty: int = 1,
    align_full_hyp: bool = True,
    normalize: bool = True,
) -> list[dict]:
    sentence_list, sentence_nums = load_sentence_list(text_path, split_fn)
    _file_id, sentence_objects, _new_norms = align_file_to_sentences(
        whisperx_path,
        sentence_list,
        sentence_nums,
        hyp_format=hyp_format,
        hyp2_path=hyp2_path,
        hyp2_format=hyp2_format,
        speaker=speaker,
        gender=gender,
        validator=validator,
        secondary_threshold=secondary_threshold,
        normalizations_path=normalizations,
        eps_symbol=eps_symbol,
        correct_score=correct_score,
        substitution_penalty=substitution_penalty,
        deletion_penalty=deletion_penalty,
        insertion_penalty=insertion_penalty,
        align_full_hyp=align_full_hyp,
        normalize=normalize,
        include_word_details=True,
    )
    return sentence_objects


def to_legacy_chunks(sentence_objects: list[dict]) -> list[dict]:
    legacy = []
    for i, sentence in enumerate(sentence_objects):
        legacy.append({
            "chunk_id": i,
            "text": sentence.get("reference", ""),
            "start": sentence.get("start"),
            "end": sentence.get("end"),
            "words": sentence.get("asr_word_details", []),
        })
    return legacy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align WhisperX JSON to reference text.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--whisperx", help="Path to WhisperX JSON (single chapter)")
    group.add_argument("--config", help="book_config.yaml (whole book)")
    parser.add_argument("--text", help="Reference text file (single chapter mode)")
    parser.add_argument("--out", help="Output JSON path (single chapter mode)")
    parser.add_argument("--outdir", default="aligned", help="Output dir (config mode)")
    parser.add_argument(
        "--output-format",
        choices=["legacy", "sentences"],
        default="legacy",
        help="legacy reproduces the old align-html chunk schema; sentences writes the richer align_whisper schema.",
    )
    parser.add_argument("--hyp-format", choices=["whisperx", "hfjson", "vv", "auto"], default="auto")
    parser.add_argument("--hyp2", help="Optional secondary hypothesis JSON")
    parser.add_argument("--hyp2-format", choices=["whisperx", "hfjson", "vv", "auto"], default="auto")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--secondary-validator", default=None)
    parser.add_argument("--secondary-threshold", type=float, default=1.0)
    parser.add_argument("--normalizations", default=None)
    parser.add_argument("--eps-symbol", default="<eps>")
    parser.add_argument("--correct-score", type=int, default=1)
    parser.add_argument("--substitution-penalty", type=int, default=1)
    parser.add_argument("--deletion-penalty", type=int, default=1)
    parser.add_argument("--insertion-penalty", type=int, default=1)
    parser.add_argument("--no-align-full-hyp", dest="align_full_hyp", action="store_false", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    return parser


def main():
    parser = build_parser()
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
            hyp_format=args.hyp_format,
            hyp2_path=args.hyp2,
            hyp2_format=args.hyp2_format,
            speaker=args.speaker,
            gender=args.gender,
            validator=args.secondary_validator,
            secondary_threshold=args.secondary_threshold,
            normalizations=args.normalizations,
            eps_symbol=args.eps_symbol,
            correct_score=args.correct_score,
            substitution_penalty=args.substitution_penalty,
            deletion_penalty=args.deletion_penalty,
            insertion_penalty=args.insertion_penalty,
            align_full_hyp=args.align_full_hyp,
            normalize=args.normalize,
        )
        if args.output_format == "legacy":
            result = to_legacy_chunks(result)
        out_path = Path(args.out) if args.out else Path("aligned.json")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Written: {out_path}  ({len(result)} chunks)")
        return

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
            hyp2_path = ch.get("hyp2_json") or args.hyp2
            result = align_chapter(
                wx_path,
                txt_path,
                split_fn,
                hyp_format=ch.get("hyp_format", args.hyp_format),
                hyp2_path=hyp2_path,
                hyp2_format=ch.get("hyp2_format", args.hyp2_format),
                speaker=ch.get("speaker", args.speaker),
                gender=ch.get("gender", args.gender),
                validator=ch.get("secondary_validator", args.secondary_validator),
                secondary_threshold=ch.get("secondary_threshold", args.secondary_threshold),
                normalizations=ch.get("normalizations", args.normalizations),
                eps_symbol=args.eps_symbol,
                correct_score=args.correct_score,
                substitution_penalty=args.substitution_penalty,
                deletion_penalty=args.deletion_penalty,
                insertion_penalty=args.insertion_penalty,
                align_full_hyp=args.align_full_hyp,
                normalize=args.normalize,
            )
            output_format = ch.get("output_format", args.output_format)
            if output_format == "legacy":
                result = to_legacy_chunks(result)
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
