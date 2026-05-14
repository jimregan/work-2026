#!/usr/bin/env python3
"""
Calculate Phoneme Error Rate (PER) for G2P model outputs.

PER is computed as WER over space-separated phoneme sequences (each phoneme
treated as a "word" by the jiwer WER metric).

Reference files are in RETSV format: word<TAB>phones
Model output files are also in word<TAB>phones format, optionally prefixed
with a superscript rank character (¹²³⁴⁵⁶) produced by some Phonetisaurus
nbest outputs.
"""

import argparse
import sys
from pathlib import Path

SUPERSCRIPTS = set("¹²³⁴⁵⁶⁷⁸⁹⁰")


def load_retsv(path: Path) -> dict[str, str]:
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            result[parts[0]] = parts[1]
    return result


def load_output(path: Path) -> dict[str, str]:
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] in SUPERSCRIPTS:
                line = line[1:]
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            result[parts[0]] = parts[1]
    return result


def per(refs: dict[str, str], hyps: dict[str, str]) -> float:
    try:
        from jiwer import wer
    except ImportError:
        print("jiwer is required: pip install jiwer", file=sys.stderr)
        sys.exit(1)

    ref_seqs, hyp_seqs = [], []
    for word in refs:
        if word not in hyps:
            continue
        ref_seqs.append(refs[word])
        hyp_seqs.append(hyps[word])

    if not ref_seqs:
        return float("nan")
    return wer(ref_seqs, hyp_seqs)


def main():
    parser = argparse.ArgumentParser(description="Compute PER for G2P model outputs")
    parser.add_argument("--ref-dir", type=Path, required=True,
                        help="Directory containing {lang}.test.retsv reference files")
    parser.add_argument("--hyp-dir", type=Path, required=True,
                        help="Directory containing model output files")
    parser.add_argument("--langs", nargs="+",
                        default=["swe", "dan", "eng", "spa", "lat", "fre", "ita"],
                        help="Language codes to evaluate")
    parser.add_argument("--models", nargs="+",
                        default=["wl", "mwl", "wl-merged", "wl-raw"],
                        help="Model suffixes to evaluate (files: {lang}.{model}.out)")
    parser.add_argument("--tsv", action="store_true",
                        help="Output tab-separated values instead of aligned text")
    args = parser.parse_args()

    results: dict[tuple[str, str], float] = {}

    for lang in args.langs:
        ref_path = args.ref_dir / f"{lang}.test.retsv"
        if not ref_path.exists():
            print(f"  skip {lang}: {ref_path} not found", file=sys.stderr)
            continue
        refs = load_retsv(ref_path)

        for model in args.models:
            hyp_path = args.hyp_dir / f"{lang}.{model}.out"
            if not hyp_path.exists():
                continue
            hyps = load_output(hyp_path)
            score = per(refs, hyps)
            results[(lang, model)] = score

    if args.tsv:
        print("lang\tmodel\tPER")
        for (lang, model), score in results.items():
            print(f"{lang}\t{model}\t{score:.4f}")
    else:
        col_w = max((len(m) for m in args.models), default=8)
        header = f"{'lang':<6}" + "".join(f"  {m:>{col_w}}" for m in args.models)
        print(header)
        for lang in args.langs:
            row = f"{lang:<6}"
            for model in args.models:
                if (lang, model) in results:
                    row += f"  {results[(lang, model)]:>{col_w}.4f}"
                else:
                    row += f"  {'--':>{col_w}}"
            print(row)


if __name__ == "__main__":
    main()
