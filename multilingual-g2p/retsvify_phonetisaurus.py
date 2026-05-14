#!/usr/bin/env python3
"""
Convert Phonetisaurus-format output to RETSV (word<TAB>phones) format.

Phonetisaurus alignment format uses '}' to separate the grapheme and phoneme
parts of each chunk, '|' to join multi-character graphemes/phones, and '_'
for null graphemes/phones (epsilon transitions).
"""

import argparse
import sys
from pathlib import Path


def retsvify_line(line: str) -> tuple[str, str]:
    parts = line.strip().split(" ")
    pieces = [p.split("}") for p in parts]
    word = "".join(p[0].replace("|", "") for p in pieces if p[0] != "_")
    phones = " ".join(p[1].replace("|", " ") for p in pieces if p[1] != "_")
    return word, phones


def convert(in_path: Path, out_path: Path) -> int:
    count = 0
    with open(in_path, encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            word, phones = retsvify_line(line)
            f_out.write(f"{word}\t{phones}\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert Phonetisaurus output to RETSV format")
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Phonetisaurus output file(s)")
    parser.add_argument("--suffix", default=".retsv",
                        help="Suffix appended to each input filename (default: .retsv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write output files here instead of alongside inputs")
    args = parser.parse_args()

    for in_path in args.inputs:
        if args.output_dir is not None:
            out_path = args.output_dir / (in_path.name + args.suffix)
        else:
            out_path = in_path.parent / (in_path.name + args.suffix)
        n = convert(in_path, out_path)
        print(f"{in_path} -> {out_path} ({n} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
