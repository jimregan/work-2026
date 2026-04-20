#!/usr/bin/env python3
"""
Preprocessing script for seanchló OCR training data.
Reads labels.txt, prints character inventory, and splits into train/val/test.
"""

import argparse
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess seanchló OCR data")
    parser.add_argument("labels", help="Path to labels.txt")
    parser.add_argument("--outdir", default=".", help="Output directory for split files")
    parser.add_argument("--train", type=float, default=0.9, help="Train fraction (default 0.9)")
    parser.add_argument("--val", type=float, default=0.05, help="Val fraction (default 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read labels
    lines = []
    with open(labels_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                print(f"WARNING: line {lineno} has no transcription, skipping: {line!r}")
                continue
            filename, transcription = parts
            lines.append((filename, transcription))

    print(f"Read {len(lines)} lines from {labels_path}\n")

    # Character inventory
    char_counts = Counter()
    for _, transcription in lines:
        char_counts.update(transcription)

    print("Character inventory:")
    print(f"  Total unique characters: {len(char_counts)}")
    print()

    # Group by rough category for readability
    categories = {
        "ASCII letters": [],
        "Dotted consonants": [],
        "Accented vowels": [],
        "Insular letters": [],
        "Digits": [],
        "Punctuation / other": [],
    }

    insular = set("ꝺꝼᵹꞇꞃꞅ")
    dotted = set("ḃḂċĊḋḊḟḞġĠṁṀṗṖṡṠṫṪ")
    accented = set("áÁéÉíÍóÓúÚ")

    for ch, count in sorted(char_counts.items(), key=lambda x: x[0]):
        if ch in insular:
            categories["Insular letters"].append((ch, count))
        elif ch in dotted:
            categories["Dotted consonants"].append((ch, count))
        elif ch in accented:
            categories["Accented vowels"].append((ch, count))
        elif ch.isascii() and ch.isalpha():
            categories["ASCII letters"].append((ch, count))
        elif ch.isdigit():
            categories["Digits"].append((ch, count))
        else:
            categories["Punctuation / other"].append((ch, count))

    for category, chars in categories.items():
        if chars:
            print(f"  {category}:")
            for ch, count in chars:
                print(f"    U+{ord(ch):04X}  {ch}  {count:6d}")
            print()

    # Check for unexpected characters
    unexpected = [
        (ch, count) for ch, count in char_counts.items()
        if not any((ch, count) in v for v in categories.values())
    ]
    if unexpected:
        print("  Uncategorised characters:")
        for ch, count in unexpected:
            print(f"    U+{ord(ch):04X}  {ch!r}  {count:6d}")
        print()

    # Shuffle and split
    random.seed(args.seed)
    shuffled = lines[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    print("Split sizes:")
    for name, data in splits.items():
        path = outdir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for filename, transcription in data:
                f.write(f"{filename} {transcription}\n")
        print(f"  {name:6s}: {len(data):5d} lines -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
