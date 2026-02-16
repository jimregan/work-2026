#!/usr/bin/env python3
"""
Unified multilingual G2P pipeline.

Stages:
1. Load: Read braxen-sv.tsv, parse ID/word/transcript/language fields
2. Split by language: Produce per-language word/transcript pairs, filter numerics
3. Validate: Hunspell spell-check per language (Nordic normalization for lookup only)
4. Filter: Apply thresholds, keep Hunspell-OK words
5. Train/test split: Consistent, reproducible splits
6. Train models: Phonetisaurus (WL, MWL, RAW), with/without accent markers
7. Evaluate: PER calculation against held-out test sets

Language tag corrections are handled separately; this pipeline takes a
(potentially corrected) braxen-sv.tsv as input.
"""

import argparse
import csv
import os
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class BraxenEntry:
    """A single entry from the Braxen lexicon."""
    entry_id: str
    word: str
    transcript: str
    language: str
    pos: str = ""
    hunspell_ok: Optional[bool] = None
    hunspell_suggestions: list = field(default_factory=list)


def load_braxen(braxen_path: Path) -> list[BraxenEntry]:
    """
    Load braxen-sv.tsv, skipping comment lines.

    Braxen TSV format (tab-separated):
    word, phones, POS, language, [many placeholder fields], ID

    Example:
    $antal  d 'o . l a r | "a n . t ,a: l   NN  swe  -  -  ...  -  732346
    """
    entries = []
    with open(braxen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            word = parts[0]
            transcript = parts[1]
            pos = parts[2] if len(parts) > 2 else ""
            language = parts[3] if len(parts) > 3 else "swe"
            # ID is the last field
            entry_id = parts[-1] if len(parts) > 4 else ""

            entries.append(BraxenEntry(
                entry_id=entry_id,
                word=word,
                transcript=transcript,
                language=language,
                pos=pos
            ))
    return entries


def has_digits(word: str) -> bool:
    """Check if word contains digits."""
    return bool(re.search(r"[0-9]", word))


def split_by_language(entries: list[BraxenEntry],
                      skip_codes: set[str] = None) -> dict[str, list[BraxenEntry]]:
    """
    Split entries by language code.

    Args:
        entries: List of BraxenEntry objects
        skip_codes: Language codes to skip (e.g., mixed/problematic entries)

    Returns:
        Dict mapping language code to list of entries
    """
    if skip_codes is None:
        skip_codes = {"afr", "asi", "aus", "sla", "mix", "fisa"}

    by_language = defaultdict(list)
    for entry in entries:
        if entry.language in skip_codes:
            continue
        if has_digits(entry.word):
            continue
        by_language[entry.language].append(entry)

    return dict(by_language)


# Mapping from Braxen language codes to Hunspell dictionary codes
CODE2DICT = {
    "lat": ["la"],
    "swe": ["sv"],
    "nob": ["nb"],
    "nno": ["nn"],
    "dan": ["da"],
    "isl": ["is"],
    "fin": ["fi"],
    "est": ["et"],
    "lav": ["lv"],
    "lit": ["lt"],
    "pol": ["pl"],
    "cze": ["cs"],
    "slk": ["sk"],
    "slv": ["sl"],
    "hrv": ["hr"],
    "srp": ["sr-Latn"],
    "bos": ["bs"],
    "mkd": ["mk"],
    "bul": ["bg"],
    "ukr": ["uk"],
    "rus": ["ru"],
    "deu": ["de"],
    "nld": ["nl", "dut"],
    "eng": ["en", "en-GB", "en-CA", "en-AU", "en-ZA"],
    "fre": ["fr"],
    "ita": ["it"],
    "spa": ["es", "es-MX", "es-AR", "es-CL", "es-ES"],
    "por": ["pt", "pt-PT"],
    "rom": ["ro"],
    "hun": ["hu"],
    "tur": ["tr"],
    "gre": ["el"],
}

# Nordic normalization: Swedish spelling -> native spelling (for lookup only)
NORDIC_NORM = {
    "ae": "æ",
    "Ae": "Æ",
    "AE": "Æ",
    "oe": "ø",
    "Oe": "Ø",
    "OE": "Ø",
    "aa": "å",
    "Aa": "Å",
    "AA": "Å",
    "ä": "æ",
    "Ä": "Æ",
    "ö": "ø",
    "Ö": "Ø",
}


def normalize_nordic(word: str) -> str:
    """
    Normalize Swedish-convention spelling to native Nordic spelling.
    Used for Hunspell lookup only, never modifies the source data.
    """
    result = word
    for swedish, native in NORDIC_NORM.items():
        result = result.replace(swedish, native)
    return result


def find_hunspell_dicts(dict_root: Path) -> dict[str, tuple[Path, Path]]:
    """
    Find Hunspell dictionary pairs (index.dic, index.aff) under dict_root.

    Returns:
        Dict mapping dict code to (dic_path, aff_path)
    """
    pairs = {}
    for aff_path in dict_root.glob("*/index.aff"):
        dic_path = aff_path.parent / "index.dic"
        if dic_path.exists():
            code = aff_path.parent.name
            pairs[code] = (dic_path, aff_path)
    return pairs


def validate_entries(entries_by_lang: dict[str, list[BraxenEntry]],
                     dict_root: Path) -> dict[str, list[BraxenEntry]]:
    """
    Run Hunspell validation on entries.

    Modifies entries in place, setting hunspell_ok and hunspell_suggestions.
    Nordic normalization is used for lookup but never written to the entry.

    Returns the same dict (entries modified in place).
    """
    try:
        import hunspell
    except ImportError:
        print("Warning: hunspell not installed, skipping validation")
        return entries_by_lang

    dict_pairs = find_hunspell_dicts(dict_root)

    for lang_code, entries in entries_by_lang.items():
        if lang_code not in CODE2DICT:
            print(f"  {lang_code}: no dict mapping, skipping validation")
            continue

        # Find a working dictionary
        hs = None
        used_dict = None
        for dict_code in CODE2DICT[lang_code]:
            if dict_code in dict_pairs:
                dic_path, aff_path = dict_pairs[dict_code]
                try:
                    hs = hunspell.HunSpell(str(dic_path), str(aff_path))
                    used_dict = dict_code
                    break
                except Exception as e:
                    print(f"  {lang_code}: failed to load {dict_code}: {e}")

        if not hs:
            print(f"  {lang_code}: no dict found for {CODE2DICT[lang_code]}")
            continue

        print(f"  {lang_code}: validating {len(entries)} entries with {used_dict}")

        is_nordic = lang_code in {"nob", "nno", "dan"}

        for entry in entries:
            # Normalize for lookup only (never modify entry.word)
            lookup_word = normalize_nordic(entry.word) if is_nordic else entry.word

            if hs.spell(lookup_word):
                entry.hunspell_ok = True
                entry.hunspell_suggestions = []
            else:
                entry.hunspell_ok = False
                entry.hunspell_suggestions = hs.suggest(lookup_word)

    return entries_by_lang


def filter_entries(entries_by_lang: dict[str, list[BraxenEntry]],
                   min_entries: int = 100,
                   require_hunspell_ok: bool = True) -> dict[str, list[BraxenEntry]]:
    """
    Filter entries by Hunspell validation and minimum count.

    Args:
        entries_by_lang: Dict from language code to entries
        min_entries: Minimum entries required to keep a language
        require_hunspell_ok: If True, only keep Hunspell-OK entries

    Returns:
        Filtered dict
    """
    filtered = {}

    for lang_code, entries in entries_by_lang.items():
        if require_hunspell_ok:
            # Keep only validated entries (hunspell_ok is True, not None or False)
            kept = [e for e in entries if e.hunspell_ok is True]
        else:
            # Keep entries that weren't explicitly rejected
            kept = [e for e in entries if e.hunspell_ok is not False]

        if len(kept) >= min_entries:
            filtered[lang_code] = kept
            print(f"  {lang_code}: {len(kept)}/{len(entries)} entries kept")
        else:
            print(f"  {lang_code}: {len(kept)} entries < {min_entries} minimum, dropping")

    return filtered


# =============================================================================
# Stage 5: Train/test split
# =============================================================================

def train_test_split(entries_by_lang: dict[str, list[BraxenEntry]],
                     test_ratio: float = 0.1,
                     seed: int = 42) -> tuple[dict, dict]:
    """
    Split entries into train and test sets.

    Args:
        entries_by_lang: Dict from language code to entries
        test_ratio: Fraction of entries to use for test
        seed: Random seed for reproducibility

    Returns:
        (train_dict, test_dict)
    """
    random.seed(seed)

    train = {}
    test = {}

    for lang_code, entries in entries_by_lang.items():
        shuffled = entries.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_ratio))
        train[lang_code] = shuffled[:split_idx]
        test[lang_code] = shuffled[split_idx:]

        print(f"  {lang_code}: {len(train[lang_code])} train, {len(test[lang_code])} test")

    return train, test


# =============================================================================
# Stage 6: Output for training
# =============================================================================

def strip_accents(transcript: str) -> str:
    """
    Strip accent/stress markers from transcript.

    Braxen conventions:
    - ' (primary stress)
    - " (secondary stress)
    - , (grave accent / accent 2 marker)

    Kept:
    - . (syllable boundary)
    - | (morpheme/compound boundary)
    """
    return re.sub(r"['\",]", "", transcript)


def write_training_files(entries_by_lang: dict[str, list[BraxenEntry]],
                         output_dir: Path,
                         suffix: str = "",
                         strip_stress: bool = False):
    """
    Write training files in word<TAB>transcript format.

    Args:
        entries_by_lang: Dict from language code to entries
        output_dir: Directory to write files
        suffix: Suffix to add to filenames (e.g., ".train", ".test")
        strip_stress: If True, strip accent/stress markers from transcripts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang_code, entries in entries_by_lang.items():
        out_path = output_dir / f"{lang_code}{suffix}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in entries:
                transcript = strip_accents(entry.transcript) if strip_stress else entry.transcript
                f.write(f"{entry.word}\t{transcript}\n")
        print(f"  Wrote {out_path} ({len(entries)} entries)")


def write_merged_training_file(entries_by_lang: dict[str, list[BraxenEntry]],
                               output_path: Path,
                               add_prefix: bool = True,
                               strip_stress: bool = False):
    """
    Write a merged training file with all languages.

    Args:
        entries_by_lang: Dict from language code to entries
        output_path: Path to write merged file
        add_prefix: If True, add language prefix to words (e.g., "eng_word")
        strip_stress: If True, strip accent/stress markers from transcripts
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for lang_code, entries in entries_by_lang.items():
            for entry in entries:
                transcript = strip_accents(entry.transcript) if strip_stress else entry.transcript
                if add_prefix:
                    word = f"{lang_code}_{entry.word}"
                else:
                    word = entry.word
                f.write(f"{word}\t{transcript}\n")

    total = sum(len(e) for e in entries_by_lang.values())
    print(f"  Wrote {output_path} ({total} entries, prefix={add_prefix})")


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(braxen_path: Path,
                 dict_root: Path,
                 output_dir: Path,
                 min_entries: int = 100,
                 test_ratio: float = 0.1,
                 seed: int = 42):
    """Run the full pipeline."""

    print("=" * 60)
    print("Stage 1: Load")
    print("=" * 60)
    entries = load_braxen(braxen_path)
    print(f"  Loaded {len(entries)} entries from {braxen_path}")

    print()
    print("=" * 60)
    print("Stage 2: Split by language")
    print("=" * 60)
    by_lang = split_by_language(entries)
    for lang, ents in sorted(by_lang.items(), key=lambda x: -len(x[1])):
        print(f"  {lang}: {len(ents)} entries")

    print()
    print("=" * 60)
    print("Stage 3: Validate (Hunspell)")
    print("=" * 60)
    by_lang = validate_entries(by_lang, dict_root)

    print()
    print("=" * 60)
    print("Stage 4: Filter")
    print("=" * 60)
    filtered = filter_entries(by_lang, min_entries=min_entries)

    print()
    print("=" * 60)
    print("Stage 5: Train/test split")
    print("=" * 60)
    train, test = train_test_split(filtered, test_ratio=test_ratio, seed=seed)

    print()
    print("=" * 60)
    print("Stage 6: Write training files")
    print("=" * 60)

    # With accents
    print("\nWith accent markers:")
    write_training_files(train, output_dir / "with_accents", suffix=".train")
    write_training_files(test, output_dir / "with_accents", suffix=".test")
    write_merged_training_file(train, output_dir / "with_accents" / "merged.train.txt", add_prefix=True)
    write_merged_training_file(train, output_dir / "with_accents" / "raw.train.txt", add_prefix=False)

    # Without accents
    print("\nWithout accent markers:")
    write_training_files(train, output_dir / "no_accents", suffix=".train", strip_stress=True)
    write_training_files(test, output_dir / "no_accents", suffix=".test", strip_stress=True)
    write_merged_training_file(train, output_dir / "no_accents" / "merged.train.txt", add_prefix=True, strip_stress=True)
    write_merged_training_file(train, output_dir / "no_accents" / "raw.train.txt", add_prefix=False, strip_stress=True)

    print()
    print("=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nOutput written to: {output_dir}")
    print("\nNext steps:")
    print("  - Train Phonetisaurus models (WL, MWL, RAW) on the train files")
    print("  - Evaluate against .test files")
    print("  - Compare with_accents vs no_accents results")


def main():
    parser = argparse.ArgumentParser(description="Multilingual G2P pipeline")
    parser.add_argument("--braxen", type=Path, required=True,
                        help="Path to braxen-sv.tsv")
    parser.add_argument("--dicts", type=Path, required=True,
                        help="Path to Hunspell dictionaries root (e.g., wooorm/dictionaries)")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory")
    parser.add_argument("--min-entries", type=int, default=100,
                        help="Minimum entries per language")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of data for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    run_pipeline(
        braxen_path=args.braxen,
        dict_root=args.dicts,
        output_dir=args.output,
        min_entries=args.min_entries,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
