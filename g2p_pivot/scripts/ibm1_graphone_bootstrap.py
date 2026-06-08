#!/usr/bin/env python3
"""Bootstrap 1:1 grapheme-phone correspondences with IBM Model 1.

This is intended as the first curriculum step for graphone learning:
only entries with equal grapheme-token and phone-token counts are used
for training, then per-language P(phone | grapheme) tables are exported.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Entry:
    row_id: str
    language: str
    word: str
    phones_text: str
    graphemes: tuple[str, ...]
    phones: tuple[str, ...]


ProbabilityTable = dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-language IBM Model 1 graphone seed tables on 1:1 entries."
    )
    parser.add_argument("input", type=Path, help="CSV/TSV lexicon input.")
    parser.add_argument("--delimiter", default=None, help="Input delimiter. Defaults from suffix.")
    parser.add_argument("--language-col", default="language")
    parser.add_argument("--word-col", default="word")
    parser.add_argument("--phones-col", default="phones")
    parser.add_argument("--language-index", type=int, default=None)
    parser.add_argument("--word-index", type=int, default=None)
    parser.add_argument("--phones-index", type=int, default=None)
    parser.add_argument("--id-col", default=None, help="Optional stable row id column.")
    parser.add_argument("--id-index", type=int, default=None, help="Optional stable row id column index.")
    parser.add_argument(
        "--language-filter",
        action="append",
        default=[],
        help="Keep only this language code. Can be repeated.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Read rows by zero-based column indices instead of header names.",
    )
    parser.add_argument(
        "--comment-prefix",
        default="#",
        help="Skip input lines beginning with this prefix. Use empty string to disable.",
    )
    parser.add_argument(
        "--phone-separator",
        default=None,
        help="Phone token separator. Default: split on whitespace.",
    )
    parser.add_argument(
        "--drop-phone-token",
        action="append",
        default=[],
        help="Phone token to remove after tokenization. Can be repeated.",
    )
    parser.add_argument(
        "--strip-phone-prefixes",
        default="",
        help="Characters to strip from the start of each phone token.",
    )
    parser.add_argument(
        "--casefold-words",
        action="store_true",
        help="Lowercase words before character tokenization.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="IBM Model 1 EM iterations per language.",
    )
    parser.add_argument(
        "--min-seed-count",
        type=int,
        default=2,
        help="Minimum expected count for exported seed graphones.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.6,
        help="Minimum P(phone | grapheme) for exported seed graphones.",
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=1.5,
        help="Maximum grapheme distribution entropy for exported seed graphones.",
    )
    parser.add_argument("--seeds-out", type=Path, default=Path("graphone_seeds.tsv"))
    parser.add_argument("--scores-out", type=Path, default=Path("entry_scores.tsv"))
    parser.add_argument(
        "--skip-scores",
        action="store_true",
        help="Do not score every entry under every language model.",
    )
    parser.add_argument(
        "--training-out",
        type=Path,
        default=None,
        help="Optional TSV of entries retained for 1:1 training.",
    )
    return parser.parse_args()


def infer_delimiter(path: Path, requested: str | None) -> str:
    if requested is not None:
        if requested == "\\t":
            return "\t"
        return requested
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    return ","


def tokenize_word(word: str, casefold: bool) -> tuple[str, ...]:
    normalized = word.strip()
    if casefold:
        normalized = normalized.casefold()
    return tuple(ch for ch in normalized if not ch.isspace())


def tokenize_phones(
    phones: str,
    separator: str | None,
    drop_tokens: set[str],
    strip_prefixes: str,
) -> tuple[str, ...]:
    text = phones.strip()
    if separator is None:
        raw_tokens = text.split()
    else:
        raw_tokens = [part for part in text.split(separator) if part]

    tokens: list[str] = []
    for token in raw_tokens:
        normalized = token.lstrip(strip_prefixes) if strip_prefixes else token
        if not normalized or normalized in drop_tokens:
            continue
        tokens.append(normalized)
    return tuple(tokens)


def read_entries(args: argparse.Namespace) -> list[Entry]:
    delimiter = infer_delimiter(args.input, args.delimiter)
    if args.no_header:
        return read_indexed_entries(args, delimiter)

    language_filter = set(args.language_filter)
    entries: list[Entry] = []
    handle = filtered_lines(args.input, args.comment_prefix)
    reader = csv.DictReader(handle, delimiter=delimiter)
    required = {args.language_col, args.word_col, args.phones_col}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        names = ", ".join(sorted(missing))
        raise SystemExit(f"Missing required input column(s): {names}")

    for index, row in enumerate(reader, start=1):
        language = row[args.language_col].strip()
        word = row[args.word_col].strip()
        phones_text = row[args.phones_col].strip()
        if language_filter and language not in language_filter:
            continue
        if not language or not word or not phones_text:
            continue
        row_id = row[args.id_col].strip() if args.id_col else str(index)
        entries.append(
            Entry(
                row_id=row_id,
                language=language,
                word=word,
                phones_text=phones_text,
                graphemes=tokenize_word(word, args.casefold_words),
                phones=tokenize_phones(
                    phones_text,
                    args.phone_separator,
                    set(args.drop_phone_token),
                    args.strip_phone_prefixes,
                ),
            )
        )
    return entries


def filtered_lines(path: Path, comment_prefix: str):
    handle = path.open(newline="", encoding="utf-8")

    def iterator():
        with handle:
            for line in handle:
                if comment_prefix and line.startswith(comment_prefix):
                    continue
                yield line

    return iterator()


def read_indexed_entries(args: argparse.Namespace, delimiter: str) -> list[Entry]:
    language_filter = set(args.language_filter)
    required = {
        "word": args.word_index,
        "phones": args.phones_index,
        "language": args.language_index,
    }
    missing = [name for name, index in required.items() if index is None]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"--no-header requires column index argument(s): {names}")

    entries: list[Entry] = []
    max_required_index = max(index for index in required.values() if index is not None)
    handle = filtered_lines(args.input, args.comment_prefix)
    reader = csv.reader(handle, delimiter=delimiter)
    for index, row in enumerate(reader, start=1):
        if len(row) <= max_required_index:
            continue
        language = row[args.language_index].strip()
        word = row[args.word_index].strip()
        phones_text = row[args.phones_index].strip()
        if language_filter and language not in language_filter:
            continue
        if not language or not word or not phones_text:
            continue
        row_id = (
            row[args.id_index].strip()
            if args.id_index is not None and len(row) > args.id_index
            else str(index)
        )
        entries.append(
            Entry(
                row_id=row_id,
                language=language,
                word=word,
                phones_text=phones_text,
                graphemes=tokenize_word(word, args.casefold_words),
                phones=tokenize_phones(
                    phones_text,
                    args.phone_separator,
                    set(args.drop_phone_token),
                    args.strip_phone_prefixes,
                ),
            )
        )
    return entries


def one_to_one_entries(entries: Iterable[Entry]) -> list[Entry]:
    return [
        entry
        for entry in entries
        if entry.graphemes
        and entry.phones
        and len(entry.graphemes) == len(entry.phones)
    ]


def train_ibm1(entries: list[Entry], iterations: int) -> tuple[ProbabilityTable, Counter[tuple[str, str]]]:
    weighted_entries = Counter((entry.graphemes, entry.phones) for entry in entries)
    phones_by_grapheme: dict[str, set[str]] = defaultdict(set)
    for graphemes, phones in weighted_entries:
        for grapheme in graphemes:
            phones_by_grapheme[grapheme].update(phones)
    probabilities: ProbabilityTable = {
        g: {p: 1.0 / len(phones) for p in phones}
        for g, phones in sorted(phones_by_grapheme.items())
        if phones
    }
    expected_counts: Counter[tuple[str, str]] = Counter()

    for _ in range(iterations):
        expected_counts = Counter()
        grapheme_totals: Counter[str] = Counter()

        for (graphemes, phones), weight in weighted_entries.items():
            for phone in phones:
                normalization = sum(
                    probabilities.get(grapheme, {}).get(phone, 0.0)
                    for grapheme in graphemes
                )
                if normalization == 0.0:
                    continue
                for grapheme in graphemes:
                    value = weight * probabilities.get(grapheme, {}).get(phone, 0.0) / normalization
                    expected_counts[(grapheme, phone)] += value
                    grapheme_totals[grapheme] += value

        updated: ProbabilityTable = {}
        for (grapheme, phone), count in expected_counts.items():
            if grapheme_totals[grapheme] == 0:
                continue
            updated.setdefault(grapheme, {})[phone] = count / grapheme_totals[grapheme]
        probabilities = updated

    return probabilities, expected_counts


def distribution_entropy(distribution: dict[str, float]) -> float:
    return -sum(probability * math.log2(probability) for probability in distribution.values() if probability)


def language_models(training_entries: list[Entry], iterations: int) -> dict[str, tuple[ProbabilityTable, Counter[tuple[str, str]]]]:
    grouped: dict[str, list[Entry]] = defaultdict(list)
    for entry in training_entries:
        grouped[entry.language].append(entry)
    return {
        language: train_ibm1(entries, iterations)
        for language, entries in sorted(grouped.items())
    }


def write_training_entries(path: Path, entries: list[Entry]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_id", "language", "word", "phones", "grapheme_count", "phone_count"])
        for entry in entries:
            writer.writerow(
                [
                    entry.row_id,
                    entry.language,
                    entry.word,
                    entry.phones_text,
                    len(entry.graphemes),
                    len(entry.phones),
                ]
            )


def write_seed_table(
    path: Path,
    models: dict[str, tuple[ProbabilityTable, Counter[tuple[str, str]]]],
    min_seed_count: int,
    min_probability: float,
    max_entropy: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "language",
                "grapheme",
                "phone",
                "probability",
                "expected_count",
                "entropy",
                "accepted_seed",
            ]
        )
        for language, (probabilities, expected_counts) in models.items():
            for grapheme, phone_probs in sorted(probabilities.items()):
                entropy = distribution_entropy(phone_probs)
                for phone, probability in sorted(
                    phone_probs.items(), key=lambda item: (-item[1], item[0])
                ):
                    count = expected_counts[(grapheme, phone)]
                    accepted = (
                        count >= min_seed_count
                        and probability >= min_probability
                        and entropy <= max_entropy
                    )
                    writer.writerow(
                        [
                            language,
                            grapheme,
                            phone,
                            f"{probability:.8f}",
                            f"{count:.3f}",
                            f"{entropy:.5f}",
                            int(accepted),
                        ]
                    )


def entry_model_score(entry: Entry, probabilities: ProbabilityTable, floor: float = 1e-9) -> float:
    scores: list[float] = []
    unique_graphemes = sorted(set(entry.graphemes))
    for phone in entry.phones:
        phone_probability = max(
            (probabilities.get(grapheme, {}).get(phone, floor) for grapheme in unique_graphemes),
            default=floor,
        )
        scores.append(math.log(phone_probability))
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


def write_entry_scores(
    path: Path,
    entries: list[Entry],
    models: dict[str, tuple[ProbabilityTable, Counter[tuple[str, str]]]],
) -> None:
    languages = sorted(models)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "row_id",
                "declared_language",
                "word",
                "phones",
                "best_language",
                "declared_rank",
                "best_score",
                "declared_score",
                "margin",
                "all_scores",
            ]
        )
        for entry in entries:
            scores = {
                language: entry_model_score(entry, models[language][0])
                for language in languages
            }
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_language, best_score = ranked[0] if ranked else ("", float("-inf"))
            declared_score = scores.get(entry.language, float("-inf"))
            declared_rank = next(
                (index for index, (language, _) in enumerate(ranked, start=1) if language == entry.language),
                "",
            )
            second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
            margin = best_score - second_score if second_score != float("-inf") else float("inf")
            all_scores = ";".join(f"{language}:{score:.5f}" for language, score in ranked)
            writer.writerow(
                [
                    entry.row_id,
                    entry.language,
                    entry.word,
                    entry.phones_text,
                    best_language,
                    declared_rank,
                    f"{best_score:.5f}",
                    f"{declared_score:.5f}",
                    f"{margin:.5f}",
                    all_scores,
                ]
            )


def main() -> None:
    args = parse_args()
    entries = read_entries(args)
    training_entries = one_to_one_entries(entries)
    if not training_entries:
        raise SystemExit("No 1:1 entries found. Check tokenization and phone separator.")

    models = language_models(training_entries, args.iterations)
    write_seed_table(
        args.seeds_out,
        models,
        min_seed_count=args.min_seed_count,
        min_probability=args.min_probability,
        max_entropy=args.max_entropy,
    )
    if not args.skip_scores:
        write_entry_scores(args.scores_out, entries, models)
    if args.training_out:
        write_training_entries(args.training_out, training_entries)

    print(f"Read {len(entries)} entries")
    print(f"Trained on {len(training_entries)} 1:1 entries")
    print(f"Wrote seed table to {args.seeds_out}")
    if args.skip_scores:
        print("Skipped entry scoring")
    else:
        print(f"Wrote entry scores to {args.scores_out}")


if __name__ == "__main__":
    main()
