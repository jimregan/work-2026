"""Induce context-dependent phone rewrite rules from aligned pronunciations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass(frozen=True)
class Edit:
    source: Optional[str]
    target: Optional[str]
    source_index: int
    target_index: int


@dataclass
class RuleStats:
    segment: str
    replacement: str
    preceding_context: str
    following_context: str
    count: int
    opportunities: int
    probability: float
    examples: list[str] = field(default_factory=list)


def align_phones(citation: list[str], observed: list[str]) -> list[Edit]:
    """Return a minimum-edit alignment between citation and observed phones."""
    rows = len(citation) + 1
    cols = len(observed) + 1
    cost = [[0] * cols for _ in range(rows)]
    back: list[list[str | None]] = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        cost[i][0] = i
        back[i][0] = "delete"
    for j in range(1, cols):
        cost[0][j] = j
        back[0][j] = "insert"

    for i in range(1, rows):
        for j in range(1, cols):
            sub_cost = 0 if citation[i - 1] == observed[j - 1] else 1
            choices = [
                (cost[i - 1][j - 1] + sub_cost, "match" if sub_cost == 0 else "sub"),
                (cost[i - 1][j] + 1, "delete"),
                (cost[i][j - 1] + 1, "insert"),
            ]
            cost[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    edits: list[Edit] = []
    i = len(citation)
    j = len(observed)
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "sub"}:
            edits.append(Edit(citation[i - 1], observed[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "delete":
            edits.append(Edit(citation[i - 1], None, i - 1, j))
            i -= 1
        elif op == "insert":
            edits.append(Edit(None, observed[j - 1], i, j - 1))
            j -= 1
        else:
            raise RuntimeError("alignment backtrace failed")

    edits.reverse()
    return edits


def _context(seq: list[str], index: int, left: int, right: int) -> tuple[str, str]:
    left_tokens = seq[max(0, index - left):index]
    right_tokens = seq[index + 1:index + 1 + right]
    return " ".join(left_tokens), " ".join(right_tokens)


def _normalise_pairs(
    pairs: Iterable[tuple[str, str, str]],
) -> list[tuple[str, list[str], list[str]]]:
    result = []
    for item_id, citation, observed in pairs:
        citation_phones = citation.split()
        observed_phones = observed.split()
        if citation_phones and observed_phones:
            result.append((item_id, citation_phones, observed_phones))
    return result


def induce_rules(
    pairs: Iterable[tuple[str, str, str]],
    left_context: int = 1,
    right_context: int = 1,
    min_count: int = 2,
    min_probability: float = 0.0,
    max_examples: int = 5,
) -> list[RuleStats]:
    """Learn rewrite candidates from citation/observed pronunciation pairs."""
    examples = _normalise_pairs(pairs)
    opportunities: Counter[tuple[str, str, str]] = Counter()
    changes: Counter[tuple[str, str, str, str]] = Counter()
    seen_examples: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)

    for item_id, citation, observed in examples:
        for i, source in enumerate(citation):
            preceding, following = _context(
                citation, i, left_context, right_context
            )
            opportunities[(source, preceding, following)] += 1

        for edit in align_phones(citation, observed):
            if edit.source is None or edit.target == edit.source:
                continue
            preceding, following = _context(
                citation, edit.source_index, left_context, right_context
            )
            replacement = edit.target or ""
            key = (edit.source, replacement, preceding, following)
            changes[key] += 1
            if len(seen_examples[key]) < max_examples:
                seen_examples[key].append(item_id)

    rules = []
    for key, count in changes.items():
        segment, replacement, preceding, following = key
        total = opportunities[(segment, preceding, following)]
        probability = count / total if total else 0.0
        if count < min_count or probability < min_probability:
            continue
        rules.append(
            RuleStats(
                segment=segment,
                replacement=replacement,
                preceding_context=preceding,
                following_context=following,
                count=count,
                opportunities=total,
                probability=probability,
                examples=seen_examples[key],
            )
        )

    return sorted(
        rules,
        key=lambda rule: (
            -rule.count,
            -rule.probability,
            rule.segment,
            rule.replacement,
            rule.preceding_context,
            rule.following_context,
        ),
    )


def load_pronunciation_pairs(path: str) -> list[tuple[str, str, str]]:
    """Load item_id, citation phones, observed phones from a TSV file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_num}: expected id, citation, observed"
                )
            pairs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return pairs


def load_timit_phn(path: str) -> list[str]:
    """Load observed phones from a TIMIT-style .PHN interval file."""
    phones = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_num}: expected start, end, phone"
                )
            phones.append(parts[2])
    return phones


def load_timit_pair_manifest(path: str) -> list[tuple[str, str, str]]:
    """Load id, citation phones, .PHN path and return pronunciation pairs."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_num}: expected id, citation, phn_path"
                )
            item_id = parts[0].strip()
            citation = parts[1].strip()
            observed = " ".join(load_timit_phn(parts[2].strip()))
            pairs.append((item_id, citation, observed))
    return pairs


def pairs_from_lexicons(
    citation_path: str,
    observed_path: str,
) -> list[tuple[str, str, str]]:
    """Create pronunciation pairs from shared words in two TSV lexicons."""
    from .lexicon import load_lexicon

    citation = defaultdict(list)
    observed = defaultdict(list)
    for word, pron in load_lexicon(citation_path):
        citation[word.lower()].append(pron)
    for word, pron in load_lexicon(observed_path):
        observed[word.lower()].append(pron)

    pairs = []
    for word in sorted(set(citation) & set(observed)):
        for obs_idx, obs_pron in enumerate(observed[word]):
            cit_pron = min(
                citation[word],
                key=lambda pron: _alignment_cost(pron.split(), obs_pron.split()),
            )
            pairs.append((f"{word}:{obs_idx}", cit_pron, obs_pron))
    return pairs


def write_rules_yaml(
    rules: list[RuleStats],
    path: str,
    include_stats: bool = True,
) -> None:
    """Write learned rules in the YAML format consumed by rules.py."""
    import yaml

    data = {"rules": []}
    for rule in rules:
        item = {
            "segment": rule.segment,
            "replacement": rule.replacement,
            "preceding_context": rule.preceding_context,
            "following_context": rule.following_context,
        }
        if include_stats:
            item["count"] = rule.count
            item["opportunities"] = rule.opportunities
            item["probability"] = round(rule.probability, 6)
            item["examples"] = rule.examples
        data["rules"].append(item)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _alignment_cost(citation: list[str], observed: list[str]) -> int:
    return sum(
        1 for edit in align_phones(citation, observed)
        if edit.source != edit.target
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn context-dependent phonetic rewrite rules"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pairs",
        help="TSV with id, citation phones, observed phones",
    )
    source.add_argument(
        "--timit-pairs",
        help="TSV with id, citation phones, TIMIT .PHN path",
    )
    source.add_argument(
        "--citation",
        help="Citation-form lexicon TSV; requires --observed",
    )
    parser.add_argument(
        "--observed",
        help="Observed/validated lexicon TSV when --citation is used",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--left-context", type=int, default=1)
    parser.add_argument("--right-context", type=int, default=1)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--min-probability", type=float, default=0.0)
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Omit count/probability/example metadata from YAML",
    )
    args = parser.parse_args(argv)
    if args.citation and not args.observed:
        parser.error("--citation requires --observed")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.pairs:
        pairs = load_pronunciation_pairs(args.pairs)
    elif args.timit_pairs:
        pairs = load_timit_pair_manifest(args.timit_pairs)
    else:
        pairs = pairs_from_lexicons(args.citation, args.observed)

    rules = induce_rules(
        pairs,
        left_context=args.left_context,
        right_context=args.right_context,
        min_count=args.min_count,
        min_probability=args.min_probability,
    )
    write_rules_yaml(rules, args.output, include_stats=not args.no_stats)


if __name__ == "__main__":
    main()
