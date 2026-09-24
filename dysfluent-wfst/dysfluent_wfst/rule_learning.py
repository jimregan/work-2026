"""Induce context-dependent phone rewrite rules from aligned pronunciations."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
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
    score: float = 0.0
    known_rule_coverage: int = 0


@dataclass(frozen=True)
class Change:
    """A contiguous source-to-target change anchored in the citation."""

    source: tuple[str, ...]
    target: tuple[str, ...]
    source_start: int
    source_end: int


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


def changes_from_alignment(edits: list[Edit]) -> list[Change]:
    """Coalesce adjacent non-matching edits into multi-phone changes."""
    changes: list[Change] = []
    source: list[str] = []
    target: list[str] = []
    start: int | None = None
    end: int | None = None

    def flush() -> None:
        nonlocal source, target, start, end
        if start is not None:
            changes.append(Change(tuple(source), tuple(target), start, end or start))
        source, target, start, end = [], [], None, None

    for edit in edits:
        if edit.source == edit.target:
            flush()
            continue
        if start is None:
            start = edit.source_index
        if edit.source is not None:
            source.append(edit.source)
            end = edit.source_index + 1
        if edit.target is not None:
            target.append(edit.target)
        if end is None:
            end = edit.source_index
    flush()
    return changes


def _context(seq: list[str], index: int, left: int, right: int) -> tuple[str, str]:
    left_tokens = seq[max(0, index - left):index]
    right_tokens = seq[index + 1:index + 1 + right]
    return " ".join(left_tokens), " ".join(right_tokens)


def _change_context(
    seq: list[str], change: Change, left: int, right: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    preceding = tuple(seq[max(0, change.source_start - left):change.source_start])
    following = tuple(seq[change.source_end:change.source_end + right])
    return preceding, following


def load_phone_classes(path: str) -> dict[str, set[str]]:
    """Load ``classes: {name: [phones...]}`` from YAML."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("classes", data)
    if not isinstance(raw, dict):
        raise ValueError("Phone classes YAML must contain a mapping")
    result = {}
    for name, phones in raw.items():
        if not isinstance(phones, list) or not all(isinstance(p, str) for p in phones):
            raise ValueError(f"Phone class {name!r} must be a list of strings")
        result[str(name)] = set(phones)
    return result


def _class_pattern(phones: set[str]) -> str:
    """Return the rule compiler's compact union syntax for a phone class."""
    return "[" + "".join(sorted(phones, key=lambda p: (len(p), p))) + "]"


def _context_forms(
    context: tuple[str, ...], phone_classes: dict[str, set[str]] | None
) -> list[tuple[str, int]]:
    """Return exact and class-generalized contexts with specificity costs."""
    forms: list[tuple[tuple[str, ...], int]] = [(context, 0)]
    if not phone_classes:
        return [(" ".join(context), 0)]
    for i, phone in enumerate(context):
        replacements = {
            _class_pattern(members)
            for members in phone_classes.values()
            if phone in members
        }
        expanded = list(forms)
        for current, cost in forms:
            for replacement in replacements:
                variant = list(current)
                variant[i] = replacement
                expanded.append((tuple(variant), cost + 1))
        forms = expanded
    return [(" ".join(value), cost) for value, cost in dict(forms).items()]


def _best_baseline(
    citation: list[str], observed: list[str], variants: Iterable[list[str]]
) -> tuple[list[str], bool]:
    candidates = [citation, *variants]
    best = min(candidates, key=lambda value: _alignment_cost(value, observed))
    return best, best == observed


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
    known_variants: Optional[dict[str, list[list[str]]]] = None,
    phone_classes: Optional[dict[str, set[str]]] = None,
    complexity_penalty: float = 0.02,
    overgeneration_penalty: float = 0.5,
) -> list[RuleStats]:
    """Learn residual rewrite rules after selecting the best known variant.

    ``known_variants`` maps item ids to pronunciations licensed by an existing
    phonological grammar. Exact matches are treated as explained; otherwise the
    closest licensed form becomes the alignment baseline.
    """
    examples = _normalise_pairs(pairs)
    opportunities: Counter[tuple[str, str, str]] = Counter()
    changes: Counter[tuple[str, str, str, str]] = Counter()
    seen_examples: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    complexity: dict[tuple[str, str, str, str], int] = {}
    explained = 0

    for item_id, citation, observed in examples:
        baseline, exact = _best_baseline(
            citation, observed, (known_variants or {}).get(item_id, [])
        )
        if exact:
            explained += int(baseline != citation)
            continue

        aligned_changes = changes_from_alignment(align_phones(baseline, observed))
        for change in aligned_changes:
            preceding, following = _change_context(
                baseline, change, left_context, right_context
            )
            source = " ".join(change.source)
            replacement = " ".join(change.target)
            for left_form, left_cost in _context_forms(preceding, phone_classes):
                for right_form, right_cost in _context_forms(following, phone_classes):
                    key = (source, replacement, left_form, right_form)
                    changes[key] += 1
                    complexity[key] = left_cost + right_cost
                    if len(seen_examples[key]) < max_examples:
                        seen_examples[key].append(item_id)

        # Count candidate opportunities against every baseline after candidates
        # are known in a second pass below.

    candidate_contexts = {
        (segment, preceding, following)
        for segment, _, preceding, following in changes
    }
    for item_id, citation, observed in examples:
        baseline, exact = _best_baseline(
            citation, observed, (known_variants or {}).get(item_id, [])
        )
        for segment, preceding, following in candidate_contexts:
            opportunities[(segment, preceding, following)] += _count_opportunities(
                baseline, segment, preceding, following, phone_classes
            )

    rules = []
    for key, count in changes.items():
        segment, replacement, preceding, following = key
        total = opportunities[(segment, preceding, following)]
        probability = count / total if total else 0.0
        if count < min_count or probability < min_probability:
            continue
        false_positive_rate = 1.0 - probability
        score = (
            math.log1p(count)
            + probability
            - overgeneration_penalty * false_positive_rate
            - complexity_penalty * complexity.get(key, 0)
        )
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
                score=score,
                known_rule_coverage=explained,
            )
        )

    return sorted(
        rules,
        key=lambda rule: (
            -rule.score,
            -rule.count,
            -rule.probability,
            rule.segment,
            rule.replacement,
            rule.preceding_context,
            rule.following_context,
        ),
    )


def _matches_context_token(
    phone: str, token: str, phone_classes: dict[str, set[str]] | None
) -> bool:
    if token.startswith("[") and token.endswith("]") and phone_classes:
        return any(phone in members and token == _class_pattern(members)
                   for members in phone_classes.values())
    return phone == token


def _count_opportunities(
    seq: list[str], segment: str, preceding: str, following: str,
    phone_classes: dict[str, set[str]] | None,
) -> int:
    source = segment.split() if segment else []
    left = preceding.split() if preceding else []
    right = following.split() if following else []
    count = 0
    for i in range(len(seq) + 1):
        if source and seq[i:i + len(source)] != source:
            continue
        if not source and i == len(seq) + 1:
            continue
        if len(left) > i or len(right) > len(seq) - i - len(source):
            continue
        left_actual = seq[i - len(left):i] if left else []
        right_actual = seq[i + len(source):i + len(source) + len(right)]
        if all(_matches_context_token(p, t, phone_classes)
               for p, t in zip(left_actual, left)) and all(
            _matches_context_token(p, t, phone_classes)
            for p, t in zip(right_actual, right)
        ):
            count += 1
    return count


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


def variants_from_known_rules(
    pairs: Iterable[tuple[str, str, str]],
    rules_path: str,
    max_variants: int = 256,
) -> dict[str, list[list[str]]]:
    """Enumerate Pynini outputs licensed by the known-rule cascade.

    The symbol inventory is derived from the data and literal rule fields.
    ``max_variants`` bounds optional-rule combinatorics per item.
    """
    import pynini

    from .rules import compile_rules, load_rules

    materialized = list(pairs)
    inventory = {
        phone
        for _, citation, observed in materialized
        for phone in (citation + " " + observed).split()
    }
    for rule in load_rules(rules_path):
        for field_name in ("segment", "replacement", "preceding_context", "following_context"):
            value = str(rule.get(field_name, ""))
            if not any(char in value for char in "[]?*$^."):
                inventory.update(value.split())

    syms = pynini.SymbolTable()
    syms.add_symbol("<eps>", 0)
    for phone in sorted(inventory):
        if phone:
            syms.add_symbol(phone)
    grammar = compile_rules(rules_path, syms)
    if grammar is None:
        return {}

    result: dict[str, list[list[str]]] = {}
    for item_id, citation, _ in materialized:
        source = pynini.accep(citation, token_type=syms)
        lattice = pynini.compose(source, grammar)
        if lattice.start() == pynini.NO_STATE_ID:
            result[item_id] = []
            continue
        outputs = pynini.project(lattice, "output")
        paths = pynini.shortestpath(
            outputs, nshortest=max_variants, unique=True
        ).paths(output_token_type=syms)
        result[item_id] = [value.split() for value in paths.ostrings()]
    return result


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
    manifest_dir = Path(path).resolve().parent
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
            phn_path = Path(parts[2].strip())
            if not phn_path.is_absolute():
                phn_path = manifest_dir / phn_path
            observed = " ".join(load_timit_phn(str(phn_path)))
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
            item["score"] = round(rule.score, 6)
            item["known_rule_coverage"] = rule.known_rule_coverage
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
        "--known-rules",
        help="Known phonological rules YAML to apply before residual inference",
    )
    parser.add_argument(
        "--phone-classes",
        help="YAML mapping phonological class names to phone lists",
    )
    parser.add_argument("--max-known-variants", type=int, default=256)
    parser.add_argument("--complexity-penalty", type=float, default=0.02)
    parser.add_argument("--overgeneration-penalty", type=float, default=0.5)
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

    known_variants = None
    if args.known_rules:
        known_variants = variants_from_known_rules(
            pairs, args.known_rules, max_variants=args.max_known_variants
        )
    phone_classes = (
        load_phone_classes(args.phone_classes) if args.phone_classes else None
    )

    rules = induce_rules(
        pairs,
        left_context=args.left_context,
        right_context=args.right_context,
        min_count=args.min_count,
        min_probability=args.min_probability,
        known_variants=known_variants,
        phone_classes=phone_classes,
        complexity_penalty=args.complexity_penalty,
        overgeneration_penalty=args.overgeneration_penalty,
    )
    write_rules_yaml(rules, args.output, include_stats=not args.no_stats)


if __name__ == "__main__":
    main()
