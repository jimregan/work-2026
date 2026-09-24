"""Text-specific standardisation overrides.

The intergaelic API misses some dialect forms (e.g. Ulster "go dé", which
should standardise to "cad é", or "acht" for modern "ach"). A rules file
supplies per-text corrections that are applied on top of the API's
[original, standard] pairs, keyed on the original tokens — so the alignment
back to the original surface is unaffected.

Rules file format: one rule per line, TAB-separated, ``#`` comments allowed::

    go dé\tcad é
    acht\tach

Both sides may be multi-word. Matching against the original tokens is
case-insensitive; an initial capital on the first matched token is preserved
on the replacement.
"""
from __future__ import annotations

from typing import List, Tuple

from .modernize import Pair

Rule = Tuple[List[str], List[str]]  # (original tokens, replacement words)


def load_rules(path: str) -> List[Rule]:
    rules: List[Rule] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{lineno}: expected TAB-separated rule: {line!r}")
            src, dst = line.split("\t", 1)
            src_toks = src.split()
            dst_toks = dst.split()
            if not src_toks:
                raise ValueError(f"{path}:{lineno}: empty original side")
            rules.append(([t.lower() for t in src_toks], dst_toks))
    # longest original first, so "go dé" wins over a hypothetical "go" rule
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def apply(pairs: List[Pair], rules: List[Rule]) -> List[Pair]:
    """Rewrite the standard side of ``pairs`` wherever a rule's original
    tokens match, leaving the original side untouched."""
    result: List[Pair] = []
    i = 0
    while i < len(pairs):
        matched = None
        for src, dst in rules:
            n = len(src)
            if i + n <= len(pairs) and all(
                pairs[i + k][0].lower() == src[k] for k in range(n)
            ):
                matched = (n, dst)
                break
        if matched is None:
            result.append(pairs[i])
            i += 1
            continue
        n, dst = matched
        origs = [pairs[i + k][0] for k in range(n)]
        # distribute the replacement words over the n original tokens
        if len(dst) >= n:
            stds = dst[: n - 1] + [" ".join(dst[n - 1 :])]
        else:
            # fewer replacement words than originals: the tail originals are
            # deleted in the standard (-> Skip=Standard placeholder rows)
            stds = list(dst) + [""] * (n - len(dst))
        if origs[0][:1].isupper() and stds[0]:
            stds[0] = stds[0][0].upper() + stds[0][1:]
        result.extend(zip(origs, stds))
        i += n
    return result
