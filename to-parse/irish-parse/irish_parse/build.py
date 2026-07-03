"""Build the primary CoNLL-U output over the original (pre-standard) tokens.

Conventions (matching the target treebank):
  * FORM is the original surface form, exactly as it appears in ``# text``.
  * LEMMA and the analysis are the parser's output for the standardized text;
    the standardized form itself appears only in ``# text_standard``.
  * An original token that standardization split into several words becomes a
    multiword token: a range line carrying the original form, followed by the
    standard word rows with their analyses.
  * An original token deleted by standardization gets a placeholder row with
    MISC ``Skip=Standard`` and an empty analysis, since the parsers never saw
    it — the human fills it in.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from .conllu import Sentence, Token
from .modernize import Alignment, Pair, UNCERTAIN


def build_primary(
    parser: Sentence, pairs: List[Pair], alignment: Alignment
) -> Sentence:
    """Rebuild ``parser`` (one row per standard word) over the original tokens."""
    mapped = alignment.mapped
    words = parser.tokens

    groups: Dict[int, List[int]] = {}
    for pj, m in enumerate(mapped):
        groups.setdefault(m.orig_index, []).append(pj)

    # units in original-token order: (original, parser_rows, placeholder_misc)
    units: List[Tuple[Optional[str], List[int], Optional[str]]] = []
    for pj in groups.pop(-1, []):
        # untraceable parser tokens (degenerate alignment); keep them, flagged
        units.append((None, [pj], None))
    for oi, (orig, std) in enumerate(pairs):
        if oi in groups:
            units.append((orig, groups[oi], None))
        elif std.strip():
            # standard word(s) merged into a neighbouring parser token; the
            # original must still appear, but its analysis lives elsewhere
            units.append((orig, [], "Align=Check"))
        else:
            # deleted by standardization; the parsers never saw it
            units.append((orig, [], "Skip=Standard"))

    # first pass: assign contiguous new word ids across all units
    old_to_new: Dict[str, str] = {}
    plan = []
    new_id = 0
    for orig, rows, ph_misc in units:
        start = new_id + 1
        new_id += max(len(rows), 1)
        for k, pj in enumerate(rows):
            old_to_new[words[pj].id] = str(start + k)
        plan.append((orig, rows, ph_misc, start, new_id))

    out: List[Token] = []
    for orig, rows, ph_misc, start, end in plan:
        if not rows:
            out.append(Token(id=str(start), form=orig or "_", misc=ph_misc or "_"))
            continue
        uncertain = any(mapped[pj].status == UNCERTAIN for pj in rows)
        if len(rows) > 1:
            # multiword token: the range line carries the original surface
            span = Token(id=f"{start}-{end}", form=orig or "_")
            if uncertain:
                span.add_misc("Align", "Check")
            out.append(span)
        for pj in rows:
            tok = dataclasses.replace(words[pj])
            tok.id = old_to_new[words[pj].id]
            if len(rows) == 1:
                # single word: the row itself carries the original form
                if orig is not None:
                    tok.form = orig
                if uncertain or orig is None:
                    tok.add_misc("Align", "Check")
            out.append(tok)

    # remap heads onto the new id space
    for tok in out:
        if "-" in tok.id:
            continue
        if tok.head not in ("_", "0"):
            tok.head = old_to_new.get(tok.head, "_")

    return Sentence(metadata=list(parser.metadata), tokens=out)
