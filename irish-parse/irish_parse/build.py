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


# quotative "ar sé"/"ar sí" (from arsé/arsí) is a deliberate split, not a
# synthetic verb form — never merge it back
QUOTATIVE_LEMMAS = {"ar", "arsa"}


def merge_synthetic_pronouns(sentence: Sentence) -> None:
    """Collapse analytic verb + subject-pronoun splits of synthetic forms.

    Standardization renders synthetic verb forms analytically (rinneas ->
    rinne mé), which the treebank rejects: the verb row keeps the original
    synthetic form and takes the pronoun's Person/Number/Gender features; the
    pronoun row is deleted and ids/heads renumbered.

    Applies only inside two-word multiword tokens (one original -> verb +
    nsubj pronoun), so genuinely analytic originals are never touched.
    """
    tokens = sentence.tokens
    out: List[Token] = []
    pron_to_verb: Dict[str, str] = {}  # deleted pronoun id -> its verb's id
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "-" in t.id and i + 2 < len(tokens):
            lo, hi = t.id.split("-")
            verb, pron = tokens[i + 1], tokens[i + 2]
            if (
                int(hi) - int(lo) == 1
                and verb.upos == "VERB"
                and pron.upos == "PRON"
                and pron.head == verb.id
                and pron.deprel.startswith("nsubj")
                and verb.lemma not in QUOTATIVE_LEMMAS
            ):
                verb.form = t.form  # the original synthetic form
                _copy_person_feats(verb, pron)
                for k, v in t.misc_dict().items():  # e.g. Align=Check
                    verb.add_misc(k, v)
                pron_to_verb[pron.id] = verb.id
                out.append(verb)
                i += 3
                continue
        out.append(t)
        i += 1

    # renumber word ids contiguously; remap heads and range lines
    old_to_new: Dict[str, str] = {}
    n = 0
    for tok in out:
        if "-" not in tok.id:
            n += 1
            old_to_new[tok.id] = str(n)
    for old, verb_old in pron_to_verb.items():
        old_to_new[old] = old_to_new[verb_old]
    for tok in out:
        if "-" in tok.id:
            lo, hi = tok.id.split("-")
            tok.id = f"{old_to_new[lo]}-{old_to_new[hi]}"
        else:
            tok.id = old_to_new[tok.id]
            if tok.head not in ("_", "0"):
                tok.head = old_to_new.get(tok.head, "_")
    sentence.tokens = out


def _copy_person_feats(verb: Token, pron: Token) -> None:
    """Move the pronoun's Person/Number/Gender onto the verb's FEATS."""
    if pron.feats in ("_", ""):
        return
    feats = {}
    if verb.feats not in ("_", ""):
        for item in verb.feats.split("|"):
            k, _, v = item.partition("=")
            feats[k] = v
    for item in pron.feats.split("|"):
        k, _, v = item.partition("=")
        if k in ("Person", "Number", "Gender") and k not in feats:
            feats[k] = v
    verb.feats = "|".join(f"{k}={feats[k]}" for k in sorted(feats, key=str.lower))
