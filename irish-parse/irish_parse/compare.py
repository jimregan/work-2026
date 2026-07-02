"""Compare a Stanza parse against a UDPipe parse of the same tokens.

Both parsers are fed identical tokens, so comparison is position-by-position.
The result is a list of disagreements a human must resolve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .conllu import Sentence

# fields worth disagreeing about (xpos tagsets differ between the tools, so
# xpos is not compared)
COMPARE_FIELDS = ("upos", "lemma", "head", "deprel")


@dataclass
class TokenDiff:
    index: int
    form: str
    field: str
    stanza: str
    udpipe: str


@dataclass
class SentenceDiff:
    sent_id: str
    text: str
    token_count: int
    diffs: List[TokenDiff]
    misaligned: bool  # token counts differed — comparison is unreliable
    # originals deleted by standardization (placeholder rows in the primary file)
    dropped: List[str] = field(default_factory=list)

    @property
    def n_tokens_affected(self) -> int:
        return len({d.index for d in self.diffs})

    @property
    def has_notes(self) -> bool:
        return bool(self.diffs or self.misaligned or self.dropped)


def compare(
    sent_id: str, text: str, stanza: Sentence, udpipe: Sentence, alignment=None
) -> SentenceDiff:
    st, ud = stanza.tokens, udpipe.tokens
    misaligned = len(st) != len(ud)
    diffs: List[TokenDiff] = []
    for i, (a, b) in enumerate(zip(st, ud), start=1):
        for f in COMPARE_FIELDS:
            av, bv = getattr(a, f), getattr(b, f)
            if av != bv:
                diffs.append(TokenDiff(i, a.form, f, av, bv))

    dropped: List[str] = []
    if alignment is not None:
        dropped = [
            f"{d.original} (after token {d.after_index + 1})"
            for d in alignment.dropped
        ]
    return SentenceDiff(sent_id, text, len(st), diffs, misaligned, dropped)


def render_markdown(sentence_diffs: List[SentenceDiff]) -> str:
    total_tokens = sum(d.token_count for d in sentence_diffs)
    total_affected = sum(d.n_tokens_affected for d in sentence_diffs)
    n_misaligned = sum(1 for d in sentence_diffs if d.misaligned)
    n_dropped = sum(len(d.dropped) for d in sentence_diffs)
    with_diffs = [d for d in sentence_diffs if d.has_notes]

    out: List[str] = []
    out.append("# Stanza vs UDPipe — differences to resolve\n")
    out.append(f"- Sentences: **{len(sentence_diffs)}**\n")
    out.append(f"- Tokens: **{total_tokens}**\n")
    out.append(
        f"- Tokens with at least one disagreement: **{total_affected}** "
        f"({_pct(total_affected, total_tokens)})\n"
    )
    out.append(f"- Sentences flagged: **{len(with_diffs)}**\n")
    if n_misaligned:
        out.append(
            f"- ⚠️ Sentences where the two parsers tokenized differently "
            f"(comparison unreliable): **{n_misaligned}**\n"
        )
    if n_dropped:
        out.append(
            f"- ➖ Originals deleted by standardization (placeholder rows "
            f"added; fill in the analysis): **{n_dropped}**\n"
        )
    out.append("\n")

    if not with_diffs:
        out.append("No disagreements. 🎉\n")
        return "".join(out)

    for d in with_diffs:
        out.append(f"## Sentence {d.sent_id}\n\n")
        out.append(f"> {d.text}\n\n")
        if d.misaligned:
            out.append(
                "⚠️ **Token count mismatch between Stanza and UDPipe** — "
                "the parsers disagree on tokenization; align by hand.\n\n"
            )
        if d.dropped:
            out.append(
                "**Deleted in standardization** (placeholder row, empty "
                "analysis — fill in):\n"
            )
            for item in d.dropped:
                out.append(f"- `{item}`\n")
            out.append("\n")
        if not d.diffs:
            continue
        out.append("| # | Token | Field | Stanza | UDPipe |\n")
        out.append("|---|-------|-------|--------|--------|\n")
        for td in d.diffs:
            out.append(
                f"| {td.index} | {_esc(td.form)} | {td.field} | "
                f"{_esc(td.stanza)} | {_esc(td.udpipe)} |\n"
            )
        out.append("\n")
    return "".join(out)


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.1f}%" if total else "0.0%"


def _esc(s: Optional[str]) -> str:
    return (s or "_").replace("|", "\\|")
