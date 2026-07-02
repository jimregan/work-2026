"""Minimal CoNLL-U reading/writing that preserves comment metadata order.

Only depends on the standard library so it can be imported without the heavy
parser stacks (stanza / ufal.udpipe) being installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

FIELDS = (
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
)


@dataclass
class Token:
    id: str = "_"
    form: str = "_"
    lemma: str = "_"
    upos: str = "_"
    xpos: str = "_"
    feats: str = "_"
    head: str = "_"
    deprel: str = "_"
    deps: str = "_"
    misc: str = "_"

    @classmethod
    def from_line(cls, line: str) -> "Token":
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 10:
            raise ValueError(f"Expected 10 columns, got {len(cols)}: {line!r}")
        return cls(*cols)

    def to_line(self) -> str:
        return "\t".join(getattr(self, f) for f in FIELDS)

    def misc_dict(self) -> dict:
        if self.misc in ("_", ""):
            return {}
        out = {}
        for item in self.misc.split("|"):
            if "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
            else:
                out[item] = ""
        return out

    def set_misc(self, mapping: dict) -> None:
        if not mapping:
            self.misc = "_"
            return
        parts = []
        for k, v in mapping.items():
            parts.append(k if v == "" else f"{k}={v}")
        self.misc = "|".join(parts)

    def add_misc(self, key: str, value: str) -> None:
        d = self.misc_dict()
        d[key] = value
        self.set_misc(d)


@dataclass
class Sentence:
    # metadata kept as ordered (key, value) pairs so the round-trip is stable
    metadata: List[Tuple[str, str]] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)

    def meta_get(self, key: str):
        for k, v in self.metadata:
            if k == key:
                return v
        return None

    def meta_set(self, key: str, value: str) -> None:
        for i, (k, _) in enumerate(self.metadata):
            if k == key:
                self.metadata[i] = (key, value)
                return
        self.metadata.append((key, value))

    def to_text(self) -> str:
        lines = []
        for k, v in self.metadata:
            if v is None:
                lines.append(f"# {k}")
            else:
                lines.append(f"# {k} = {v}")
        for tok in self.tokens:
            lines.append(tok.to_line())
        return "\n".join(lines) + "\n"


def parse(text: str) -> List[Sentence]:
    sentences: List[Sentence] = []
    cur = Sentence()
    have_content = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line == "":
            if have_content:
                sentences.append(cur)
                cur = Sentence()
                have_content = False
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                k, v = body.split("=", 1)
                cur.metadata.append((k.strip(), v.strip()))
            else:
                cur.metadata.append((body, None))
            have_content = True
        else:
            cur.tokens.append(Token.from_line(line))
            have_content = True
    if have_content:
        sentences.append(cur)
    return sentences


def dump(sentences: List[Sentence]) -> str:
    return "\n".join(s.to_text() for s in sentences)


def iter_file(path: str) -> Iterator[Sentence]:
    with open(path, encoding="utf-8") as fh:
        yield from parse(fh.read())
