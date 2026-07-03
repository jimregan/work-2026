"""Stanza wrapper: parse standardized Irish, one sentence per input segment."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from .conllu import Sentence, Token


@lru_cache(maxsize=1)
def _pipeline():
    import stanza

    # tokenize_no_ssplit: never split a segment into multiple sentences, so each
    # input line stays one sentence and stays aligned with the modernization.
    # processors is left unset: ga has no mwt model (as of stanza 1.13), so the
    # language default set is what actually loads.
    return stanza.Pipeline(
        lang="ga",
        tokenize_no_ssplit=True,
        download_method=None,
        verbose=False,
    )


def parse_sentence(standard_text: str) -> Sentence:
    doc = _pipeline()(standard_text)
    tokens: List[Token] = []
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append(
                Token(
                    id=str(word.id),
                    form=word.text or "_",
                    lemma=word.lemma or "_",
                    upos=word.upos or "_",
                    xpos=word.xpos or "_",
                    feats=word.feats or "_",
                    head=str(word.head),
                    deprel=word.deprel or "_",
                    deps="_",
                    misc=word.misc or "_",
                )
            )
    _renumber(tokens)
    return Sentence(tokens=tokens)


def _renumber(tokens: List[Token]) -> None:
    """Flatten possibly multi-sentence output into one contiguous id space."""
    if all(t.id == str(i + 1) for i, t in enumerate(tokens)):
        return
    remap = {}
    offset = 0
    prev = 0
    for i, t in enumerate(tokens):
        cur = int(t.id)
        if cur <= prev:
            offset = i
        prev = cur
        remap[(offset, cur)] = i + 1
        t._offset = offset  # type: ignore[attr-defined]
    for t in tokens:
        off = getattr(t, "_offset", 0)
        t.id = str(remap[(off, int(t.id))])
        if t.head != "_" and t.head != "0":
            t.head = str(remap[(off, int(t.head))])
        if hasattr(t, "_offset"):
            delattr(t, "_offset")


def forms(sentence: Sentence) -> List[str]:
    return [t.form for t in sentence.tokens]
