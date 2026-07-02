"""UDPipe wrapper (ufal.udpipe) using a local Irish-IDT model.

The parser is fed the *same* tokens Stanza produced (horizontal input format:
one sentence per line, tokens separated by spaces) so the two parses share a
tokenization and can be compared position by position.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from . import conllu
from .conllu import Sentence

DEFAULT_MODEL = "/models/irish-idt.udpipe"


def model_path() -> str:
    return os.environ.get("UDPIPE_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _pipeline():
    from ufal.udpipe import Model, Pipeline

    path = model_path()
    model = Model.load(path)
    if model is None:
        raise RuntimeError(
            f"Could not load UDPipe model at {path!r}. "
            "Set UDPIPE_MODEL or build the devcontainer image."
        )
    # 'horizontal': input is already tokenized (whitespace), one sentence/line.
    return Pipeline(model, "horizontal", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")


def parse_tokens(tokens: List[str]) -> Sentence:
    """Parse a single pre-tokenized sentence, returning a CoNLL-U Sentence."""
    line = " ".join(tokens)
    out = _pipeline().process(line)
    sentences = conllu.parse(out)
    if not sentences:
        return Sentence()
    # horizontal + single line yields exactly one sentence
    return sentences[0]
