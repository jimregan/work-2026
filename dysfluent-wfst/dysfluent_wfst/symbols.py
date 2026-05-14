"""Vocabulary extraction from wav2vec2 and pynini SymbolTable construction."""

from __future__ import annotations

import pynini


def extract_vocab(model_id: str) -> dict[str, int]:
    """Extract the token vocabulary from a wav2vec2 model.

    Returns a dict mapping token strings to their integer indices.
    Index 0 is typically <pad> which doubles as CTC blank.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    return processor.tokenizer.get_vocab()


def build_symbol_table(vocab: dict[str, int]) -> pynini.SymbolTable:
    """Build a pynini SymbolTable from a wav2vec2 vocabulary.

    Index 0 is mapped to <eps> (OpenFst convention), which also serves
    as the CTC blank label.
    """
    syms = pynini.SymbolTable()
    syms.add_symbol("<eps>", 0)

    for symbol, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if idx == 0:
            continue  # already have epsilon/blank at 0
        syms.add_symbol(symbol, idx)

    return syms


def build_output_symbol_table(input_syms: pynini.SymbolTable) -> pynini.SymbolTable:
    """Create a mutable copy of the input symbol table for output labels.

    The output table will be extended at runtime with <trans> markers
    of the form ``{i}<trans>{j}`` that encode state transitions for
    variation detection.
    """
    return input_syms.copy()
