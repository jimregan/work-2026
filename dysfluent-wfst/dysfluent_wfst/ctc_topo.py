"""CTC topology construction using pynini."""

from __future__ import annotations

import pynini


def build_ctc_topo(num_tokens: int, syms: pynini.SymbolTable) -> pynini.Fst:
    """Build the standard (non-modified) CTC topology as a pynini FST.

    State 0 = blank state.
    States 1..num_tokens-1 = one per non-blank token.

    Arcs:
      - From state 0: self-loop on blank (0:0);
        arc to state t on input t / output t, for t in 1..num_tokens-1.
      - From state t (non-blank): self-loop on t (t:t);
        arc to state 0 on blank (0:0).

    No direct label-to-different-label transitions; distinct tokens must
    transition through the blank state.

    All weights are 0 (tropical semiring). The FST is arc-sorted on
    output labels for downstream composition.

    Args:
        num_tokens: Total number of tokens including blank (index 0).
        syms: Symbol table matching the CTC model's vocabulary.

    Returns:
        A pynini FST representing the CTC topology.
    """
    compiler = pynini.Compiler()
    blank = 0

    # State 0: blank state
    compiler.add_arc(0, pynini.Arc(blank, blank, 0, 0))  # self-loop
    for t in range(1, num_tokens):
        compiler.add_arc(0, pynini.Arc(t, t, 0, t))  # blank -> token

    # States 1..N-1: non-blank token states
    for t in range(1, num_tokens):
        compiler.add_arc(t, pynini.Arc(t, t, 0, t))  # self-loop
        compiler.add_arc(t, pynini.Arc(blank, blank, 0, 0))  # -> blank

    # All states are final with weight 0
    for s in range(num_tokens):
        compiler.set_final(s, 0)

    fst = compiler.compile()
    fst.set_input_symbols(syms)
    fst.set_output_symbols(syms)
    fst.arcsort("olabel")
    return fst
