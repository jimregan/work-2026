"""Reference FST with variation arcs (pynini)."""

from __future__ import annotations

import math
from typing import Optional

import pynini


def build_ref_fst(
    phoneme_ids: list[int],
    beta: float,
    input_syms: pynini.SymbolTable,
    output_syms: pynini.SymbolTable,
    skip: bool = False,
    back: bool = True,
    sub: bool = True,
    similarity_matrix=None,
    phn2idx: Optional[dict[str, int]] = None,
    lexicon: Optional[list[str]] = None,
    is_ipa: bool = False,
    ipa_to_cmu_fn=None,
) -> pynini.Fst:
    """Build the reference FST encoding expected phonemes plus variation paths.

    This is a port of the original ``create_fsa_graph()`` using pynini's
    Compiler. Weights are in the tropical semiring (``-log(prob)``).

    States 0..L represent positions in the phoneme sequence (L = len(phoneme_ids)).
    State L is the final state.

    Arc types:
      (a) Correct transition i -> i+1: weight = -log(alpha)
      (b) Substitution arcs i -> i+1 (if sub=True): top-2 similar phonemes
      (c) Skip/deletion arcs i -> j where j > i+1, j-i <= 3 (if skip=True)
      (d) Back/repetition arcs i -> j where j < i, i-j <= 2 (if back=True)

    <trans> markers are added to output_syms dynamically.

    Args:
        phoneme_ids: List of phoneme indices (into the symbol table).
        beta: Variation tolerance. alpha = 1 - 10^(-beta).
        input_syms: Input symbol table (from wav2vec2 vocab).
        output_syms: Output symbol table (mutable; will be extended
            with <trans> markers).
        skip: Allow skip/deletion arcs.
        back: Allow back/repetition arcs.
        sub: Allow substitution arcs.
        similarity_matrix: Phoneme similarity matrix (torch.Tensor or
            numpy array). Required if sub=True.
        phn2idx: Mapping from phoneme string to similarity matrix index.
            Required if sub=True.
        lexicon: List mapping phoneme indices to phoneme strings.
            Required if sub=True.
        is_ipa: Whether the lexicon uses IPA (needs conversion for
            similarity matrix lookup).
        ipa_to_cmu_fn: Function to convert IPA phoneme to CMU. Required
            if is_ipa=True and sub=True.

    Returns:
        A pynini FST representing the reference graph with variation arcs.
    """
    alpha = 1 - 10 ** (-beta)
    error_score = 1 - alpha

    compiler = pynini.Compiler(
        isymbols=input_syms,
        osymbols=output_syms,
    )

    L = len(phoneme_ids)
    next_osym_id = output_syms.num_symbols()

    for i, phone in enumerate(phoneme_ids):
        for j in range(L + 1):
            if i == j:
                continue

            if j == i + 1:
                # (a) Correct transition
                w = -math.log(alpha) if alpha > 0 else 0.0
                compiler.add_arc(i, pynini.Arc(phone, phone, w, j))

                if skip:
                    # Skip arc on the correct transition position
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.find(marker)
                    if mid == -1:
                        mid = output_syms.add_symbol(marker, next_osym_id)
                        next_osym_id += 1
                    w_err = -math.log(
                        error_score * math.exp(-((i - j) ** 2) / 2)
                    )
                    compiler.add_arc(i, pynini.Arc(0, mid, w_err, j))

                if sub and similarity_matrix is not None and lexicon is not None and phn2idx is not None:
                    # (b) Substitution arcs
                    phone_text = lexicon[phone]
                    if is_ipa and ipa_to_cmu_fn is not None:
                        cmu_phone = ipa_to_cmu_fn(phone_text)
                    else:
                        cmu_phone = phone_text

                    if cmu_phone in phn2idx:
                        import torch

                        sim_idx = phn2idx[cmu_phone]
                        sim_row = similarity_matrix[sim_idx]
                        if not isinstance(sim_row, torch.Tensor):
                            sim_row = torch.tensor(sim_row)
                        top2 = torch.topk(sim_row, 2).indices

                        specials = {
                            "|", "-", "<pad>", "<s>", "</s>",
                            "<unk>", "SIL", "SPN", "<blank>", "<b>",
                        }

                        for sid in top2:
                            sid_val = sid.item()
                            if sid_val == sim_idx:
                                continue
                            # Reverse lookup in phn2idx
                            sim_phoneme = None
                            for k, v in phn2idx.items():
                                if v == sid_val:
                                    sim_phoneme = k
                                    break
                            if sim_phoneme is None or sim_phoneme in specials:
                                continue

                            # Map similar phoneme to its input-symbol ID
                            sim_pid = input_syms.find(sim_phoneme)
                            if sim_pid == -1:
                                # No corresponding symbol in input_syms; skip
                                continue

                            w_sub = -math.log(error_score / 10000)
                            compiler.add_arc(
                                i, pynini.Arc(sim_pid, sim_pid, w_sub, j)
                            )
            else:
                if alpha == 1:
                    continue
                if j > i and skip and j - i <= 3:
                    # (c) Skip / deletion
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.find(marker)
                    if mid == -1:
                        mid = output_syms.add_symbol(marker, next_osym_id)
                        next_osym_id += 1
                    w_skip = -math.log(
                        error_score * math.exp(-((i - j) ** 2) / 2)
                    )
                    compiler.add_arc(i, pynini.Arc(0, mid, w_skip, j))
                elif j < i and back and i - j <= 2:
                    # (d) Back / repetition
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.find(marker)
                    if mid == -1:
                        mid = output_syms.add_symbol(marker, next_osym_id)
                        next_osym_id += 1
                    w_back = -math.log(
                        error_score * math.exp(-((i - j) ** 2) / 2)
                    )
                    compiler.add_arc(i, pynini.Arc(0, mid, w_back, j))

    # Final state
    compiler.set_final(L, 0)

    fst = compiler.compile()
    return fst
