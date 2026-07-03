"""Reference FST with variation arcs (pynini)."""

from __future__ import annotations

import math
from typing import Optional

import pynini
import torch


def dynamic_error_probability(error_score: float, distance: int) -> float:
    """Return the distance-scaled non-forward transition probability."""
    return (
        error_score
        * (1 / math.sqrt(2 * math.pi))
        * math.exp(-(distance ** 2) / 2)
    )


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

    States 0..L represent positions in the phoneme sequence (L = len(phoneme_ids)).
    State L is the final state.

    Arc types:
      (a) Correct transition i -> i+1: weight = -log(alpha)
      (b) Substitution arcs i -> i+1 (if sub=True): top-2 similar phonemes
      (c) Skip/deletion arcs i -> j where j > i+1, j-i <= 3 (if skip=True)
      (d) Back/repetition arcs i -> j where j < i, i-j <= 2 (if back=True)

    <trans> markers are added to output_syms dynamically.
    """
    alpha = 1 - 10 ** (-beta)
    error_score = 1 - alpha

    fst = pynini.Fst()

    L = len(phoneme_ids)
    # States 0..L (state L is the final state).
    for _ in range(L + 1):
        fst.add_state()
    fst.set_start(0)

    next_osym_id = output_syms.num_symbols()

    # Precompute reverse index map for substitution arc lookups
    idx2phn: dict[int, str] = {}
    if sub and phn2idx is not None:
        idx2phn = {v: k for k, v in phn2idx.items()}

    specials = frozenset({
        "|", "-", "<pad>", "<s>", "</s>",
        "<unk>", "SIL", "SPN", "<blank>", "<b>",
    })

    for i, phone in enumerate(phoneme_ids):
        for j in range(L + 1):
            if i == j:
                continue

            if j == i + 1:
                # (a) Correct transition
                w = -math.log(alpha) if alpha > 0 else 0.0
                fst.add_arc(i, pynini.Arc(phone, phone, w, j))

                if (sub and error_score > 0 and similarity_matrix is not None
                        and lexicon is not None and phn2idx is not None):
                    # (b) Substitution arcs
                    phone_text = lexicon[phone]
                    if is_ipa and ipa_to_cmu_fn is not None:
                        cmu_phone = ipa_to_cmu_fn(phone_text)
                    else:
                        cmu_phone = phone_text

                    if cmu_phone in phn2idx:
                        sim_idx = phn2idx[cmu_phone]
                        sim_row = similarity_matrix[sim_idx]
                        if not isinstance(sim_row, torch.Tensor):
                            sim_row = torch.tensor(sim_row)
                        top2 = torch.topk(sim_row, 2).indices

                        for sid in top2:
                            sid_val = sid.item()
                            if sid_val == sim_idx:
                                continue
                            sim_phoneme = idx2phn.get(sid_val)
                            if sim_phoneme is None or sim_phoneme in specials:
                                continue

                            sim_pid = input_syms.find(sim_phoneme)
                            if sim_pid == -1:
                                continue

                            w_sub = -math.log(error_score / 10000)
                            fst.add_arc(
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
                        dynamic_error_probability(error_score, abs(i - j))
                    )
                    fst.add_arc(i, pynini.Arc(0, mid, w_skip, j))
                elif j < i and back and i - j <= 2:
                    # (d) Back / repetition
                    marker = f"{i}<trans>{j}"
                    mid = output_syms.find(marker)
                    if mid == -1:
                        mid = output_syms.add_symbol(marker, next_osym_id)
                        next_osym_id += 1
                    w_back = -math.log(
                        dynamic_error_probability(error_score, abs(i - j))
                    )
                    fst.add_arc(i, pynini.Arc(0, mid, w_back, j))

    fst.set_final(L, 0)

    fst.set_input_symbols(input_syms)
    fst.set_output_symbols(output_syms)
    return fst
