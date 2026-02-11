"""Pynini-to-k2 bridge: FST text conversion, DenseFsaVec, intersection."""

from __future__ import annotations

import math

import pynini
import torch
import torch.nn.functional as F


def fst_to_k2_str(fst: pynini.Fst, to_log_probs: bool = True) -> str:
    """Convert a pynini FST to k2 text format.

    k2's ``Fsa.from_str()`` expects lines of the form:
        ``src_state dest_state input_label output_label weight``
    with the last line being just the superfinal state number.

    Tropical weights in pynini are ``-log(prob)``. When ``to_log_probs``
    is True, weights are converted back to raw probabilities via
    ``exp(-w)`` for k2 (which uses raw scores internally).

    Args:
        fst: A compiled pynini FST.
        to_log_probs: If True, convert tropical weights back to
            raw probabilities.

    Returns:
        A string in k2 FSA text format.
    """
    lines = []
    superfinal = fst.num_states()

    for state in fst.states():
        for arc in fst.arcs(state):
            w = float(arc.weight)
            if to_log_probs:
                w = math.exp(-w)
            lines.append(
                f"{state} {arc.nextstate} {arc.ilabel} {arc.olabel} {w}"
            )

        final_w = fst.final(state)
        if final_w != pynini.Weight.zero("tropical"):
            w = float(final_w)
            if to_log_probs:
                w = math.exp(-w)
            lines.append(f"{state} {superfinal} -1 -1 {w}")

    lines.append(str(superfinal))
    return "\n".join(lines)


def create_dense_fsa_vec(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
) -> "k2.DenseFsaVec":
    """Create a k2 DenseFsaVec from model output log-probabilities.

    Applies log_softmax normalisation and builds supervision segments.

    Args:
        log_probs: Tensor of shape (B, T, C) — batch of frame-level
            logits/log-probs from the CTC model.
        lengths: Tensor of shape (B,) — valid frame count per sample.

    Returns:
        A k2.DenseFsaVec for dense intersection.

    Raises:
        ValueError: If tensor dimensions are incompatible.
    """
    import k2

    if log_probs.ndim != 3:
        raise ValueError(
            f"log_probs must be 3D (B, T, C), got {log_probs.ndim}D"
        )

    B, T, C = log_probs.shape
    if lengths.shape[0] != B:
        raise ValueError(
            f"lengths batch size {lengths.shape[0]} != log_probs batch size {B}"
        )

    lengths = lengths.to(dtype=torch.int32)
    log_probs = F.log_softmax(log_probs, dim=-1)

    supervision_segments = torch.tensor(
        [[i, 0, lengths[i].item()] for i in range(B)],
        dtype=torch.int32,
    )

    return k2.DenseFsaVec(log_probs, supervision_segments)


def intersect_and_decode(
    fst_str: str,
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    device: str = "cpu",
    output_beam: float = 25.0,
) -> list[int]:
    """Import FST into k2, intersect with dense log-probs, return best path.

    This is the k2-only portion of the pipeline: it takes the text-format
    FST (from ``fst_to_k2_str``), builds a DenseFsaVec from the neural
    network output, performs dense intersection, and extracts the
    shortest path's auxiliary (output) labels.

    Args:
        fst_str: FST in k2 text format (from ``fst_to_k2_str``).
        log_probs: Tensor of shape (1, T, C) — single utterance.
        lengths: Tensor of shape (1,) — valid frame count.
        device: Torch device string.
        output_beam: Beam width for ``k2.intersect_dense``.

    Returns:
        List of output label IDs from the best path (excluding the
        final -1 label).
    """
    import k2

    dev = torch.device(device)

    # Import composed FST into k2
    composed_k2 = k2.Fsa.from_str(fst_str, acceptor=False)
    composed_k2 = k2.arc_sort(composed_k2).to(dev)

    # Build dense FSA from log-probs
    dense_fsa = create_dense_fsa_vec(
        log_probs.to(dev), lengths.to(dev)
    )

    # Intersect and find shortest path
    lattice = k2.intersect_dense(composed_k2, dense_fsa, output_beam=output_beam)
    shortest = k2.shortest_path(lattice, use_double_scores=True)

    # Extract output (aux) labels, dropping the final -1
    aux_labels = shortest[0].aux_labels.tolist()
    if aux_labels and aux_labels[-1] == -1:
        aux_labels = aux_labels[:-1]

    return aux_labels
