"""State trajectory analysis and variation classification.

Port of the original decoder's ``_deduplicate_and_filter``,
``extract_phoneme_states``, and ``detect_dysfluency`` methods.
"""

from __future__ import annotations

import pynini


# Tokens to filter out during deduplication
_SPECIAL_TOKENS = frozenset({
    "|", "-", "<pad>", "<s>", "</s>", "<unk>",
    "SIL", "SPN", "STL", "<blank>", "<b>",
})


def deduplicate_and_filter(
    label_ids: list[int],
    output_syms: pynini.SymbolTable,
) -> list[str]:
    """Consecutive deduplication and special-token filtering.

    Converts label IDs to symbol strings, removes consecutive
    duplicates, and filters out special tokens.

    Args:
        label_ids: Output label IDs from the shortest path.
        output_syms: Symbol table for ID-to-string lookup.

    Returns:
        Filtered list of phoneme/marker strings.
    """
    filtered: list[str] = []
    prev_label: str | None = None

    for lid in label_ids:
        symbol = output_syms.find(lid)
        if symbol == "" or symbol is None:
            continue
        if symbol != prev_label and symbol not in _SPECIAL_TOKENS:
            filtered.append(symbol)
        prev_label = symbol

    return filtered


def merge_trans_markers(phoneme_list: list[str]) -> list[str]:
    """Merge consecutive ``<trans>`` markers.

    If two ``<trans>`` tokens appear consecutively, keep the source of
    the first and the destination of the last. For example:
    ``3<trans>5`` followed by ``5<trans>7`` becomes ``3<trans>7``.

    Args:
        phoneme_list: Deduplicated/filtered symbol list.

    Returns:
        List with consecutive <trans> markers merged.
    """
    merged: list[str] = []
    current_merge: str | None = None

    for item in phoneme_list:
        if "<trans>" in item:
            if current_merge is None:
                current_merge = item
            else:
                # Keep source of first, destination of last
                src = current_merge.split("<trans>")[0]
                dst = item.split("<trans>")[-1]
                current_merge = f"{src}<trans>{dst}"
        else:
            if current_merge is not None:
                merged.append(current_merge)
                current_merge = None
            merged.append(item)

    if current_merge is not None:
        merged.append(current_merge)

    return merged


def build_state_trajectory(
    merged_seq: list[str],
    ref_phonemes: list[str] | None = None,
) -> list[dict]:
    """Build a state trajectory and classify each element.

    Walks through the merged sequence. ``<trans>`` tokens update the
    current state; phoneme tokens occupy ``(current_state, current_state+1)``.

    Classification:
      - **repetition**: start_state already in history
      - **insertion**: start_state < min(history)
      - **deletion**: start_state > prev_end + 1 (gap in states)
      - **normal**: standard forward transition

    Args:
        merged_seq: Merged phoneme/marker sequence from
            ``merge_trans_markers``.
        ref_phonemes: Optional reference phoneme list (for future
            substitution detection based on position comparison).

    Returns:
        List of dicts with keys: phoneme, start_state, end_state,
        variation_type.
    """
    results: list[dict] = []
    state_history: set[int] = set()
    prev_end: int = -1

    # First pass: extract (start, end, phoneme) tuples
    clean_states: list[tuple[int, int, str]] = []
    current_state = 0

    for elem in merged_seq:
        if "<trans>" in elem:
            _, j_str = elem.split("<trans>")
            current_state = int(j_str)
        else:
            start = current_state
            end = start + 1
            clean_states.append((start, end, elem))
            current_state = end

    # Second pass: classify each phoneme
    for start, end, phoneme in clean_states:
        min_hist = min(state_history) if state_history else -1

        if start in state_history:
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "variation_type": "repetition",
            })
        elif start < min_hist:
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "variation_type": "insertion",
            })
        elif start > prev_end + 1:
            # Gap: insert a deletion marker, then the normal phoneme
            results.append({
                "phoneme": "<del>",
                "start_state": prev_end,
                "end_state": start,
                "variation_type": "deletion",
            })
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "variation_type": "normal",
            })
        else:
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "variation_type": "normal",
            })

        state_history.add(start)
        prev_end = end

    return results
