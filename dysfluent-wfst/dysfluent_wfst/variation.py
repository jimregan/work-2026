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


def collapse_label_runs(
    label_ids: list[int],
    output_syms: pynini.SymbolTable,
) -> list[dict]:
    """Collapse consecutive labels into runs with frame spans.

    Converts label IDs to symbol strings, removes consecutive duplicates,
    filters out special tokens, and retains the frame interval for each
    surviving run.

    Args:
        label_ids: Output label IDs from the shortest path.
        output_syms: Symbol table for ID-to-string lookup.

    Returns:
        List of dicts with keys ``symbol``, ``start_frame``, ``end_frame``.
        ``end_frame`` is exclusive.
    """
    runs: list[dict] = []
    prev_symbol: str | None = None
    run_start = 0

    for frame_idx, lid in enumerate(label_ids):
        symbol = output_syms.find(lid)
        if symbol == "" or symbol is None:
            continue

        if prev_symbol is None:
            prev_symbol = symbol
            run_start = frame_idx
            continue

        if symbol != prev_symbol:
            if prev_symbol not in _SPECIAL_TOKENS:
                runs.append({
                    "symbol": prev_symbol,
                    "start_frame": run_start,
                    "end_frame": frame_idx,
                })
            prev_symbol = symbol
            run_start = frame_idx

    if prev_symbol is not None and prev_symbol not in _SPECIAL_TOKENS:
        runs.append({
            "symbol": prev_symbol,
            "start_frame": run_start,
            "end_frame": len(label_ids),
        })

    return runs


def merge_trans_markers(label_runs: list[dict]) -> list[dict]:
    """Merge consecutive ``<trans>`` markers.

    If two ``<trans>`` tokens appear consecutively, keep the source of
    the first and the destination of the last. For example:
    ``3<trans>5`` followed by ``5<trans>7`` becomes ``3<trans>7``.

    Args:
        label_runs: Collapsed run list from ``collapse_label_runs``.

    Returns:
        Run list with consecutive <trans> markers merged.
    """
    merged: list[dict] = []
    current_merge: dict | None = None

    for item in label_runs:
        symbol = item["symbol"]
        if "<trans>" in symbol:
            if current_merge is None:
                current_merge = dict(item)
            else:
                # Keep source of first, destination of last
                src = current_merge["symbol"].split("<trans>")[0]
                dst = symbol.split("<trans>")[-1]
                current_merge["symbol"] = f"{src}<trans>{dst}"
                current_merge["end_frame"] = item["end_frame"]
        else:
            if current_merge is not None:
                merged.append(current_merge)
                current_merge = None
            merged.append(item)

    if current_merge is not None:
        merged.append(current_merge)

    return merged


def build_state_trajectory(
    merged_seq: list[dict],
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
        symbol = elem["symbol"]
        if "<trans>" in symbol:
            _, j_str = symbol.split("<trans>")
            current_state = int(j_str)
        else:
            start = current_state
            end = start + 1
            clean_states.append((
                start,
                end,
                symbol,
                elem["start_frame"],
                elem["end_frame"],
            ))
            current_state = end

    # Second pass: classify each phoneme
    for start, end, phoneme, start_frame, end_frame in clean_states:
        min_hist = min(state_history) if state_history else -1
        expected = (
            ref_phonemes[start]
            if ref_phonemes is not None and 0 <= start < len(ref_phonemes)
            else None
        )

        if start in state_history:
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "variation_type": "repetition",
            })
        elif start < min_hist:
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "variation_type": "insertion",
            })
        elif start > prev_end + 1:
            # Gap: insert a deletion marker, then the normal phoneme
            results.append({
                "phoneme": "<del>",
                "start_state": prev_end,
                "end_state": start,
                "start_frame": start_frame,
                "end_frame": start_frame,
                "variation_type": "deletion",
            })
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "variation_type": (
                    "substitution" if expected is not None and phoneme != expected
                    else "normal"
                ),
            })
        else:
            results.append({
                "phoneme": phoneme,
                "start_state": start,
                "end_state": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "variation_type": (
                    "substitution" if expected is not None and phoneme != expected
                    else "normal"
                ),
            })

        state_history.add(start)
        prev_end = end

    return results
