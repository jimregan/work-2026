"""Stage 1: combine matching strategies to align chunks to etext."""

import difflib

from .normalize import normalize_for_matching
from .ngram_match import contiguous_ngram_match
from .fuzzy_match import fuzzy_contiguous_match
from .vibevoice import Chunk


BOILERPLATE_THRESHOLD = 40.0


def _words_to_char_span(text: str, words: list[str], start_idx: int, end_idx: int):
    """Convert word indices back to character positions in the original text."""
    pos = 0
    char_start = None
    char_end = None
    for i, word in enumerate(words):
        idx = text.find(word, pos)
        if idx == -1:
            idx = pos
        if i == start_idx:
            char_start = idx
        if i == end_idx - 1:
            char_end = idx + len(word)
        pos = idx + len(word)
    if char_start is None:
        char_start = 0
    if char_end is None:
        char_end = len(text)
    return char_start, char_end


def match_chunk_to_etext(
    chunk_text: str, etext: str, etext_words: list[str], norm_etext: str = None
) -> dict:
    """Match a single chunk's text against the etext.

    Returns a dict with keys: passage, start, end, score, method.
    """
    norm_chunk = normalize_for_matching(chunk_text)
    if norm_etext is None:
        norm_etext = normalize_for_matching(etext)
    norm_etext_words = norm_etext.split()
    norm_chunk_words = norm_chunk.split()

    # Try n-gram match: chunk words in etext
    result = contiguous_ngram_match(norm_etext_words, norm_chunk_words)
    method = "ngram"

    if result is not None:
        start_idx, end_idx = result
        char_start, char_end = _words_to_char_span(
            norm_etext, norm_etext_words, start_idx, end_idx
        )
        # Map normalised positions back to original etext approximately
        # by using word indices on the original etext_words
        if end_idx <= len(etext_words):
            passage = " ".join(etext_words[start_idx:end_idx])
        else:
            passage = norm_etext[char_start:char_end]
        return {
            "passage": passage,
            "start": start_idx,
            "end": end_idx,
            "score": 100.0,
            "method": method,
        }

    # Fall back to fuzzy match
    fuzzy_result = fuzzy_contiguous_match(norm_chunk, norm_etext)
    if fuzzy_result is not None:
        start_idx, end_idx, score = fuzzy_result
        # Map to original etext words
        if end_idx <= len(etext_words):
            passage = " ".join(etext_words[start_idx:end_idx])
        else:
            passage = ""
        return {
            "passage": passage,
            "start": start_idx,
            "end": end_idx,
            "score": score,
            "method": "fuzzy",
        }

    return {
        "passage": "",
        "start": 0,
        "end": 0,
        "score": 0.0,
        "method": "none",
    }


def _generate_diff(vibevoice_text: str, etext_passage: str) -> str:
    """Generate a unified diff between VibeVoice text and matched etext."""
    vv_lines = vibevoice_text.splitlines(keepends=True)
    et_lines = etext_passage.splitlines(keepends=True)
    diff = difflib.unified_diff(et_lines, vv_lines, fromfile="etext", tofile="vibevoice")
    return "".join(diff)


def match_all_chunks(
    chunks: list[Chunk], etext: str
) -> tuple[list[dict], list[dict]]:
    """Match all merged chunks against the etext.

    Returns:
        segments: list of matched segment dicts
        mismatches: list of mismatch dicts with diffs
    """
    norm_etext = normalize_for_matching(etext)
    etext_words = etext.split()

    segments = []
    mismatches = []

    for chunk in chunks:
        match = match_chunk_to_etext(
            chunk.content, etext, etext_words, norm_etext=norm_etext
        )

        is_boilerplate = match["score"] < BOILERPLATE_THRESHOLD

        segment = {
            "start": chunk.start,
            "end": chunk.end,
            "etext": match["passage"],
            "vibevoice": chunk.content,
            "etext_word_start": match["start"],
            "etext_word_end": match["end"],
            "match_score": match["score"],
            "match_method": match["method"],
            "boilerplate": is_boilerplate,
        }
        segments.append(segment)

        # Log mismatch if there is a match but texts differ
        if match["passage"] and not is_boilerplate:
            norm_vv = normalize_for_matching(chunk.content)
            norm_passage = normalize_for_matching(match["passage"])
            if norm_vv != norm_passage:
                diff = _generate_diff(chunk.content, match["passage"])
                mismatches.append({
                    "chunk_start": chunk.start,
                    "chunk_end": chunk.end,
                    "vibevoice": chunk.content,
                    "etext": match["passage"],
                    "diff": diff,
                })

    return segments, mismatches
