"""Fuzzy matching using rapidfuzz."""

from rapidfuzz.fuzz import partial_ratio_alignment
from nltk.tokenize import TreebankWordTokenizer

_tokenizer = TreebankWordTokenizer()


def _char_to_word_index(text: str, char_idx: int, spans: list[tuple[int, int]]) -> int:
    """Convert a character index to a word index using pre-computed spans."""
    for i, (start, end) in enumerate(spans):
        if char_idx < end:
            return i
    return len(spans) - 1


def fuzzy_contiguous_match(
    text_a: str, text_b: str, threshold: int = 55
) -> tuple[int, int, float] | None:
    """Find the best fuzzy match of text_a within text_b.

    Uses rapidfuzz partial_ratio_alignment to locate text_a inside text_b.
    Returns (start_word_idx, end_word_idx, score) into text_b, or None if
    the score is below threshold.
    """
    result = partial_ratio_alignment(text_a, text_b)
    if result is None or result.score < threshold:
        return None

    spans = list(_tokenizer.span_tokenize(text_b))
    if not spans:
        return None

    start_word = _char_to_word_index(text_b, result.dest_start, spans)
    end_word = _char_to_word_index(text_b, result.dest_end, spans)

    return (start_word, end_word + 1, result.score)
