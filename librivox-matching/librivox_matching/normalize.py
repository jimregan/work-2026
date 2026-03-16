"""Text normalisation for matching."""

import re
import unicodedata

from num2words import num2words


def _verbalize_numbers(text: str) -> str:
    """Replace digit sequences with their English word forms."""
    def _replace_match(m):
        num_str = m.group(0)
        try:
            n = int(num_str)
            if n > 999999:
                return num_str
            return num2words(n, lang="en")
        except (ValueError, OverflowError):
            return num_str

    return re.sub(r"\d+", _replace_match, text)


def normalize_for_matching(text: str) -> str:
    """Normalise text for matching.

    Steps: lowercase, NFKC, replace hyphens with spaces, remove punctuation,
    verbalize numbers, collapse whitespace.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = _verbalize_numbers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
