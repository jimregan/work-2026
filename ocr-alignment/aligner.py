"""OCR Text Alignment Algorithm

Aligns OCR'd text to a reference/ground-truth text using confidence-weighted
sequential search. Based on the algorithm described in:
  https://link.springer.com/chapter/10.1007/978-3-030-22871-2_58
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass
from typing import List, Optional, Set


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

class DictionaryChecker:
    """Checks whether a word exists in a dictionary.

    Priority:
      1. Custom word set passed at construction.
      2. pyenchant (if installed).
      3. Fallback: always returns False (dictionary check disabled).
    """

    def __init__(
        self,
        word_set: Optional[Set[str]] = None,
        lang: str = "en_US",
    ) -> None:
        self._word_set: Optional[Set[str]] = (
            {w.lower() for w in word_set} if word_set else None
        )
        self._enchant_dict = None
        if self._word_set is None:
            try:
                import enchant  # type: ignore
                self._enchant_dict = enchant.Dict(lang)
            except Exception:
                pass

    def check(self, word: str) -> bool:
        if self._word_set is not None:
            return word.lower() in self._word_set
        if self._enchant_dict is not None:
            return self._enchant_dict.check(word)
        return False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WordAlignment:
    """Result of aligning one OCR word to the reference text."""

    ocr_word: str
    ref_position: int       # index into ref_words
    confidence: float
    failed: bool = False
    ref_word: Optional[str] = None  # the reference word at ref_position


# ---------------------------------------------------------------------------
# Aligner
# ---------------------------------------------------------------------------

class OCRAligner:
    """Aligns a sequence of OCR words to a reference word list.

    Parameters
    ----------
    ref_words:
        Tokenised reference (ground-truth) text.
    dictionary:
        Optional DictionaryChecker. Defaults to auto-detection (pyenchant or disabled).
    window_size:
        Search window X (words searched forward/backward). Default: 20.
    max_edit_distance:
        Maximum Levenshtein distance Y for fuzzy matching. Default: 3.
    max_failures:
        Warn after this many alignment failures Z. Default: 20.
    """

    def __init__(
        self,
        ref_words: List[str],
        dictionary: Optional[DictionaryChecker] = None,
        window_size: int = 20,
        max_edit_distance: int = 3,
        max_failures: int = 20,
    ) -> None:
        self.ref_words = ref_words
        self.dictionary = dictionary or DictionaryChecker()
        self.window_size = window_size
        self.max_edit_distance = max_edit_distance
        self.max_failures = max_failures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(word: str) -> str:
        return word.lower()

    def _eq(self, a: str, b: str) -> bool:
        return self._norm(a) == self._norm(b)

    def _exact_forward(self, word: str, start: int) -> Optional[int]:
        """First exact match in ref_words[start : start + window_size]."""
        end = min(start + self.window_size, len(self.ref_words))
        for i in range(start, end):
            if self._eq(word, self.ref_words[i]):
                return i
        return None

    def _exact_backward(self, word: str, before: int) -> Optional[int]:
        """Last exact match in ref_words[before - window_size : before] (reversed)."""
        lo = max(0, before - self.window_size)
        for i in range(before - 1, lo - 1, -1):
            if self._eq(word, self.ref_words[i]):
                return i
        return None

    def _fuzzy_forward(self, word: str, start: int, max_dist: int) -> Optional[int]:
        """First fuzzy match (edit distance <= max_dist) forward from start."""
        end = min(start + self.window_size, len(self.ref_words))
        norm = self._norm(word)
        for i in range(start, end):
            if levenshtein(norm, self._norm(self.ref_words[i])) <= max_dist:
                return i
        return None

    def _fuzzy_backward(self, word: str, before: int, max_dist: int) -> Optional[int]:
        """First fuzzy match (edit distance <= max_dist) backward from before."""
        lo = max(0, before - self.window_size)
        norm = self._norm(word)
        for i in range(before - 1, lo - 1, -1):
            if levenshtein(norm, self._norm(self.ref_words[i])) <= max_dist:
                return i
        return None

    # ------------------------------------------------------------------
    # Main alignment loop
    # ------------------------------------------------------------------

    def align(self, ocr_words: List[str]) -> List[WordAlignment]:
        """Align *ocr_words* to :attr:`ref_words`.

        Returns a list of WordAlignment objects, one per OCR word.
        """
        alignments: List[WordAlignment] = []
        last_pos = -1       # ref_words index of the last successfully aligned word
        failure_count = 0

        for ocr_word in ocr_words:
            confidence = 0.0

            # ----------------------------------------------------------
            # Step 1 — Dictionary check
            # ----------------------------------------------------------
            if self.dictionary.check(ocr_word):
                confidence += 0.5

            confidence_after_dict = confidence
            next_pos = last_pos + 1  # expected next position in reference

            # ----------------------------------------------------------
            # Step 2 — Immediate next word
            # ----------------------------------------------------------
            if next_pos < len(self.ref_words):
                if self._eq(ocr_word, self.ref_words[next_pos]):
                    # Hit: reward this word and the previous one
                    confidence += 0.25
                    if alignments:
                        alignments[-1].confidence += 0.25
                    last_pos = next_pos
                    alignments.append(WordAlignment(
                        ocr_word=ocr_word,
                        ref_position=last_pos,
                        confidence=confidence,
                        ref_word=self.ref_words[last_pos],
                    ))
                    continue

                # Miss: penalise this word (and previous if it was a dict word)
                confidence -= 0.25
                if confidence_after_dict == 0.5 and alignments:
                    alignments[-1].confidence -= 0.25

            # ----------------------------------------------------------
            # Step 3 — Exact forward search (skip already-checked next_pos)
            # ----------------------------------------------------------
            pos = self._exact_forward(ocr_word, next_pos + 1)
            if pos is not None:
                confidence += 0.25
                last_pos = pos
                alignments.append(WordAlignment(
                    ocr_word=ocr_word,
                    ref_position=last_pos,
                    confidence=confidence,
                    ref_word=self.ref_words[last_pos],
                ))
                continue

            # Exact backward search
            if last_pos > 0:
                pos = self._exact_backward(ocr_word, last_pos)
                if pos is not None:
                    confidence += 0.1
                    if alignments:
                        alignments[-1].confidence -= 0.4
                    last_pos = pos
                    alignments.append(WordAlignment(
                        ocr_word=ocr_word,
                        ref_position=last_pos,
                        confidence=confidence,
                        ref_word=self.ref_words[last_pos],
                    ))
                    continue

            # ----------------------------------------------------------
            # Step 4 — Fuzzy forward search, iterating edit distance 1…Y
            # ----------------------------------------------------------
            found = False
            for ed in range(1, self.max_edit_distance + 1):
                pos = self._fuzzy_forward(ocr_word, next_pos, ed)
                if pos is not None:
                    confidence += 0.1
                    last_pos = pos
                    alignments.append(WordAlignment(
                        ocr_word=ocr_word,
                        ref_position=last_pos,
                        confidence=confidence,
                        ref_word=self.ref_words[last_pos],
                    ))
                    found = True
                    break

            if found:
                continue

            # ----------------------------------------------------------
            # Step 5 — Fuzzy backward search, iterating edit distance 1…Y
            # ----------------------------------------------------------
            if last_pos > 0:
                for ed in range(1, self.max_edit_distance + 1):
                    pos = self._fuzzy_backward(ocr_word, last_pos, ed)
                    if pos is not None:
                        confidence += 0.1
                        last_pos = pos
                        alignments.append(WordAlignment(
                            ocr_word=ocr_word,
                            ref_position=last_pos,
                            confidence=confidence,
                            ref_word=self.ref_words[last_pos],
                        ))
                        found = True
                        break

            if found:
                continue

            # ----------------------------------------------------------
            # Step 6 — Alignment failure
            # ----------------------------------------------------------
            failure_count += 1
            last_pos = min(last_pos + 1, len(self.ref_words) - 1)
            alignments.append(WordAlignment(
                ocr_word=ocr_word,
                ref_position=last_pos,
                confidence=-1.0,
                failed=True,
                ref_word=self.ref_words[last_pos] if self.ref_words else None,
            ))

            if failure_count > self.max_failures:
                warnings.warn(
                    f"More than {self.max_failures} alignment failures. "
                    "The alignment may have lost its way due to out-of-order text "
                    "or other unforeseen circumstances.",
                    stacklevel=2,
                )

        return alignments


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def tokenise(text: str) -> List[str]:
    """Whitespace tokeniser that keeps only word characters and apostrophes."""
    return re.findall(r"[A-Za-z0-9'\u2018\u2019\u201b]+", text)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Align OCR'd text to a reference/ground-truth text.",
    )
    parser.add_argument("ocr_file", help="File containing OCR'd text")
    parser.add_argument("ref_file", help="File containing reference/ground-truth text")
    parser.add_argument(
        "--wordlist", metavar="FILE",
        help="Plain-text word list (one word per line) for dictionary check",
    )
    parser.add_argument(
        "--window", type=int, default=20, metavar="N",
        help="Search window size (default: 20)",
    )
    parser.add_argument(
        "--max-edit-distance", type=int, default=3, metavar="Y",
        help="Maximum edit distance for fuzzy matching (default: 3)",
    )
    parser.add_argument(
        "--max-failures", type=int, default=20, metavar="Z",
        help="Warn after this many failures (default: 20)",
    )
    parser.add_argument(
        "--lang", default="en_US",
        help="Language for pyenchant dictionary (default: en_US)",
    )
    parser.add_argument(
        "--format", choices=["tsv", "json"], default="tsv",
        dest="output_format",
        help="Output format (default: tsv)",
    )
    args = parser.parse_args(argv)

    with open(args.ocr_file, encoding="utf-8") as fh:
        ocr_words = tokenise(fh.read())
    with open(args.ref_file, encoding="utf-8") as fh:
        ref_words = tokenise(fh.read())

    word_set: Optional[Set[str]] = None
    if args.wordlist:
        with open(args.wordlist, encoding="utf-8") as fh:
            word_set = {line.strip().lower() for line in fh if line.strip()}

    dictionary = DictionaryChecker(word_set=word_set, lang=args.lang)
    aligner = OCRAligner(
        ref_words=ref_words,
        dictionary=dictionary,
        window_size=args.window,
        max_edit_distance=args.max_edit_distance,
        max_failures=args.max_failures,
    )
    results = aligner.align(ocr_words)

    if args.output_format == "json":
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        print("ocr_word\tref_word\tref_position\tconfidence\tfailed")
        for r in results:
            print(
                f"{r.ocr_word}\t{r.ref_word or ''}\t{r.ref_position}"
                f"\t{r.confidence:.3f}\t{r.failed}"
            )


if __name__ == "__main__":
    main()
