from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

SILENCE_PHONES: frozenset[str] = frozenset({
    "h#", "pau", "epi",
    "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
})

TIMIT_61_PHONES: frozenset[str] = frozenset({
    "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao",
    "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h",
    "jh", "ch", "b", "d", "g", "p", "t", "k", "dx", "s",
    "sh", "z", "zh", "f", "th", "v", "dh", "m", "n", "ng",
    "em", "nx", "en", "eng", "l", "r", "w", "y", "hh", "hv",
    "el", "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
    "q", "pau", "epi", "h#",
})

# TIMIT phone → IPA, for Charsiu
TIMIT_TO_IPA: dict[str, str] = {
    "iy": "iː", "ih": "ɪ", "eh": "ɛ", "ey": "eɪ", "ae": "æ",
    "aa": "ɑ", "aw": "aʊ", "ay": "aɪ", "ah": "ʌ", "ao": "ɔ",
    "oy": "ɔɪ", "ow": "oʊ", "uh": "ʊ", "uw": "uː", "ux": "ʉ",
    "er": "ɝ", "ax": "ə", "ix": "ɨ", "axr": "ɚ", "ax-h": "ə̥",
    "jh": "dʒ", "ch": "tʃ",
    "b": "b", "d": "d", "g": "ɡ", "p": "p", "t": "t", "k": "k",
    "dx": "ɾ", "q": "ʔ",
    "s": "s", "sh": "ʃ", "z": "z", "zh": "ʒ",
    "f": "f", "th": "θ", "v": "v", "dh": "ð",
    "m": "m", "n": "n", "ng": "ŋ", "em": "m̩", "nx": "ɾ̃", "en": "n̩", "eng": "ŋ̩",
    "l": "l", "r": "ɹ", "w": "w", "y": "j",
    "hh": "h", "hv": "ɦ", "el": "l̩",
    "bcl": "b̚", "dcl": "d̚", "gcl": "ɡ̚", "pcl": "p̚", "tcl": "t̚", "kcl": "k̚",
    "pau": "∅", "epi": "∅", "h#": "∅",
}


def parse_phn_file(path: str | Path) -> list[tuple[int, int, str]]:
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                entries.append((int(parts[0]), int(parts[1]), parts[2]))
    return entries


def parse_wrd_file(path: str | Path) -> list[tuple[int, int, str]]:
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                entries.append((int(parts[0]), int(parts[1]), parts[2]))
    return entries


def load_timit_lexicon(path: str | Path) -> dict[str, list[list[str]]]:
    """Load TIMIT .dic file into {word: [[phones], ...]} with alternate pronunciations."""
    lexicon: dict[str, list[list[str]]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower().rstrip("(0123456789)")
            phones = parts[1:]
            lexicon.setdefault(word, []).append(phones)
    return lexicon


def extract_file_lexicon(
    phn_path: str | Path,
    wrd_path: str | Path,
    silence_phones: frozenset[str] = SILENCE_PHONES,
) -> dict[str, list[list[str]]]:
    """
    Build a per-file pronunciation lexicon from TIMIT ground-truth files.

    Each phone is assigned to the word whose interval contains the phone's
    midpoint (start + end) // 2. This handles sandhi across word boundaries
    without double-counting.

    Returns {word: [[phones_occurrence_1], [phones_occurrence_2], ...]} where
    multiple entries exist only when the same word type appears more than once
    with a different realised phone sequence.
    """
    phones = parse_phn_file(phn_path)
    words = parse_wrd_file(wrd_path)

    result: dict[str, list[list[str]]] = {}
    for w_start, w_end, word in words:
        word_phones = []
        for p_start, p_end, phone in phones:
            midpoint = (p_start + p_end) // 2
            if w_start <= midpoint < w_end and phone not in silence_phones:
                word_phones.append(phone)

        if not word_phones:
            continue

        existing = result.get(word, [])
        if word_phones not in existing:
            existing.append(word_phones)
        result[word] = existing

    return result


def iter_timit_files(timit_root: str | Path) -> Iterator[tuple[Path, Path, Path]]:
    """Yield (wav_path, phn_path, wrd_path) for every TIMIT utterance."""
    root = Path(timit_root)
    for wav in sorted(root.rglob("*.wav")):
        phn = wav.with_suffix(".phn")
        wrd = wav.with_suffix(".wrd")
        if phn.exists() and wrd.exists():
            yield wav, phn, wrd


def get_word_sequence(wrd_path: str | Path) -> list[str]:
    """Return ordered list of words from a .wrd file (for transcript)."""
    return [w for _, _, w in parse_wrd_file(wrd_path)]
