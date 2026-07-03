"""Standardize pre-standard Irish via the Cadhan intergaelic API (ga->ga).

Returns the modernized (Caighdean) form plus a word-level alignment back to the
original surface tokens, so a parse of the standardized text can be mapped onto
the pre-standard source. Only the standard library is used; results are cached
on disk (sharing the cache layout used by the intergaelic-modernize skill).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

CACHE_DIR = Path.home() / ".cache" / "intergaelic"
API_URL = "https://cadhan.com/api/intergaelic/3.0"

Pair = Tuple[str, str]  # (original, standardized)


def _cache_get(key: str):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _cache_set(key: str, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value))


def request_pairs(text: str, *, offline: bool = False) -> Optional[List[Pair]]:
    """Return [original, standard] pairs, or None if the API was unreachable.

    A cached result is always used when present. With offline=True, a cache miss
    returns None rather than hitting the network.
    """
    key = hashlib.sha256(text.encode()).hexdigest()
    cached = _cache_get(key)
    if cached is not None:
        return [tuple(p) for p in cached]
    if offline:
        return None

    params = urllib.parse.urlencode({"foinse": "ga", "teacs": text})
    req = urllib.request.Request(
        API_URL,
        params.encode("ascii"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                pairs = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < retries:
                print(
                    f"intergaelic HTTP {e.code}, retrying ({attempt}/{retries})",
                    file=sys.stderr,
                )
                time.sleep(2 * attempt)
                continue
            print(f"intergaelic HTTP error: {e.code}", file=sys.stderr)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # covers connection errors and socket read timeouts alike
            if attempt < retries:
                print(
                    f"intergaelic connection problem ({e}), "
                    f"retrying ({attempt}/{retries})",
                    file=sys.stderr,
                )
                time.sleep(2 * attempt)
                continue
            print(f"intergaelic connection error: {e}", file=sys.stderr)
            return None
        except ValueError:
            print("intergaelic returned malformed JSON", file=sys.stderr)
            return None
        _cache_set(key, pairs)
        return [tuple(p) for p in pairs]
    return None


def standardize(text: str, *, offline: bool = False):
    """Return (standard_text, pairs).

    On failure pairs is None and standard_text falls back to the input, so the
    caller can still parse the original (with a warning) instead of crashing.
    """
    pairs = request_pairs(text, offline=offline)
    if not pairs:
        return text, None
    standard = " ".join(std for _, std in pairs).strip()
    return standard, pairs


# status values for an aligned parser token
MATCH = "match"          # confidently traced back to one original token
UNCERTAIN = "uncertain"  # surfaces diverged; the mapping is a guess -> Align=Check


@dataclass
class Mapped:
    """One parser token, tied to the original token it traces back to.

    ``orig_index`` indexes into ``pairs`` so that consecutive parser tokens
    belonging to the *same* original (i.e. a split / multiword token) can be
    grouped even when unrelated originals share a surface form. Every parser
    token is assigned an original: extra standard words produced by
    standardization are a split of some original, never free-standing.
    """

    form: str
    original: Optional[str]
    orig_index: int
    status: str


@dataclass
class Dropped:
    """An original token that standardization deleted (empty standard form).

    The parser never saw it, so there is no parse row for it. ``after_index`` is
    the index into parser_forms of the token it followed in the original (-1 if
    it came first) — i.e. where a human should re-insert a row for it.
    """

    original: str
    after_index: int


@dataclass
class Alignment:
    mapped: List[Mapped] = field(default_factory=list)  # parallel to parser_forms
    dropped: List[Dropped] = field(default_factory=list)


_SPACE_BEFORE = re.compile(r"\s+([.,;:?!)\]])")
_SPACE_AFTER = re.compile(r"([(\[])\s+")


def detokenize(text: str) -> str:
    """Undo the token join's spacing around punctuation, for display.

    The space-separated form is still what the parsers are fed (it keeps their
    tokenization aligned with the intergaelic pairs); this is only for the
    ``# text_standard`` comment.
    """
    return _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", text))


def align(pairs: List[Pair], parser_forms: List[str]) -> Alignment:
    """Align a parse of the standardized text back onto the original tokens.

    Handles the ways original and standard diverge:
      * substitution / 1-to-1  -> MATCH, the parser token belongs to one original
      * expansion / splitting   -> several consecutive parser tokens share one
        original (grouped into a multiword token by the caller)
      * deletion (original token dropped, empty standard) -> recorded in
        ``Alignment.dropped`` for manual re-insertion

    A proper sequence alignment (difflib) is used between the standard word
    stream and the parser tokens, so a single mismatch cannot desynchronise the
    rest of the sentence.
    """
    # 1. Explode pairs into standard *words*, remembering each word's original
    #    token index, and collect originals with empty standard forms (deletions).
    std_words: List[str] = []
    std_orig: List[str] = []
    std_idx: List[int] = []
    dropped_raw: List[Tuple[int, str]] = []  # (std position it sat before, orig)
    for oi, (orig, std) in enumerate(pairs):
        words = std.split()
        if not words:
            dropped_raw.append((len(std_words), orig))
        else:
            for w in words:
                std_words.append(w)
                std_orig.append(orig)
                std_idx.append(oi)

    # -1 = not yet resolved (an inserted parser token, filled from a neighbour)
    mapped: List[Mapped] = [Mapped(f, None, -1, UNCERTAIN) for f in parser_forms]
    std_to_parser: dict = {}

    sm = difflib.SequenceMatcher(a=std_words, b=parser_forms, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                si, pj = i1 + off, j1 + off
                mapped[pj] = Mapped(parser_forms[pj], std_orig[si], std_idx[si], MATCH)
                std_to_parser[si] = pj
        elif tag == "replace":
            # surfaces differ; best-effort map by position, flag for review
            for off, pj in enumerate(range(j1, j2)):
                si = min(i1 + off, i2 - 1)
                mapped[pj] = Mapped(
                    parser_forms[pj], std_orig[si], std_idx[si], UNCERTAIN
                )
            for si in range(i1, i2):
                std_to_parser.setdefault(si, j1)
        elif tag == "delete":
            # standard word(s) the parser merged away; anchor for dropped placement
            for si in range(i1, i2):
                std_to_parser.setdefault(si, max(j1 - 1, -1))
        # 'insert': parser tokens with no standard match — left at orig_index=-1
        # and absorbed into a neighbouring original below.

    _absorb_unresolved(mapped)

    # 2. Resolve where each dropped original should be re-inserted.
    dropped: List[Dropped] = []
    for std_pos, orig in dropped_raw:
        if std_pos in std_to_parser:
            after = std_to_parser[std_pos] - 1
        elif std_pos >= len(std_words):
            after = len(parser_forms) - 1
        else:
            after = -1
        dropped.append(Dropped(orig, after))

    return Alignment(mapped=mapped, dropped=dropped)


def _absorb_unresolved(mapped: List["Mapped"]) -> None:
    """Attach parser tokens with no standard match to a neighbouring original.

    Such a token is part of a split of the adjacent original token, so it joins
    that original's group (becoming part of a multiword token) and is flagged
    UNCERTAIN for review — never left as a free-standing insertion.
    """
    n = len(mapped)
    for k in range(n):
        if mapped[k].orig_index != -1:
            continue
        donor = None
        # prefer the preceding resolved token, else the following one
        for p in range(k - 1, -1, -1):
            if mapped[p].orig_index != -1:
                donor = mapped[p]
                break
        if donor is None:
            for q in range(k + 1, n):
                if mapped[q].orig_index != -1:
                    donor = mapped[q]
                    break
        if donor is not None:
            mapped[k].original = donor.original
            mapped[k].orig_index = donor.orig_index
            mapped[k].status = UNCERTAIN
