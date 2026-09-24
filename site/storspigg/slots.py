"""Extract the variable slots from the MTM talking-book announcement.

The announcement is a fixed template; everything that varies between books sits
in a small number of slots, so correction is mostly a matter of filling those
from the two ASR systems plus the metadata.
"""
import json
import os
import re

import svnum


def norm_tokens(text):
    """Lowercase, drop punctuation, canonicalise numerals.

    Letters are kept by Unicode category rather than an explicit charset: a
    hand-listed one silently ate the 'æ' in the narrator name Sædén.
    """
    t = text.lower().replace("’", "'")
    t = "".join(c if (c.isalpha() or c.isdigit() or c.isspace()) else " " for c in t)
    return [svnum.canon(w) for w in t.split()]


def load_whisperx(path):
    j = json.load(open(path))
    return " ".join(s["text"] for s in j["segments"])


def load_wav2vec(path):
    return json.load(open(path))["text"]


def _between(toks, before, after):
    """Tokens strictly between the first match of `before` and the next `after`."""
    n = len(toks)
    for i in range(n - len(before) + 1):
        if toks[i:i + len(before)] == before:
            start = i + len(before)
            for j in range(start, n - len(after) + 1):
                if toks[j:j + len(after)] == after:
                    return toks[start:j]
            return None
    return None


def _after(toks, before, count):
    n = len(toks)
    for i in range(n - len(before) + 1):
        if toks[i:i + len(before)] == before:
            return toks[i + len(before):i + len(before) + count]
    return None


def _first(toks, *specs):
    """First spec that matches, so alternative phrasings can be tried in order."""
    for before, after in specs:
        got = _between(toks, before, after)
        if got is not None:
            return got
    return None


SLOTS = {
    "pages": lambda t: _first(t, (["talboken", "har"], ["sidor"]),
                              (["boken", "har"], ["sidor"])),
    # "rubriker på en nivå" (singular) is as common as the plural "nivåer".
    "levels": lambda t: _first(t, (["rubriker", "på"], ["nivåer"]),
                               (["rubriker", "på"], ["nivå"])),
    "year": lambda t: _after(t, ["år"], 1),
    "reader": lambda t: _reader(t),
    "company": lambda t: _company(t),
}


def _reader_anchor(toks):
    """Index just past the 'inläsare är' anchor, tolerating ASR variants.

    Observed: inlässare, inläser, inlasare, and a dropped 'är'.
    """
    for i, w in enumerate(toks):
        if w.startswith("inläs") or w.startswith("inlas") or w.startswith("inläss"):
            if w in ("inläst",):  # "inläst för myndigheten" is the wrong anchor
                continue
            j = i + 1
            if j < len(toks) and toks[j] in ("är", "ar", "e", "i"):
                j += 1
            return j
    return None


def _find(toks, seq, start=0):
    for i in range(start, len(toks) - len(seq) + 1):
        if toks[i:i + len(seq)] == seq:
            return i
    return None


# The studio is introduced by either "vid" or "för" ("... är David Zetterstad
# för Gramma Korrektur"), so both have to end the reader slot.
_COMPANY_PREP = ("vid", "för", "hos", "på")


def _company_prep(toks, start):
    for i in range(start, len(toks)):
        if toks[i] in _COMPANY_PREP:
            return i
    return None


def _reader(toks):
    start = _reader_anchor(toks)
    if start is None:
        return None
    end = _company_prep(toks, start)
    if end is None:
        end = _find(toks, ["denna", "talbok"], start)
    if end is None or end <= start:
        return None
    return toks[start:end]


def _company(toks):
    start = _reader_anchor(toks)
    if start is None:
        return None
    prep = _company_prep(toks, start)
    if prep is None:
        return None
    end = _find(toks, ["denna", "talbok"], prep)
    if end is None:
        end = _find(toks, ["denna", "bok"], prep)
    if end is None or end <= prep + 1:
        return None
    return toks[prep + 1:end]

NUMERIC_SLOTS = ("pages", "levels", "year")


def _as_number(toks):
    """Collapse a slot's tokens to a single int.

    wav2vec sometimes splits a numeral across tokens ("två hundra" -> #2 #100),
    so adjacent numeral tokens are recombined rather than compared piecewise.
    """
    if not toks:
        return None
    vals = []
    for w in toks:
        if w.startswith("#"):
            vals.append(int(w[1:]))
        elif w in ("en", "ett"):  # unambiguously the numeral inside a count slot
            vals.append(1)
        else:
            return None
    if len(vals) == 1:
        return vals[0]
    total = 0
    current = 0
    for v in vals:
        if v in (100, 1000) and current:
            current *= v
        elif v >= 100 and current:
            total += current
            current = v
        else:
            current += v
    return total + current


def extract(toks):
    out = {name: fn(toks) for name, fn in SLOTS.items()}
    for name in NUMERIC_SLOTS:
        if out[name] is not None:
            out[name] = _as_number(out[name])
    return out


def load_all(root="."):
    """Return {id: {'wav2vec': tokens, 'whisperx': tokens}}."""
    out = {}
    wv_dir = os.path.join(root, "storspigg-tbi-wav2vec")
    wx_dir = os.path.join(root, "storspigg-tbi-whisperx")
    for fn in sorted(os.listdir(wx_dir)):
        if not fn.endswith(".json"):
            continue
        key = fn[:-5]
        out[key] = {
            "whisperx": norm_tokens(load_whisperx(os.path.join(wx_dir, fn))),
            "wav2vec": norm_tokens(load_wav2vec(os.path.join(wv_dir, fn))),
        }
    return out
