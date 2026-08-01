"""Swedish numeral parsing, for comparing ASR outputs that differ in how
numbers are rendered (wav2vec spells them out, whisperx writes digits)."""

UNITS = {
    "noll": 0, "ett": 1, "en": 1, "två": 2, "tre": 3, "fyra": 4,
    "fem": 5, "sex": 6, "sju": 7, "åtta": 8, "nio": 9,
}

TEENS = {
    "tio": 10, "elva": 11, "tolv": 12, "tretton": 13, "fjorton": 14,
    "femton": 15, "sexton": 16, "sjutton": 17, "arton": 18, "aderton": 18,
    "nitton": 19,
}

TENS = {
    "tjugo": 20, "tjugu": 20, "trettio": 30, "fyrtio": 40, "förtio": 40,
    "femtio": 50, "sextio": 60, "sjuttio": 70, "åttio": 80, "nittio": 90,
}

SCALES = {"hundra": 100, "tusen": 1000, "miljon": 10**6, "miljoner": 10**6}

ORDINALS = {
    "förste": 1, "första": 1, "andre": 2, "andra": 2, "tredje": 3,
    "fjärde": 4, "femte": 5, "sjätte": 6, "sjunde": 7, "åttonde": 8,
    "nionde": 9, "tionde": 10, "elfte": 11, "tolfte": 12, "trettonde": 13,
    "fjortonde": 14, "femtonde": 15, "sextonde": 16, "sjuttonde": 17,
    "artonde": 18, "nittonde": 19, "tjugonde": 20, "trettionde": 30,
    "fyrtionde": 40, "femtionde": 50, "sextionde": 60, "sjuttionde": 70,
    "åttionde": 80, "nittionde": 90, "hundrade": 100, "tusende": 1000,
}

_ADDITIVE = {**UNITS, **TEENS, **TENS}
_ALL = {**_ADDITIVE, **SCALES}
_MAXLEN = max(len(k) for k in _ALL)


def _tokenize(s):
    """Split a run-together numeral into parts, longest match with backtracking.

    Backtracking matters because several pairs collide at equal length:
    nittio/nitton, sextio/sexton, sjuttio/sjutton.
    """
    if not s:
        return []
    for n in range(min(_MAXLEN, len(s)), 0, -1):
        head = s[:n]
        if head in _ALL:
            rest = _tokenize(s[n:])
            if rest is not None:
                return [head] + rest
    return None


def _combine(tokens):
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok in SCALES:
            scale = SCALES[tok]
            if current == 0:
                current = 1
            if scale == 100:
                current *= 100
            else:
                total += current * scale
                current = 0
        else:
            current += _ADDITIVE[tok]
        seen = True
    if not seen:
        return None
    return total + current


def parse(word):
    """Parse a Swedish cardinal or ordinal written as one word. None if not a numeral."""
    w = word.lower()
    if w in ORDINALS:
        return ORDINALS[w]
    toks = _tokenize(w)
    if toks is None:
        return None
    return _combine(toks)


# Bare "en"/"ett" are the indefinite articles far more often than the numeral,
# so they stay words. In a compound ("etthundra") the tokenizer still sees them.
AMBIGUOUS_BARE = {"en", "ett"}


def canon(word):
    """Canonical form of a token: '#<int>' for numerals, else the token itself."""
    w = word.lower()
    if w.isdigit():
        return "#" + str(int(w))
    if w in AMBIGUOUS_BARE:
        return w
    v = parse(w)
    if v is None:
        return w
    return "#" + str(v)
