---
name: intergaelic-modernize
description: Modernize archaic or classical Irish (Gaeilge) text into contemporary standard Irish using the Cadhán intergaelic API. Use this skill whenever the user wants to modernize, normalize, or update old Irish text, classical Irish, early Modern Irish, manuscript Irish, or any Irish text described as archaic, old-fashioned, or hard to read. Also trigger when the user pastes Irish text and asks what it means or asks for a readable version — modernization is often the right first step.
---

# Intergaelic Irish Modernizer

Modernizes archaic/classical Irish text into contemporary standard Irish via the
[Cadhán intergaelic API](https://cadhan.com) (ga→ga roundtrip, which normalizes
spelling and grammar to the Caighdeán Oifigiúil).

## When to use

- User has old Irish text (manuscript, pre-standardization, early Modern Irish, etc.)
- User asks to "modernize", "normalize", "update spelling", or "make readable"
- User pastes Irish and says it looks old-fashioned or is hard to parse

## How to use

The script lives at `scripts/modernize_irish.py` relative to this skill. It needs
no dependencies beyond the Python standard library.

### Inline text
```bash
python scripts/modernize_irish.py "Ní bhfuair sé aon ní"
```

### From a file
```bash
python scripts/modernize_irish.py < input.txt
```

### From a variable in a pipeline
```bash
echo "$old_text" | python scripts/modernize_irish.py
```

## API behaviour

- The Cadhán API accepts `foinse=ga` (Irish source) and returns an array of
  `[original_token, modernized_token]` pairs.
- The script reassembles the pairs into a single modernized string.
- On network or parse failure the original text is returned unchanged and a
  warning is printed to stderr.

## Workflow

1. Copy the script from the skill's `scripts/` directory to your working directory
   if you need to invoke it as a subprocess, or import `modernize()` directly.
2. Run on the text.
3. Present both the original and modernized versions to the user, noting any tokens
   that were left unchanged (may indicate proper nouns or already-modern forms).

## Notes

- The API is stateless and processes text as a flat token stream — it does not
  parse document structure, so feed it paragraph by paragraph for best results.
- Very long inputs should be chunked (e.g. 500-word blocks) to stay within
  reasonable HTTP request sizes.
- The API also supports `gd` (Scottish Gaelic) and `gv` (Manx) as source
  languages if those ever come up.
