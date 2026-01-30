# KB-LABB text matching reimplementation

## Goal

Reimplement the text matching pipeline from
[kb-labb/riksdagen_anforanden](https://github.com/kb-labb/riksdagen_anforanden)
using off-the-shelf libraries, removing the dependency on their custom `src/`
package. The pipeline aligns ASR transcriptions against official parliamentary
transcripts to find the contiguous matching region in each.

## What the original code does

There are two matching strategies, applied in both directions (ASR→official and
official→ASR):

### 1. N-gram contiguous matching (`contiguous_ngram_match`)

- For n-gram sizes 1 to `n` (default 6):
  - Build n-grams of both texts.
  - For each n-gram in the reference text, check if it appears in the
    hypothesis text, producing a boolean match vector.
  - Convolve the match vector with a kernel of size `n+2` weighted by
    `sqrt(log(n+1))` (longer n-grams get higher weight).
- Stack and sum the scores across n-gram sizes, divide by 3.
- Find contiguous regions where the score exceeds a threshold (default 1.3).
- Filter regions shorter than `min_continuous_match` (default 8 words).
- Join regions separated by gaps smaller than `max_gap` (default 30 words).
- Return the start and end word indices of the matched region.

### 2. Fuzzy contiguous matching (`contiguous_fuzzy_match`)

- Uses `rapidfuzz.fuzz.partial_ratio_alignment` to find the best fuzzy
  substring match of text A within text B.
- Converts character-level indices to word-level indices using NLTK's
  `TreebankWordTokenizer.span_tokenize`.
- Returns start index, end index, and the fuzzy match score.
- Threshold default: 55/100.

### Text normalisation (`normalize_text`)

- Remove punctuation
- Lowercase
- Unicode NFKC normalisation
- Replace hyphens between words with spaces
- Collapse multiple spaces
- Remove spaces between digits, then convert digit sequences to Swedish words
  via `num2words`

## What to implement

Write a Python module (or notebook) that provides the same functionality using
standard/off-the-shelf libraries. The key tasks are:

### Task 1: Text normalisation

Replace `src.data.normalize_text` with a function that uses NeMo text
processing for the number/abbreviation verbalisation step.

Libraries to use:
- Python stdlib (`string`, `re`, `unicodedata`) for punctuation removal,
  lowercasing, whitespace normalisation
- **NeMo text processing** (`nemo_text_processing`) — specifically the
  **non-deterministic** Swedish text normalisation (TN) grammars. These
  produce a WFST lattice of possible verbalisations (e.g., "23" could be
  "tjugotre" or "tjuge tre"), intended for intersection with a language
  model to select the best variant. This is preferable to a single
  deterministic expansion because the matching step can benefit from
  multiple candidate forms.
  - Repo: https://github.com/NVIDIA/NeMo-text-processing
  - The non-deterministic TN grammars live under
    `nemo_text_processing/text_normalization/sv/`
  - Use `nemo_text_processing.text_normalization.normalize.Normalizer`
    with `deterministic=False`
- This replaces the original's use of `num2words` with a more complete
  system that handles ordinals, currency, dates, etc., not just cardinals.

The remaining normalisation steps (punctuation removal, lowercasing, NFKC,
hyphen replacement, whitespace collapsing) stay as simple string ops.

### Task 2: N-gram contiguous region matching

Replace `src.metrics.contiguous_ngram_match` and helpers.

Libraries to use:
- `nltk.ngrams` (or just write a simple n-gram generator — it's trivial)
- `numpy` for the boolean matching, convolution, and contiguous region
  detection

The original implementation is already mostly numpy. The main work is
extracting it from their project structure into a self-contained function.
The core logic is:

1. `get_ngrams_array` — use `nltk.ngrams` or a list comprehension
2. `get_ngram_index_match` — boolean array of where an n-gram appears
3. `get_weighted_ngram_score` — stack weighted convolved match vectors
4. `contiguous_regions` — find contiguous True regions in a boolean array
5. `contiguous_ngram_match` — apply threshold, filter by min length, join
   across gaps

### Task 3: Fuzzy contiguous matching

Replace `src.metrics.contiguous_fuzzy_match`.

Libraries to use:
- `rapidfuzz` (`fuzz.partial_ratio_alignment`) — this is already off-the-shelf
- `nltk` (`TreebankWordTokenizer`) for character-to-word index conversion

This is already nearly standalone. Just extract the function.

### Task 4: Parallel processing wrapper

The original uses `multiprocessing.Pool.imap` with `*_star` wrapper functions.
Keep this pattern or simplify with `Pool.starmap`.

### Task 5: End-to-end pipeline

Provide a function or script that:
1. Takes two text columns (reference and hypothesis)
2. Normalises both
3. Runs n-gram matching in both directions
4. Runs fuzzy matching in both directions
5. Returns a DataFrame with the match indices and scores

## Notes

- The original code has no licence. This reimplementation should be a clean
  rewrite, not a copy.
- The n-gram matching logic is the most complex part. The weighted scoring
  with convolution is the key insight — it gives fuzzy-ish matching using
  exact n-gram lookups at multiple scales.
- The `contiguous_regions` helper (from StackOverflow) is a standard pattern
  for finding runs of True values in a boolean array. Numpy doesn't have a
  built-in for this but the pattern is well-known.
- Consider whether `rapidfuzz.fuzz.partial_ratio_alignment` alone (Task 3)
  is sufficient for the use case, or whether the n-gram approach (Task 2)
  adds meaningful value. The original runs both, so presumably they
  complement each other.
- The Swedish-specific parts are only in normalisation. Everything else is
  language-agnostic.
- The non-deterministic NeMo TN grammars produce multiple candidate
  verbalisations as a lattice. For matching purposes, this is useful: if the
  ASR output says "tjugotre" but a deterministic normaliser would produce
  "tjuge tre", the match would fail. With multiple candidates, at least one
  is likely to match.
- When using `deterministic=False`, the normaliser returns the top-n
  candidates. For matching, generate all candidates and check which gives
  the best alignment score, or concatenate them into an FST for lattice-
  level matching.
- The NeMo text processing grammars require `pynini` (which depends on
  OpenFst). Install via `conda install -c conda-forge pynini` or
  `pip install pynini`.
