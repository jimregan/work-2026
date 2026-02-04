# Multilingual G2P

## Goal

Train a multilingual G2P system using data extracted from the Braxen Swedish TTS lexicon.
Original plan: ByT5-based model. Fallback baseline: Phonetisaurus. Intermediate: DeepPhonemizer (CTC-based).

## Kaggle notebooks

- [Check Braxen multilingual entries](https://www.kaggle.com/code/jimregan/notebookb68ecf0f4e)
- [Second version](https://www.kaggle.com/code/jimregan/notebook13614fe245)
- [Split Braxen by language](https://www.kaggle.com/code/jimregan/split-braxen-by-language)
- [Download and fix Braxen](https://www.kaggle.com/code/jimregan/download-and-fix-braxen)

## Local notebooks

- `2025-10-15-phone-freqs.ipynb`: Phone frequency analysis; reverse lookup of rare phones to source words
- `2025-10-20-filter-braxen.ipynb`: Initial TSV parsing from Braxen, numeric filtering
- `2025-10-23-check-braxen-multilingual-entries.ipynb`: Hunspell validation across 26+ languages
- `2025-10-24-per-calcs.ipynb`: PER/WER evaluation for 7 languages, 3 Phonetisaurus strategies (WL, MWL, WLM)
- `2025-10-25-filter-braxen-hunspell.ipynb`: Nordic character normalization + filtered output
- `2025-10-25-retsvify-phonetisaurus.ipynb`: Convert Phonetisaurus alignment output to word/phone pairs

## TODO

### Accent markers
- Accent/stress markers were stripped (ASR habit); need to be restored for TTS use.
- **Blocking**: all current PER results in the paper are based on accent-stripped data.
  Must restore accents first, then re-run the entire pipeline (filtering, training, evaluation)
  before any results are publishable.
- Keep the accent-stripped results as a comparison: the numbers are strong (Swedish <1% PER)
  and the with/without accent comparison is itself informative (shows the cost of including
  stress marking for TTS). Doubles the number of experiments but worth it.

### English data
- Current English data from Braxen is small (~19,700 tokens); reviewers will reject without more.
- Plan: use existing English lexica and get language-tagged data from Wikipedia.

### Filter by graphone
- English words not marked as English, etc.
- Need to identify and fix language misattributions in the lexicon.
- Some notebooks already identified misclassified items, but nothing was done with the results.
- Correct approach: update a copy of the original Braxen lexicon by ID field with corrected language tags, then re-run the split/filter pipeline.

### Paper
- Location: `/Users/joregan/Playing/lrec-g2p/LREC2026 Author's kit/lrec2026-example.tex`
- Old ByT5/pivot draft preserved as `junk.tex` (includes discussion comments with JE).
- Target: LREC 2026
- Current state: rewritten to focus on Phonetisaurus/WFST multilingual G2P.
  - Intro: framing around loanwords in TTS, SSML language tags. Done.
  - Background: end-to-end TTS limitations, phoneset mapping. Done.
  - Data/Braxen: MTM context, language tag errors (lat/lav/lit), Nordic spelling normalization. Done.
  - Method: character filtering (too noisy) → Hunspell filtering (limitations noted), 3 Phonetisaurus configs (WL, MWL, WLM, RAW). Done.
  - Results: PER table present. Needs prose.
  - Discussion: empty.
  - Conclusion: empty.
- Still TODO for paper: results prose, discussion, conclusion; add DeepPhonemizer results; add English data results.
- The WFST-based multilingual G2P approach is novel — no prior work does exactly this.

### Ideas from junk.tex worth keeping

1. **Pivot/bridge language analogy from MT**: The connection to pivot languages in
   SMT/RBMT (Forcada, Kumar et al.), and Google NMT's implicit interlingua via shared
   modules + language token. Good framing for the ByT5 version — the language prefix
   in Phonetisaurus is essentially the same mechanism as the prepended token in Google NMT.

2. **T5 prompt design**: `g2p|lang=sv` extended to `g2p|lang=sv|from=en`, plus explicit
   pivot task `ipa2ipa|from=X|to=Y`. Open research question: are both needed?

3. **Compound splitting (JE, supervisor)**: JE's view is that compound splitting should be
   preprocessing, not baked into the G2P task — "baking segmentation into the G2P task is
   an at least relatively bad idea. Similar to baking language detection into it."
   **Disagree**: the point of multitask learning in ByT5 is that auxiliary NLP tasks
   (compound splitting, POS tagging, etc.) share representations with the primary G2P
   task and reinforce each other. JE's objection likely stems from unfamiliarity with
   multitask learning rather than a fundamental issue with the approach.

4. **"Zero-shot G2P" critique**: Producing arbitrary IPA for unknown languages is useless
   for TTS because output must be bounded to the speaker's phoneme inventory. The non-goal
   statement is well-articulated and could go in a related work section.

5. **NST lexicon compound information**: Compounds in NST are split with linkers; the
   linking 's' matters for syllabification (Östermalmstorg → Östermalms + torg, not
   Östermalm + *storg). Limitation: Braxen lacks compound markers, so nested compounds
   (Östermalm = öster + malm) can't be recovered.

6. **The core use case**: "a mechanism that can be plugged into the text processing layer
   of a TTS system that can be specifically told 'X is an English word, pronounce it as
   a Swede would'."

Items 3, 5, and 6 are directly relevant to the current paper. Items 1, 2, and 4 are
relevant if/when extending to DeepPhonemizer/ByT5.

### Nordic spelling normalization bug
- **`2025-10-23-check-braxen-multilingual-entries.ipynb`** (cell `6deb0d1f`):
  Does `w.replace("ö","ø").replace("Ö","Ø")` before spell-checking for `{"nor","dan"}`,
  but uses the wrong code (`"nor"` instead of `"nob"`), so it never fires for Norwegian
  Bokmål. Also incomplete — doesn't do `ä→æ` or `ae→æ`.
- **`2025-10-25-filter-braxen-hunspell.ipynb`** (cell `487ef0c1` / `d096145a`):
  `check_nobdan()` does full normalization (ae→æ, ä→æ, ö→ø, etc.) and writes the
  **normalized native spelling** to the output, replacing the original Swedish-convention
  spelling. This is the problem: the word can no longer be traced back to the Braxen entry,
  and the G2P training data now has the native spelling where Braxen had the Swedish spelling,
  so the model learns a different grapheme-to-phoneme mapping than what Braxen intended.
- **Fix**: normalization should only be used for Hunspell lookup, never written to output.
  The output should keep the original Braxen spelling. If native-spelling variants are
  needed, they should be a separate column/annotation, not a replacement.

### Pipeline unification
The current pipeline is fragmented across 6 notebooks + Kaggle, with intermediate files
landing in `/tmp`, hardcoded paths, and no clear ordering. This makes it hard to re-run
and leads to bugs like the normalization issue above.

Suggested unified pipeline (single script or notebook with clear stages):

1. **Load**: Read `braxen-sv.tsv`, skip comments, parse ID/word/transcript/language fields.
   Keep accent markers. Work from a copy if corrections are needed.
2. **Correct**: Apply any language tag corrections by ID (from graphone filtering / manual
   review). This is where misclassified items get fixed.
3. **Split by language**: Produce per-language word/transcript pairs. Filter numerics.
4. **Validate**: Hunspell spell-check per language. Nordic normalization for lookup only,
   never modifying the source word. Record OK/MISS/suggestions as metadata, don't discard
   entries yet.
5. **Filter**: Apply minimum entry thresholds, keep only Hunspell-OK words (+ configurable
   exceptions like known-good proper names). Output filtered per-language files.
6. **Train/test split**: Consistent, reproducible splits.
7. **Train models**: Phonetisaurus (WL, MWL, RAW), DeepPhonemizer, (ByT5).
   Two variants: with and without accent markers.
8. **Evaluate**: PER calculation against held-out test sets. Output results tables.

Benefits:
- Single source of truth (the corrected Braxen copy).
- Re-runnable end to end when data or corrections change.
- Accent marker stripping becomes a flag, not a one-way data loss.
- Nordic normalization is isolated to the validation step.

### Compound language identification
- Determine the language of each part of a compound word.
- Approach: build a reverse phone-to-word mapping, skipping entries with subword joiners;
  collect potential compounds (entries with joiners) for a second pass,
  matching their parts against the reverse map to assign per-part language labels.
