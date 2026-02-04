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

### English data
- Current English data from Braxen is small (~19,700 tokens); reviewers will reject without more.
- Plan: use existing English lexica and get language-tagged data from Wikipedia.

### Filter by graphone
- English words not marked as English, etc.
- Need to identify and fix language misattributions in the lexicon.
- Some notebooks already identified misclassified items, but nothing was done with the results.
- Correct approach: update a copy of the original Braxen lexicon by ID field with corrected language tags, then re-run the split/filter pipeline.

### Compound language identification
- Determine the language of each part of a compound word.
- Approach: build a reverse phone-to-word mapping, skipping entries with subword joiners;
  collect potential compounds (entries with joiners) for a second pass,
  matching their parts against the reverse map to assign per-part language labels.
