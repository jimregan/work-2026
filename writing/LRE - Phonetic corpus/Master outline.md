
## 1. Scientific object

This is the root note.

### Core claim

A continuously growing phonetic infrastructure over Swedish parliamentary speech that supports:

* acoustic phonetic annotation
* pronunciation lexicon induction
* phonological rule validation
* longitudinal vowel measurement
* future sitting-by-sitting expansion

This note should define:

> what the enduring artifact is

Everything else derives from this.

---

## 2. Why transcripts are not enough

This comes directly from the good ancestor draft.

### Claims

* modern ASR optimizes for lexical correctness
* official parliamentary transcripts optimize for legislative record
* both suppress speech-scientific detail
* fillers, hesitations, reductions, false starts, segmental variation are lost
* speech science needs **how**, not just **what**

This is your **best motivation note**.

Reusable in both papers.

---

## 3. Parliamentary speech as a living substrate

This is the corpus motivation note.

### Key points

* 60 years of recordings
* stable institutional recording setup
* consistent metadata
* repeated speakers
* regional diversity
* formal + reactive speech mixture
* new sittings extend the corpus naturally

This is mainly for LRE.

---

## 4. Core mechanism: rules as acoustic hypotheses

This is the **central bridge note**.

### Main logic

1. citation lexicon gives canonical forms
2. phonological rules generate plausible variants
3. phonemic recognizer / CTC posterior lattice supplies evidence
4. acoustics validate or reject variants
5. surviving variants enter the living lexicon

This note is the **highest-value prose asset**.

It feeds both:

* SPL method
* LRE derived resource

---

## 5. Swedish phonemic supervision lineage

This is the Waxholm/TIMIT note.

### Include

* Waxholm corpus
* phoneme-level supervision
* Swedish TIMIT analogue
* phonetically rich sentences
* spoken noise annotations
* fillers and hesitations

This is mainly:

* SPL background
* methodological credibility

---

## 6. Derived resource: living pronunciation lexicon

This is the strongest LRE-derived resource note.

### Include

* citation forms
* acoustically attested variants
* rule-supported variants
* confidence scores
* timestamped evidence
* decade-level variant tracking
* release format

This may later become its own section or even separate paper.

---

## 7. Longitudinal vowel measurement

This is the corpus-phonetics bridge.

### Include

* automatic vowel extraction
* formants
* duration
* F0
* speech-rate conditioned reduction
* diachronic movement
* institutional sound change
* repeated political figures

This is very strong for LRE and future phonetics papers.

---

## 8. Continuous expansion pipeline

This is the “living corpus” infrastructure note.

### Loop

new sitting
→ ingest
→ phonemic decode
→ rule expansion
→ acoustic validation
→ lexicon update
→ yearly release

This is your **LRE sustainability section**.

---

## 9. Future backend migration

This is your technical future-proofing note.

### Include

* wav2vec2 → Gemma 4 CTC migration
* short-segment temporal precision
* deletion and lenition sensitivity
* encoder-agnostic validation loop

This feeds:

* SPL future work
* later TASLP

---

# Fork 1: LRE paper outline

This pulls from the master notes.

## 1. Introduction

Use notes:

* 1
* 2
* 3

## 2. Related work

Use:

* 5
* 7
* classic corpus phonetics refs

## 3. Corpus lifecycle and annotation

Use:

* 3
* 8

## 4. Phonetic validation infrastructure

Use:

* 4
* short version of 5

## 5. Derived pronunciation lexicon

Use:

* 6

## 6. Longitudinal phonetic use case

Use:

* 7

## 7. Sustainability and future releases

Use:

* 8
* 9

---

# Fork 2: SPL paper outline

This must stay very tight.

## 1. Introduction

Use:

* 2
* short version of 4

## 2. Method

Use:

* 4
* 5

## 3. Pilot experiment

citation only
vs

* rules
  vs
* acoustic validation

## 4. Discussion

living lexicon implications

## 5. Conclusion

rule-expanded hypotheses improve acoustic validation

---

# My strongest recommendation

The **first three notes to write in full prose** should be:

1. **Scientific object**
2. **Why transcripts are not enough**
3. **Core mechanism: rules as acoustic hypotheses**

Those three notes together already define the entire paper family.

Once they exist, the rest becomes controlled expansion rather than discovery.

