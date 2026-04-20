
Absolutely — here’s the **clean, current LRE-first master outline**, updated to include the **Lidingö /iː/ sociophonetic pilot** and your **living-resource architecture**.

This is the version I’d recommend putting directly into **Obsidian as numbered section notes**.

---

# Master source outline (Obsidian → LRE)

This is the **source tree** from which the LRE paper is assembled.

## 1. Scientific object

### Core claim

A continuously growing phonetic infrastructure over **60 years of Swedish parliamentary speech**, supporting:

- phone-aligned corpus layers
    
- acoustically validated pronunciation variants
    
- phonological rule testing
    
- longitudinal sociophonetics
    
- automatic expansion after each parliamentary sitting
    

### Enduring artifact

The enduring contribution is:

> a **living phonetic corpus and pronunciation lexicon ecosystem**

This is the root note.

---

## 2. Why transcripts are not enough

This is the main motivation note.

### Core argument

Both:

- modern ASR
    
- official parliamentary transcripts
    

optimize for **lexical content**, but suppress:

- reductions
    
- hesitations
    
- fillers
    
- false starts
    
- breathing and vocal events
    
- socially meaningful pronunciation variants
    
- temporal fine structure
    

### Key sentence

Speech science requires **how it was said**, not only **what was said**.

This is highly reusable.

---

## 3. Parliamentary speech as a living substrate

This is the corpus-resource note.

### Key strengths

- 60 years of recordings
    
- stable institutional recording conditions
    
- speaker identity and metadata
    
- constituency / geographic linkage
    
- repeated speakers across years
    
- formal register + spontaneous floor reactions
    
- every new sitting extends the corpus
    

This note feeds the corpus description and sustainability sections.

---

## 4. Core mechanism: rules as acoustic hypotheses

This is the central technical bridge.

### Pipeline logic

1. canonical citation lexicon
    
2. phonological rule expansion
    
3. phonemic / CTC recognition lattice
    
4. acoustic validation of rule-generated variants
    
5. validated forms enter the living lexicon
    
6. rejected variants inform rule refinement
    

### Core scientific sentence

Phonological rules become **acoustically testable lexical hypotheses**.

This is the highest-value shared section.

---

## 5. Swedish phonemic supervision lineage

This is the methodological background note.

### Include

- Waxholm corpus
    
- Swedish phoneme-level supervision
    
- Swedish TIMIT analogue
    
- fillers and speech-noise labels
    
- phonetically rich sentence coverage
    
- legacy phonetic corpus tradition
    

This gives strong historical grounding.

---

## 6. Released resource layers

This is the main LRE contribution note.

## 6.1 Phone-aligned corpus layer

- phones
    
- words
    
- fillers
    
- hesitations
    
- false starts
    
- timestamps
    
- speaker IDs
    

## 6.2 Living pronunciation lexicon

- lemma
    
- canonical form
    
- validated variants
    
- phonological-rule provenance
    
- token frequency
    
- year-by-year change
    
- speaker metadata
    

## 6.3 Rule inventory

- productive alternations
    
- confidence
    
- validation evidence
    
- temporal productivity
    

This section should make the resource feel _multi-layered and reusable_.

---

## 7. Use cases

This section demonstrates why the resource matters.

---

## 7.1 Pronunciation lexicon growth across decades

### Demonstration

Track:

- reduced forms
    
- citation vs attested forms
    
- rule productivity changes
    
- speech-rate conditioned variants
    
- lexical diffusion
    

This validates the living-lexicon story.

---

## 7.2 Longitudinal sociophonetic pilot: emergence of the Lidingö /iː/

This is now the flagship linguistic demo.

### Scientific goal

Recover the historically known rise of the **Lidingö /iː/** in Stockholm speech.

### Data selection

- Stockholm speakers
    
- nearby control constituencies
    
- repeated speakers
    
- known decades of diffusion
    
- stressed /iː/ tokens
    

### Acoustic measures

- F1
    
- F2
    
- duration
    
- trajectory if possible
    
- speaker normalization
    

### Validation claim

The corpus recovers a **known Swedish sound change in progress**.

This is a very strong LRE use case.

---

## 8. Continuous expansion pipeline

This is the sustainability and maintenance note.

### Loop

new sitting  
→ ingest  
→ phonemic decode  
→ rule expansion  
→ acoustic validation  
→ lexicon update  
→ yearly release

This is one of your clearest novelty claims.

Most corpora are static.  
Yours is explicitly **living infrastructure**.

---

## 9. Future backend migration

This is the extensibility note.

### Include

- wav2vec2 → Gemma 4 CTC
    
- improved short-segment precision
    
- better vowel boundary stability
    
- backend model agnosticism
    
- portability of validation loop
    

This strengthens the sustainability argument.

---

# Direct LRE assembly outline

When ready to assemble into manuscript form:

## 1 Introduction

Use notes:

- 1
    
- 2
    
- 3
    

## 2 Related work

Use:

- 5
    
- pronunciation lexicon refs
    
- sociophonetics refs
    

## 3 Corpus and lifecycle

Use:

- 3
    
- 8
    

## 4 Annotation and validation pipeline

Use:

- 4
    
- 5
    

## 5 Released resource layers

Use:

- 6
    

## 6 Evaluation

manual lexicon validation + decade robustness

## 7 Use cases

Use:

- 7.1
    
- 7.2
    

## 8 Sustainability and future releases

Use:

- 8
    
- 9
    

---

# My strongest recommendation

The **first three notes to expand in prose** should still be:

1. **Scientific object**
    
2. **Why transcripts are not enough**
    
3. **Core mechanism: rules as acoustic hypotheses**
    

Those three define the whole paper.

Then jump straight to:

> **7.2 Lidingö /iː/**

because that use case will keep the whole paper scientifically grounded instead of drifting into pure infrastructure.