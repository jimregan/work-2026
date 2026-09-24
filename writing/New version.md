
Yes. To recreate the useful parts of **VoxClamantis + VoxCommunis + LibriSpeech**, I’d outline Paper 1 like this:

## **1. Introduction**

Problem:

Large historical speech archives exist, but official transcripts are not always verbatim enough for direct phonetic analysis.

Claim:

This paper presents a validated chunk-based phonetic corpus of Swedish Riksdag speech, using a Kaldi/MFA-style pipeline with speech-specific language-model biasing and evidence-graded n-gram chunks.

Main question:

Can the resulting corpus recover a known diachronic phonetic change?

## **2. Background and models**

Position the three guides:

|**Source**|**What you borrow**|
|---|---|
|VoxClamantis|explicit caveats, filtering, failure modes|
|VoxCommunis|corpus presentation + phonetic case study|
|LibriSpeech|biased/local LM idea for corpus construction|

Then state the gap:

Existing phonetic corpus papers often assume transcript fidelity or treat alignment error as the main problem. In parliamentary speech, the harder problem is that the official transcript is editorially normalized.

## **3. Source data**

Describe:

- 60 years of Swedish Riksdag speeches
- KB speech-level alignment
- official transcript status
- parliamentary transcription conventions
- why transcript sentence boundaries are not reliable spoken-sentence boundaries

Key principle:

The corpus does not attempt sentence reconstruction. It targets locally validated chunks.

## **4. Textual and acoustic witnesses**

Describe the four textual/orthographic sources:

1. official transcript
2. Whisper
3. wav2vec2 + LM
4. wav2vec2 no LM

Explain their roles:

- official transcript = documentary authority
- ASR systems = spoken-content witnesses
- no-LM system = useful because less normalized by language-model expectations

Then mention, only if included as feasibility:

- phonetic wav2vec2 recognizer as an independent phonetic witness

## **5. Speech-specific biased language model**

This is the LibriSpeech bridge.

For each speech:

```text
official transcript
+ Whisper
+ wav2vec2+LM
+ wav2vec2-no-LM
→ concatenated text
→ speech-specific biased LM
```

Purpose:

The biased LM increases the probability of sequences supported by multiple witnesses while still allowing alternatives where the witnesses diverge.

This is elegant because agreement is not just counted after the fact; it shapes the decoding search space.

## **6. Kaldi/MFA-style decoding and alignment**

This is the Vox bridge.

Pipeline:

```text
speech audio
+ biased LM
+ pronunciation lexicon
→ Kaldi-style decoding
→ candidate word/phone sequence
→ MFA forced alignment
→ phone-level intervals
```

Emphasize:

The canonical corpus boundaries come from the HMM/MFA alignment, not from wav2vec2 timing or DTW.

This keeps the paper methodologically conservative.

## **7. Chunk extraction**

Define output unit:

A chunk is a locally coherent aligned span containing one or more n-grams with sufficient evidence for phonetic analysis.

Then define the tiers:

|**Tier**|**Definition**|
|---|---|
|Gold|unique n-gram sequence present in official transcript, modulo fillers/hesitations/normalizations|
|Silver|n-gram sequence supported by one or more ASR witnesses but not securely anchored in the official transcript|
|Bronze|other candidate n-grams/chunks retained for diagnostic or exploratory use|

Important clarification:

The tiers grade evidential support, not phonetic quality. A gold chunk can still be excluded by acoustic-quality filters.

## **8. Validation and filtering**

This is where your contribution departs from VoxCommunis.

Include:

- transcript–speech mismatch types
- filler and false-start handling
- conjunctions absent from transcript
- repeated/omitted material
- local confidence criteria
- minimum duration / maximum duration
- alignment-quality thresholds
- formant-quality filters

Main point:

Filtering is not cleanup; it is the mechanism by which the corpus becomes scientifically usable.

## **9. Corpus description**

Report:

- number of speeches
- time span
- number of chunks
- gold/silver/bronze counts
- tokens by target vowel
- tokens by decade or period
- speaker/party/gender metadata if available
- attrition table from raw speeches to validated chunks

This is your VoxCommunis-style resource section.

## **10. Case study: known Swedish vowel change**

This is the empirical demonstration.

Question:

Can the validated chunk corpus recover a known diachronic vowel shift?

Method:

- select one dialect/variety if possible
- one vowel phenomenon
- one measurement pipeline
- one periodization
- compare gold-only vs gold+silver if useful

Results:

- formant trajectories
- decade/period plots
- model estimates
- robustness by evidence tier

Interpretation:

The corpus is useful if the known change appears under conservative filtering.

## **11. Small feasibility study, only if worth including**

Not “future work.”

Frame it as:

Concurrently with this corpus construction, we developed a phonetic wav2vec2 recognizer. We include a small feasibility analysis to test whether its output provides an independent signal of chunk validity.

Keep it tiny:

- one sample
- one question
- one result

Example question:

Do chunks supported by both word-level and phonetic recognition show higher manual-validity rates than chunks supported by word-level recognition alone?

Then stop.

Do not make it part of the main pipeline.

## **12. Discussion**

Emphasize:

- why chunks, not sentences
- why transcript fidelity matters
- how this differs from VoxCommunis
- how it inherits caution from VoxClamantis
- why biased decoding is appropriate for parliamentary speech
- what the corpus can and cannot support

## **13. Conclusion**

Final claim:

A locally validated, evidence-graded chunk corpus can support diachronic phonetic analysis of noisy historical parliamentary speech without requiring sentence reconstruction or assuming transcript fidelity.

That is the paper.