# AGENTS.md

## Project: Multi-Axis Speech-to-Speech Similarity Benchmark

---

## 1. Project Overview

This project investigates **multidimensional similarity in speech-to-speech retrieval**.

Unlike speech-to-text retrieval systems (e.g., CLIP-style or S2R-style alignment), this project treats speech as a **first-class retrieval modality**, where similarity is inherently multi-axis rather than purely semantic.

The core contribution is:

* A **vector-valued similarity formulation** for speech embeddings.
* A controlled evaluation framework across multiple speech corpora.
* A minimal extension to the SentenceTransformers framework that returns similarity vectors instead of a single scalar.

This is intended for publication at **Interspeech**.

---

## 2. Core Hypothesis

Speech similarity is not unidimensional.

Two speech segments may be similar along distinct axes:

* Content (same text)
* Speaker identity
* Accent
* Production / fluency / articulation stability

Current embedding approaches collapse these into a single scalar similarity score.

This project formalizes similarity as:

[
\mathbf{s}(x_i, x_j) =
[
s_content,
s_speaker,
s_accent,
s_production
]
]

Each component is computed independently and exposed explicitly.

---

## 3. Retrieval Regime

This project focuses on:

> **Speech-to-Speech retrieval**

NOT:

* Speech-to-text retrieval
* ASR cascade removal
* Cross-modal embedding alignment

Both query and index are acoustic.

Text may be used only for supervision/label definition, not for final retrieval.

---

## 4. Datasets

Datasets serve different structural roles.

### 4.1 Controlled TTS Corpora (Axis Validation)

Used to validate clean axis separability.

* CMU ARCTIC
* VCTK
* Google Britain & Ireland TTS

These provide:

* Same text across speakers
* Known speaker IDs
* Known accent labels (VCTK / GB&I)
* Clean recordings

Use these to:

* Establish content vs speaker separation
* Establish accent axis
* Demonstrate scalar similarity trade-offs

These datasets provide controlled experimental credibility.

---

### 4.2 LibriVox (Naturalistic Variability)

Used as a stress test and realism layer.

Characteristics:

* Volunteer-driven recordings
* Variable fluency
* Limited preparation time
* Unfamiliar proper nouns
* Foreign word difficulty
* Repeated front matter across speakers
* Repeated text across speakers

LibriVox enables:

* Production/fluency axis
* Delivery variability analysis
* Real-world embedding behavior

LibriVox is not the sole experimental base. It complements controlled TTS data.

---

## 5. Encoders

Use off-the-shelf models only.

No new encoder architectures.

Recommended minimal set:

* 1 SSL encoder (HuBERT or wav2vec2)
* 1 ASR encoder (Whisper encoder states)
* Optional: 1 codec-LM encoder (if trivial to extract)

Pooling:

* Mean pooling (default)
* Optional attention pooling (if trivial)

Avoid model zoo explosion.

---

## 6. Technical Contribution

Minimum acceptable contribution:

* New SentenceTransformer-compatible class:

  * `encode()` returns multiple projection vectors.
  * `similarity()` returns a vector instead of scalar.

Preferred stronger version:

* Small projection heads per axis (linear or 2-layer MLP).
* Joint training on axis-specific supervision.

Do NOT:

* Train full encoders from scratch.
* Perform heavy fine-tuning.
* Attempt SOTA performance.

---

## 7. Axis Definitions

Axes must be defined using metadata, not model behavior.

### Content

Positive pairs:

* Same sentence text across speakers.

### Speaker

Positive pairs:

* Same speaker across different sentences.

### Accent (TTS corpora only)

Positive pairs:

* Same accent label.

### Production / Fluency (LibriVox)

Use acoustic proxies only:

* Pause density
* Duration variance
* Disfluency rate (if extractable)
* Stability across repeated takes

Avoid manual annotation.

---

## 8. Evaluation Protocol

### 8.1 Axis-Specific Retrieval

For each axis:

* Compute Recall@K
* Compute MRR

### 8.2 Mixed Retrieval Task

Define a composite task:

* Retrieve items similar by content OR speaker.

Demonstrate:

* Scalar similarity fails or trades off axes.
* Vector similarity + simple weighting succeeds.

This is essential to justify the vector interface.

---

## 9. Comparison to Prior Work

Position clearly:

This work is NOT:

* A speech-to-text retrieval system.
* A replacement for ASR cascades.
* A universal sound benchmark.
* A disentanglement paper.

This work IS:

* A formulation of multidimensional speech similarity.
* A speech-to-speech retrieval study.
* An evaluation of embedding geometry across axes.
* A demonstration that acoustic properties are retrieval-relevant.

---

## 10. Scope Constraints

To avoid project explosion:

* Maximum 3 encoders.
* Maximum 3 datasets (ARCTIC, VCTK/GB&I, LibriVox subset).
* No large-scale hyperparameter search.
* No full benchmark suite like MSEB.
* No exhaustive accent modeling.

Focus on clarity over scale.

---

## 11. Deliverables

Minimum for publication:

1. Vector similarity interface.
2. Controlled TTS experiments showing axis separability.
3. LibriVox stress-test results.
4. Mixed retrieval experiment demonstrating need for vector similarity.
5. Clear speech-to-speech framing.

Optional enhancement:

* Light multi-head projection training.

---

## 12. Non-Goals

This project does NOT aim to:

* Beat CLIP/CLAP-style cross-modal models.
* Replace MSEB.
* Achieve state-of-the-art semantic retrieval.
* Solve accent recognition.
* Build a production system.

---

## 13. Philosophical Position

Speech similarity is inherently multidimensional.

In speech-to-speech retrieval:

* Acoustic properties are not nuisance factors.
* They are retrieval-relevant dimensions.

This project formalizes and evaluates that claim.

