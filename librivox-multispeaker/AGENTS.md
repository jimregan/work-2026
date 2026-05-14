# AGENTS.md

## Project: Multi-Axis Speech-to-Speech Similarity

---

## 1. Project Overview

This project investigates **multidimensional similarity in speech-to-speech retrieval**.

Unlike speech-to-text retrieval systems (e.g., CLIP-style or S2R-style alignment), this project treats speech as a **first-class retrieval modality**, where similarity is inherently multi-axis rather than purely semantic.

The core contribution is:

* A **factor-partitioned speech embedding** whose subspaces correspond to distinct relational axes (content, speaker, accent, …).
* A controlled evaluation framework across multiple speech corpora.
* A minimal, installable extension to the SentenceTransformers framework (`spoken_sentence_transformers`).

This is intended for publication at **Interspeech**.

---

## 2. Core Hypothesis

Speech similarity is not unidimensional.

Two speech segments may be similar along distinct axes — the specific axes depend on what supervision a given dataset provides. The set of axes is not fixed; it is configured at training time and stored in a HuggingFace-compatible `config.json`.

The architecture produces a **concatenated, factor-partitioned embedding**:

```
z = [ z_semantic || z_speaker_id || z_gender || ... ]
```

where each block `z_a = W_a · z_base / ‖W_a · z_base‖` is the L2-normalised projection onto axis `a`'s hypersphere.

Key geometric properties:
* Each block has norm ≈ 1 (unit sphere per axis).
* Full vector norm ≈ √A (A = number of axes).
* Cosine similarity over the full vector = **unweighted mean** of per-axis cosines (independent of dimensionality).
* Weighted retrieval requires computing per-axis cosines explicitly and summing with explicit weights — do NOT rely on dimensionality to weight axes.

**This is supervised subspace separation, not disentanglement.** Nothing prevents axes from sharing information; we make no claims of statistical independence. We claim only that the projection heads are trained to make each subspace useful for its declared relational axis.

**Axes are specified in the `MultiAxisProjection` constructor as a `dict[str, int]` (axis name → output dimension) and serialised to `config.json`. They are not hardcoded anywhere in the implementation. Declaration order is preserved through save/load.**

---

## 3. Retrieval Regime

This project focuses on:

> **Speech-to-Speech retrieval**

NOT:

* Speech-to-text retrieval
* ASR cascade removal
* Cross-modal embedding alignment

Both query and index are acoustic.

Text may be used for supervision/label definition (e.g. pre-cached sentence transformer embeddings as content-axis positives) — not for final retrieval.

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
* Real-world embedding behaviour

LibriVox is not the sole experimental base. It complements controlled TTS data.

---

## 5. Encoders

Use off-the-shelf models only.

No new encoder architectures.

Recommended minimal set:

* 1 SSL encoder (HuBERT or WavLM)
* 1 ASR encoder (Whisper encoder states)
* Optional: 1 codec-LM encoder (if trivial to extract)

Pooling:

* Mean pooling (default)
* Optional attention pooling (if trivial)

Avoid model zoo explosion.

---

## 6. Technical Contribution

Minimum acceptable contribution:

* Installable package `spoken_sentence_transformers`:

  * `encode()` returns a single structured embedding.
  * `encode_axis()` / `encode_all_axes()` expose per-axis vectors.
  * `similarity_vector()` returns per-axis cosine matrices.

Preferred stronger version:

* Small projection heads per axis (linear or 2-layer MLP).
* Joint training on axis-specific supervision.

Do NOT:

* Train full encoders from scratch.
* Perform heavy fine-tuning.
* Attempt SOTA performance.

---

## 6a. Implementation

Install:

```bash
pip install -e ".[dev]"
```

### Package layout

```
spoken_sentence_transformers/
    __init__.py                  ← public API re-exports
    projection.py                ← MultiAxisProjection, MultiAxisProjectionConfig
    sentence_transformer.py      ← MultiAxisSentenceTransformer
    trainer.py                   ← MultiAxisProjectionTrainer
    sampler.py                   ← MultiAxisNoDuplicatesBatchSampler
    loss.py                      ← MultiAxisInfoNCELoss
    encoders/
        __init__.py
        base.py                  ← AcousticEncoder (abstract)
        hf.py                    ← HFAcousticEncoder
tests/
    test_projection.py
docs/
    adding-a-dataset.md
    adding-an-encoder.md
    adding-a-loss.md
```

---

### `projection.py` — `MultiAxisProjection`

A `sentence_transformers.models.Module` subclass.  Sits at the end of the
`SentenceTransformer` pipeline after `Encoder → Pooling`.

* Constructor: `MultiAxisProjection(in_features, axes: dict[str, int], hidden_dim=None, default_axis=None)`
* `axes` keys are preserved in declaration order through save/load (stored as
  an ordered list of `[name, dim]` pairs in `config.json`; a JSON array is not
  subject to `sort_keys`).
* `forward()` reads `features["sentence_embedding"]`, **L2-normalises** each
  head output, writes `features["embedding_{axis}"]` for every axis, and sets
  `features["sentence_embedding"]` to the requested/default/concatenated projection.
* Explicit geometry attributes computed at init:
  * `axis_names: list[str]` — declaration-order axis names
  * `axis_dims: list[int]` — output dimension per axis
  * `axis_slices: dict[str, tuple[int, int]]` — `(start, end)` into the concat vector
* `forward_kwargs = {"axis"}` — callers can pass `axis=` to `encode()`.

---

### `sentence_transformer.py` — `MultiAxisSentenceTransformer`

Subclasses `SentenceTransformer`.  Adds:

* `axes` property — axis names in declaration order.
* `axis_slices` property — delegates to the projection module.
* `encode_axis(sentences, axis)` — embeddings for one axis.
* `encode_all_axes(sentences)` — `dict[axis, np.ndarray]`.
* `similarity_vector(a, b)` — `dict[axis, Tensor]` of per-axis cosine matrices.

Standard `encode()` and all built-in evaluators continue to work unchanged.

---

### `trainer.py` — `MultiAxisProjectionTrainer`

Adapted copy of `SentenceTransformerTrainer`.  Key changes:

* **`collect_features(inputs)`** returns `dict[str, dict[str, Tensor]]` keyed
  by role (`"anchor"`, `"semantic_pos"`, `"speaker_id_pos"`, …) instead of a
  flat list.  Loss functions receive axis-specific views with no cross-axis
  contamination.
* **`DEFAULT_FEATURE_SUFFIXES`** class attribute lists recognised input column
  suffixes.  Extend per-instance via the `feature_suffixes` constructor
  parameter, or globally by subclassing:

  ```python
  trainer = MultiAxisProjectionTrainer(
      ...,
      feature_suffixes=MultiAxisProjectionTrainer.DEFAULT_FEATURE_SUFFIXES
          + ("pitch_values",),
  )
  ```

* Loss is **required** — no default is applied.

Dataset column naming convention:

```
anchor_{suffix}          — anchor example
{axis}_pos_{suffix}      — positive for each axis
{axis}_label             — label for batch sampler duplicate detection
```

where `{suffix}` ∈ `{sentence_embedding, input_features, input_ids, pixel_values}`.

See `docs/adding-a-dataset.md` for a full example including text-encoder-derived
content positives.

---

### `sampler.py` — `MultiAxisNoDuplicatesBatchSampler`

Prevents false negatives at the data level.  For each axis, maintains a set of
labels already present in the batch.  A candidate sample is admitted only if its
label for **every** axis is absent.

Dataset must contain `{axis}_label` columns.

---

### `loss.py` — `MultiAxisInfoNCELoss`

Per-axis InfoNCE loss.

* Anchor is run through the model **once** — all axis projections computed simultaneously.
* Each axis's positives run through the model once.
* Embeddings (`embedding_{axis}`) are already L2-normalised unit vectors — do not normalise again.
* Returns `dict[str, Tensor]` (per-axis scalar losses).  Trainer sums for backprop, logs individually.

```python
loss = MultiAxisInfoNCELoss(model, temperature=0.05,
                            axis_weights={"semantic": 1.0, "speaker_id": 1.0})
```

---

### `encoders/base.py` — `AcousticEncoder`

Abstract base for all acoustic encoders.  Five abstract methods:

* `tokenize(audio_list)` → feature dict
* `forward(features)` → feature dict (must write `token_embeddings` [B,T,H] and `attention_mask` [B,T])
* `get_word_embedding_dimension()` → int
* `save(output_path, ...)` → None
* `load(model_name_or_path, ...)` → Self

See `docs/adding-an-encoder.md` for a full `CodecEncoder` skeleton and a
`PitchAugmentedEncoder` pattern.

### `encoders/hf.py` — `HFAcousticEncoder`

Wraps any HuggingFace audio encoder (WavLM, wav2vec2, HuBERT, Whisper encoder).
Uses `AutoFeatureExtractor` + `AutoModel`.  For seq2seq models, only the encoder
half is used.

---

## 7. Axis Definitions

Axes must be defined using metadata or acoustic proxies, not model behaviour.
**The axis set differs between the TTS subset and the LibriVox subset.**

### 7.1 TTS Subset Axes (CMU ARCTIC / VCTK / Google Britain & Ireland)

**content / semantic**
Positive pairs: same sentence text, different speaker.
Recommended: use a text sentence transformer (e.g. `all-MiniLM-L6-v2`) to
pre-compute transcript embeddings as `semantic_pos_sentence_embedding`.

**speaker**
Positive pairs: same speaker ID, different sentence.

**accent**
Positive pairs: same accent label (from corpus metadata), different speaker.

### 7.2 LibriVox Subset Axes

**content**
Positive pairs: same source text / passage, different speaker.

**speaker**
Positive pairs: same reader ID, different recording.

**fluency** (acoustic proxy)
Pause density, duration variance, disfluency rate.  Avoid manual annotation.

**prosody** (acoustic proxy)
Pitch range, speech rate, energy contour statistics.

> Note: Accent labels for LibriVox readers are available via the LibriVox
> Accents Table but are self-reported and incomplete.

---

## 8. Evaluation Protocol

### 8.1 Axis-Specific Retrieval

For each axis:

* Recall@K
* MRR
* (Speaker axis) Equal Error Rate against classical speaker embedding baselines

### 8.2 Classical Speaker Embedding Comparison

Reviewers will expect comparison against x-vectors / ECAPA-TDNN / WavLM-based
speaker embeddings.

* Extract `model.encode_axis(utterances, axis="speaker_id")`.
* Compare EER against SpeechBrain ECAPA-TDNN on the same test set.
* Framing: we are not claiming to beat dedicated speaker systems; we are showing
  that the speaker axis of a multi-axis embedding matches them *while
  simultaneously encoding other axes*.  That is the actual claim.
* `SpeakerAlignmentLoss` in `docs/adding-a-loss.md` shows how to use a
  classical system as training supervision.

### 8.3 Mixed Retrieval Task

Define a composite task: retrieve items similar by content OR speaker.

Demonstrate:

* Scalar similarity fails or trades off axes.
* Per-axis cosines + explicit weighting succeeds.

This is essential to justify the structured embedding interface.

---

## 9. Comparison to Prior Work

Position clearly:

This work is NOT:

* A speech-to-text retrieval system.
* A disentanglement paper (no claims of latent independence or orthogonality).
* A replacement for ASR cascades.
* A universal sound benchmark.

This work IS:

* A **supervised subspace separation** approach: each axis head is trained
  directly from relational labels; information may overlap across axes.
* A **factor-partitioned embedding** architecture enabling controllable weighted retrieval.
* A speech-to-speech retrieval study.
* A demonstration that acoustic properties are retrieval-relevant dimensions,
  not nuisance factors to be normalised away.

---

## 10. Scope Constraints

* Maximum 3 encoders.
* Maximum 3 datasets (ARCTIC, VCTK/GB&I, LibriVox subset).
* No large-scale hyperparameter search.
* No full benchmark suite like MSEB.
* No exhaustive accent modelling.

Focus on clarity over scale.

---

## 11. Deliverables

Minimum for publication:

1. Installable `spoken_sentence_transformers` package.
2. Controlled TTS experiments showing axis separability.
3. LibriVox stress-test results.
4. Mixed retrieval experiment demonstrating need for structured similarity.
5. Speaker axis comparison against at least one classical baseline.
6. Clear speech-to-speech framing.

---

## 12. Non-Goals

* Beat CLIP/CLAP-style cross-modal models.
* Replace MSEB.
* Achieve SOTA semantic retrieval.
* Solve accent recognition.
* Build a production system.
* Prove statistical disentanglement.

---

## 13. Philosophical Position

Speech similarity is inherently multidimensional.

In speech-to-speech retrieval, acoustic properties — speaker identity, accent,
prosody, fluency — are not nuisance factors to be normalised away.  They are
retrieval-relevant dimensions in their own right.

The structured embedding constructed here makes that claim operationally precise:
each axis occupies its own unit-sphere subspace, comparison is explicit and
controllable, and the geometry is preserved end-to-end from training through
retrieval.
