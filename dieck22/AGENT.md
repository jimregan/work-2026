# AGENTS.md — Re-implementation Guide (dieck22_interspeech: “Wav2vec behind the Scenes”)

This repository is a **re-implementation** of the core experiments from:
**tom Dieck et al. (Interspeech 2022)**, *“Wav2vec behind the Scenes: How end2end Models learn Phonetics”*.

Your job as an agent is to produce a clean, reproducible pipeline that:
1) fine-tunes wav2vec 2.0 for multilingual phone recognition (CTC),
2) extracts hidden activations at **CTC non-blank emission frames**,
3) reproduces the paper’s PCA-based analyses and plots.

---

## 0) Primary goals (what “done” means)

### Must-have deliverables
- [ ] A **train/fine-tune script** for wav2vec2-base (CTC) on a multilingual phone dataset (Common Phone if accessible; otherwise a documented substitute).
- [ ] An **activation extraction script** that:
  - runs inference on a held-out test set,
  - obtains decoded phone emissions,
  - extracts hidden vectors at time steps where the model emits **non-blank** symbols.
  - extracts from **two layers**:
    - last CNN feature encoder layer (≈512-d)
    - last Transformer layer output (≈768-d)
  - stores aligned sequences: `phones[]`, `cnn_vecs[]`, `tfm_vecs[]` (same length per utterance).
- [ ] A **PCA analysis module** that reproduces:
  1) vowel triangle recovery (PCA fit on [a:], [i:], [u:], then project [e:], [o:], and a plosive [p])
  2) short vs long vowel centralization (mean points for long/short variants)
  3) consonant place/manner organization (plosive/fricative mean points + “correspondence” arrows)
  4) CNN vs Transformer comparison:
     - gender effect on [a:], [i:], [u:]
     - language effect on [a]
- [ ] Figures saved to `artifacts/figures/` and a short **reproduction report** in `REPRODUCE.md`.

### Nice-to-have
- [ ] Unit tests for alignment logic (CTC emission indexing).
- [ ] Hydra/argparse config + seed control.
- [ ] Minimal “dataset adapter” layer so others can swap corpora.

---

## 1) Constraints & rules

### Reproducibility
- Deterministic runs where possible:
  - fixed random seeds
  - log exact versions (PyTorch, Transformers, CUDA)
- Never hide preprocessing steps.
- Every artifact should be regenerable from raw inputs + config.

### Faithfulness to the paper (core methodological commitments)
- Use **wav2vec2-base** (or document any deviation).
- Use **CTC** decoding without a language model (beam search is OK; no LM scoring).
- Activation extraction must be **tied to non-blank emissions**:
  - If the output frame is blank, skip.
  - If non-blank, record vectors from both layers at that time step.
- PCA should be:
  - fit on specified training sets (e.g., triangle vowels),
  - then applied as a transform to other phones.

### If Common Phone is not available
- Implement the pipeline anyway with a substitute dataset (e.g., multilingual phone-labeled speech).
- Document the mismatch and what changes in the results you expect.
- Keep interfaces identical so Common Phone can be plugged in later.

---

## 2) Repository structure (expected)

- `configs/`
  - training configs (model, lr schedule, batch sizes, etc.)
  - analysis configs (phones to include, balancing rules)
- `src/`
  - `data/` dataset loading + IPA label mapping
  - `model/` wav2vec2 CTC wrapper
  - `decode/` CTC decoding + beam search
  - `extract/` activation extraction + storage
  - `analysis/` PCA + plotting code
  - `utils/` seeding, logging, I/O
- `scripts/`
  - `train_ctc.py`
  - `infer_and_extract.py`
  - `run_pca_experiments.py`
- `artifacts/`
  - `models/`
  - `activations/`
  - `figures/`
  - `logs/`
- `REPRODUCE.md` (how to run everything end-to-end)
- `RESULTS.md` (what you got vs what you expected)

---

## 3) Data & label expectations

### Audio
- 16kHz, mono WAV preferred.
- If resampling is required, do it explicitly and log it.

### Labels
- Phone sequences as IPA symbols.
- Provide:
  - a vocabulary mapping `phone -> id`, plus `blank` token.
  - optional mappings for “long vowel” notation (e.g., `a:`) and language/gender metadata.

### Metadata
- Each utterance should expose:
  - `language` (one of 6 in the paper, if possible)
  - `gender` (male/female) OR a documented proxy/unknown

---

## 4) Model/training spec (paper-inspired default)

Implement a sane approximation of the paper’s schedule; exact matching is not required if you clearly document differences.

### Default model
- `facebook/wav2vec2-base` (pretrained)
- Add a linear layer to target phone inventory + blank.
- Train with CTC loss.

### Decoding (for extraction)
- Beam search width default: 10
- No language model
- Output: best phone sequence + per-frame predicted token ids.

### Logging
- PER (phone error rate) on dev/test if ground truth exists.
- Save checkpoints and config hashes.

---

## 5) Critical logic: extracting at non-blank emissions

This is the *most important* part and is easy to get subtly wrong.

### Required extraction behavior
For each utterance:
1) Run forward pass and obtain per-frame logits over `V` tokens (phones + blank).
2) Decode to get predicted token id per frame (argmax or beam-aligned best path).
3) Identify time steps `t` where predicted token != blank **and** (optionally) differs from previous emitted non-blank state.
   - Paper motivation: CTC blanks represent “stay in state”; emissions represent “state change”.
4) For each such `t`, record:
   - emitted phone symbol
   - CNN last-layer vector at `t`
   - Transformer last-layer vector at `t`

Store as a compact format (recommended):
- `parquet` or `npz` per utterance, plus an index file.

### Validation checks
- Vectors and phone list lengths match exactly.
- Time indices strictly increasing.
- Distribution sanity: vowels should show broader dispersion than many consonants in PCA.

---

## 6) PCA experiments (exact targets)

All experiments must be runnable with one command and produce saved figures.

### EXP1: vowel triangle recovery
- Fit PCA on elongated vowels: `[a:], [i:], [u:]` (balanced counts).
- Project:
  - those three vowels
  - `[e:], [o:]`
  - plosive `[p]` (should cluster near origin)
- Use Transformer vectors for this experiment.

**Output**: scatter plot similar to paper Fig. 1.

### EXP2: short vs long vowel centralization
- Using the PCA from EXP1, plot mean points for:
  - long vowels vs their short counterparts
- Expect short vowels closer to triangle center.

**Output**: mean-point plot similar to paper Fig. 2.

### EXP3: plosive vs fricative place/manner structure
- Collect plosives + fricatives (exclude elongated/palatalized variants).
- Fit PCA on balanced samples across groups.
- Plot mean point per phone, draw arrows from plosives to corresponding fricatives.

**Output**: mean-point plot similar to paper Fig. 3.

### EXP4A: CNN vs Transformer — gender effect (triangle vowels)
- Phones: `[a:], [i:], [u:]`
- Balance samples by phone and gender.
- Fit PCA separately for CNN vectors and Transformer vectors.
- Plot mean points per phone per gender.

Expected:
- CNN: strong gender separation
- Transformer: gender mostly disappears

**Output**: side-by-side plot similar to paper Fig. 4.

### EXP4B: CNN vs Transformer — language effect ([a])
- Phone: `[a]` (short open front vowel)
- Balance by language (6 languages)
- Fit PCA separately for CNN and Transformer vectors.
- Plot per-language clusters/means.

Expected:
- CNN: languages overlap near origin
- Transformer: language separation appears (context/phonotactics)

**Output**: side-by-side plot similar to paper Fig. 5.

---

## 7) Reporting requirements

### REPRODUCE.md must include
- Exact commands to:
  - prepare data
  - train/fine-tune
  - run inference + extraction
  - run PCA experiments
- Expected runtime ranges (rough) and hardware notes.
- Where outputs appear on disk.

### RESULTS.md must include
- PER (if measurable) and decoding settings.
- A short qualitative comparison against expected plots:
  - Did you recover the triangle?
  - Did gender disappear after transformer?
  - Did language separation appear in transformer?

---

## 8) Common failure modes (avoid these)

- Mixing frame indices between CNN features and transformer outputs (they must align).
- Extracting on *ground truth* phone boundaries (not allowed; must be model-emission-driven).
- Using t-SNE/UMAP instead of PCA (not equivalent to the paper’s “transform unseen data” requirement).
- Not balancing samples across vowels/gender/language → biased PCA axes.
- Accidentally using a language model during decoding (disallowed for core reproduction).

---

## 9) “If you must deviate” protocol

If you cannot match the paper exactly (dataset availability, metadata missing, etc.):
1) implement the pipeline anyway,
2) clearly document the deviation in `RESULTS.md`,
3) explain how you expect it to affect each experiment,
4) keep the code structured so the original setup can be swapped in later.

---

## 10) Commands (placeholders — implement)

After implementation, these should work:

- Train:
  - `python scripts/train_ctc.py --config configs/train.yaml`
- Extract:
  - `python scripts/infer_and_extract.py --ckpt artifacts/models/best.pt --split test`
- PCA + plots:
  - `python scripts/run_pca_experiments.py --activations artifacts/activations/test/`

---

## 11) Definition of success

A successful re-implementation produces:
- a functioning end-to-end pipeline,
- activation extraction tied to CTC non-blank emissions,
- PCA plots that qualitatively match the paper’s claims:
  - vowel triangle emerges from transformer vectors,
  - short vowels centralize,
  - articulatory relationships are visible,
  - CNN encodes gender strongly while transformer suppresses it,
  - transformer encodes language-related structure for [a].

If anything fails, prioritize fixing extraction alignment and balancing first.

