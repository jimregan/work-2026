# AGENTS.md — Minimal re-implementation (Transformers-first)

Goal: reproduce the *core* ideas of “Wav2vec behind the Scenes” with the smallest, cleanest codebase, built on
🤗 Transformers + PyTorch. Prefer clarity over architecture.

## What we’re reproducing (minimum)
1) Fine-tune `facebook/wav2vec2-base` with **CTC** for phone recognition (or load an existing fine-tuned checkpoint).
2) During inference, extract hidden activations at frames where the model emits **non-blank**.
3) Run PCA analyses that demonstrate:
   - vowel triangle emergence (transformer layer)
   - short vs long centralization (means)
   - CNN vs Transformer: gender suppression + language separation (if metadata exists)

If the dataset/metadata isn’t available, still implement the pipeline and run the analyses you can; document gaps.

---

## Non-negotiables (to stay faithful)
- Use **Transformers** `Wav2Vec2ForCTC` (or subclass it lightly).
- **No language model** in decoding.
- Extraction frames are based on the model’s own best-path emissions:
  - compute per-frame argmax token ids
  - keep indices where token != blank
  - (optional but recommended) also require token != previous non-blank token to mimic “state change”
- Extract two representations for each kept frame:
  - **CNN feature encoder output** (last conv block)
  - **Transformer last hidden state**

Store per-utterance:
- `phones` (string list)
- `t_idx` (frame indices)
- `cnn` (N x 512)
- `tfm` (N x 768)
- plus metadata (language, gender) if available

---

## Repo should be tiny
Suggested layout:

- `scripts/`
  - `finetune_ctc.py`
  - `extract_activations.py`
  - `pca_plots.py`
- `src/`
  - `data.py`        (dataset adapter, minimal)
  - `extract.py`     (core extraction logic)
  - `analysis.py`    (PCA helpers)
  - `io.py`          (npz/parquet save/load)
- `artifacts/`
  - `ckpt/`
  - `activations/`
  - `figures/`

No frameworks unless they earn their keep (no Hydra, no Lightning). Use argparse.

---

## Dataset stance (pragmatic)
- If **Common Phone** is available, use it.
- If not: implement a dataset adapter interface and support at least one fallback.
- Required dataset fields per item:
  - `audio` (float32 PCM, 16k preferred)
  - `phones` (list of IPA strings or space-separated string)
  - optional: `language`, `gender`

Keep label vocab in `labels.json`:
- include `blank` at id 0 (recommended) or use HF default conventions consistently.

---

## Implementation details (Transformers way)

### Fine-tuning
Use:
- `Wav2Vec2Processor` (feature extractor + tokenizer)
- `Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")`
- Replace tokenizer vocab with phone inventory (or build a `Wav2Vec2CTCTokenizer` from phones)
- CTC loss is handled by `Wav2Vec2ForCTC` when `labels` provided.

Decoding for PER (optional):
- Greedy collapse repeats + remove blanks.

### Activation extraction (core)
In `extract_activations.py`:

Run forward with:
- `output_hidden_states=True`
- and capture CNN features via either:
  - model internals: `model.wav2vec2.feature_extractor` / `feature_projection`
  - or forward hooks on the last conv layer output
Prefer the simplest reliable method; document it.

Then:
- `logits = outputs.logits` (T x V)
- `best = logits.argmax(-1)` (T)
- blank_id = `model.config.pad_token_id` or tokenizer’s blank id (be explicit!)
- `keep = (best != blank_id)`
- recommended: `keep &= (best != best.shift(1))` to enforce “state change”

For each kept `t`:
- phone = id→token
- cnn_vec = cnn_feats[t]
- tfm_vec = outputs.hidden_states[-1][t]

Save as `.npz` per utterance (fast + simple):
- arrays + a small JSON sidecar for strings/metadata, OR store phones as object array if you must.

---

## PCA analyses (keep them lean)

Implement in `pca_plots.py`:

### EXP1: vowel triangle
- Fit PCA on transformer vectors for `[a:], [i:], [u:]` (balanced counts).
- Project `[e:], [o:]` and `[p]`.
- Plot scatter.

### EXP2: short vs long
- Using PCA from EXP1, plot mean points for long vs short variants.

### EXP4: CNN vs Transformer comparisons
- If gender available:
  - Fit PCA separately on CNN vectors and Transformer vectors for `[a:], [i:], [u:]` balanced by gender.
  - Plot mean points (male/female).
- If language available:
  - For phone `[a]`, fit PCA separately on CNN and Transformer balanced by language.
  - Plot per-language clusters/means.

If metadata missing, skip the relevant plots but keep code paths.

---

## “Don’t overbuild” rules
- No pipeline abstraction unless it removes code.
- No fancy experiment runner. Just scripts with sensible defaults.
- Avoid premature generalization; implement for the paper’s needs.
- Add only two tests if any:
  1) blank/removal + repeat-collapse logic
  2) alignment: lengths of phones/cnn/tfm match after filtering

---

## Success criteria
You’re done when:
- one command fine-tunes or loads a checkpoint,
- one command extracts activations to disk,
- one command generates the PCA plots,
- and the plots show the qualitative effects:
  - vowel triangle structure in transformer space
  - CNN shows more speaker-related variation than transformer
  - transformer shows more language/context separation than CNN (if metadata exists)

---

## Example commands (must work)
- Fine-tune:
  - `python scripts/finetune_ctc.py --data <path> --out artifacts/ckpt/w2v2_ctc`
- Extract:
  - `python scripts/extract_activations.py --ckpt artifacts/ckpt/w2v2_ctc --split test --out artifacts/activations`
- PCA plots:
  - `python scripts/pca_plots.py --acts artifacts/activations --out artifacts/figures`

