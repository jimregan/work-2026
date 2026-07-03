# Linux Smoke Test

This is a first-pass checklist for moving `dysfluent-wfst` to a Linux server and running it on a small Swedish corpus.

## 1. Environment Setup

Create a fresh conda environment:

```bash
conda create -n dysfluent-wfst python=3.10 -y
conda activate dysfluent-wfst
```

Try installing the repo directly first:

```bash
pip install -e .
```

If that fails because of native or ABI issues, install the easier Python dependencies first:

```bash
pip install torch torchaudio transformers datasets pyyaml numpy
```

The two dependencies most likely to need special handling on Linux are:

- `pynini`
- `k2`

Install those using whatever method matches the server's Python/CUDA/toolchain setup.

Praat support is optional. The core decoder does not require it. The
package now has a backend model for acoustic enrichment:

- `basic`: built in, no Praat required
- `praat`: optional Parselmouth-backed enrichment

If you want Praat-based enrichment, install the extra:

```bash
pip install -e '.[praat]'
```

## 2. Smoke Import

Before attempting any decode, make sure the core stack imports:

```bash
python -c "import pynini, k2, torch, torchaudio, transformers, datasets; import dysfluent_wfst.cli; print('imports ok')"
```

If this fails, do not continue to corpus testing until the import problem is resolved.

## 3. Single-Utterance Smoke Test

Run one utterance end to end:

```bash
dysfluent-decode \
  --model-id YOUR_MODEL \
  --lexicon lexicon.tsv \
  --rules rules/swedish_hypo.yaml \
  --audio path/to/one.wav \
  --ref-text "din referenstext här" \
  --output smoke.json
```

Inspect the output:

```bash
python -m json.tool smoke.json | head -n 80
```

Sanity checks:

- `segments` is non-empty
- `frame_shift_ms` is plausible
- `start_time_s` and `end_time_s` increase monotonically
- `decoded_phonemes` is not empty
- `variation_type` values look plausible

## 4. Short-Corpus Manifest

Create a small manifest file, for example `mini.jsonl`:

```json
{"id":"utt1","audio_path":"path/to/utt1.wav","ref_text":"..."}
{"id":"utt2","audio_path":"path/to/utt2.wav","ref_text":"..."}
{"id":"utt3","audio_path":"path/to/utt3.wav","ref_text":"..."}
```

Run batch decoding:

```bash
dysfluent-decode \
  --model-id YOUR_MODEL \
  --lexicon lexicon.tsv \
  --rules rules/swedish_hypo.yaml \
  --audio mini.jsonl \
  --output mini_out.jsonl
```

Quick checks:

```bash
wc -l mini.jsonl mini_out.jsonl
head -n 3 mini_out.jsonl
```

You want output line count to roughly match the number of usable manifest items.

## 5. Recommended Comparison Run

Run the same small subset in two modes.

Strict citation-form mode:

```bash
dysfluent-decode \
  --model-id YOUR_MODEL \
  --lexicon lexicon.tsv \
  --audio mini.jsonl \
  --output strict.jsonl \
  --no-back \
  --no-skip \
  --no-sub
```

Rules plus variation mode:

```bash
dysfluent-decode \
  --model-id YOUR_MODEL \
  --lexicon lexicon.tsv \
  --rules rules/swedish_hypo.yaml \
  --audio mini.jsonl \
  --output free.jsonl
```

Interpretation:

- `strict.jsonl` gives citation-form-only decoding
- `free.jsonl` gives rules plus variation arcs

This is the fastest way to see whether the more permissive graph is finding materially different alignments.

## 6. Failure Modes To Watch

The most likely Linux runtime problems are:

- `pynini` import/build issues
- `k2` install or CUDA/CPU mismatch
- model download/auth issues from Hugging Face
- lexicon phones not matching the model vocabulary
- rule expansion creating too many candidate paths for ambiguous entries

The rule compiler is still the shakiest semantic area in the codebase. If the server run looks strange, test with and without `--rules` before assuming the decoder core is wrong.

## 7. Minimal Batch Helper

Use this shell script as a first-pass test harness.

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: smoke_run.sh MODEL_ID LEXICON MANIFEST [RULES]}"
LEXICON="${2:?usage: smoke_run.sh MODEL_ID LEXICON MANIFEST [RULES]}"
MANIFEST="${3:?usage: smoke_run.sh MODEL_ID LEXICON MANIFEST [RULES]}"
RULES="${4:-}"

echo "[1/5] Import check"
python -c "import pynini, k2, torch, torchaudio, transformers, datasets; import dysfluent_wfst.cli; print('imports ok')"

echo "[2/5] Strict run"
dysfluent-decode \
  --model-id "$MODEL_ID" \
  --lexicon "$LEXICON" \
  --audio "$MANIFEST" \
  --output strict.jsonl \
  --no-back \
  --no-skip \
  --no-sub

echo "[3/5] Free run"
if [[ -n "$RULES" ]]; then
  dysfluent-decode \
    --model-id "$MODEL_ID" \
    --lexicon "$LEXICON" \
    --rules "$RULES" \
    --audio "$MANIFEST" \
    --output free.jsonl
else
  dysfluent-decode \
    --model-id "$MODEL_ID" \
    --lexicon "$LEXICON" \
    --audio "$MANIFEST" \
    --output free.jsonl
fi

echo "[4/5] Line counts"
wc -l "$MANIFEST" strict.jsonl free.jsonl

echo "[5/5] Sample output"
head -n 2 strict.jsonl
head -n 2 free.jsonl
```

Save it as `smoke_run.sh`, make it executable, and run:

```bash
chmod +x smoke_run.sh
./smoke_run.sh YOUR_MODEL lexicon.tsv mini.jsonl rules/swedish_hypo.yaml
```

## 8. Suggested Next Step After Smoke Testing

If the short run looks reasonable, the next useful layer is a small evaluator that reports:

- utterance count in manifest
- utterance count successfully decoded
- average segment count per utterance
- frequency of each `variation_type`
- examples of extreme outputs such as empty segments or excessive deletions

That will make it much easier to judge whether the system is stable on the Swedish corpus without manually reading JSONL files.
