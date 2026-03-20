# Evaluation Plan

## 1. Basic sanity test

- **Index**: training data (merged VCTK / CMU Arctic / GBI dataset)
- **Query**: untranscribed held-out VCTK speaker
- **Status**: VCTK transcription corrections now complete
- **Goal**: verify the model correctly clusters utterances of the same sentence when the speaker is unseen

## 2. Rehasp preference-flip (three cases)

Run via `run_rehasp_eval.sh` / `run_all_rehasp_evals.sh`.

### Case 1: rehasp query, OSR-only index

- **Index**: OSR dataset (no rehasp)
- **Query**: rehasp reps (rep002–005), axis weights `semantic=1.0,speaker_id=0.0`
- **Goal**: baseline — show rehasp queries retrieve the correct OSR sentence with no same-speaker distractor

### Case 2: rehasp query, OSR+rehasp mixed index (default weights)

- **Index**: OSR + rehasp rep001, axis weights `semantic=1.0,speaker_id=1.0`
- **Goal**: show the preference-flip setup — with equal weights, same-speaker rep001 distractors compete against semantic matches

### Case 3: speaker ID axis negated (sweep)

- **Index**: OSR + rehasp rep001
- **Axis weights**: `semantic=1.0,speaker_id=W` sweeping over negative values (default `W=-1.0`)
- **Goal**: demonstrate that negating the speaker axis recovers the correct same-sentence different-speaker match from OSR
- Set `SPEAKER_SWEEP="-2.0 -1.5 -1.0 -0.5 0.0"` for a full sweep

## 4. Librivox poetry test (maybe)

- **Index**: one poetry reading from LibriVox, segmented with VAD or breath detection
- **Query**: another reading of the same poems
- **Goal**: make the case that SST can be used to create its own training data by aligning
  multiple readings of the same text

## Setup scripts

- `experiment/bootstrap_rehasp.sh` — downloads rehasp, builds OSR-only and OSR+rehasp mixed index
  datasets, prepares query audio and labels for eval 2
- `experiment/run_rehasp_eval.sh` — runs all three rehasp cases for one model
- `experiment/run_all_rehasp_evals.sh` — runs rehasp eval for all models under `$BASE_DIR`
- `experiment/run_preference_flip.sh` — runs a single preference-flip eval (lower-level)
- `experiment/run_eval_osr.sh` — runs basic OSR retrieval eval for one model
- `experiment/run_all_evals.sh` — runs OSR retrieval eval for all models under `$BASE_DIR`
- `experiment/run_eval_p315.sh` — runs p315 retrieval eval for one model
- `experiment/run_all_p315_evals.sh` — runs p315 eval for all models, skips `wavlm-resemblyzer`
- `experiment/make_p315_labels.py` — generates `/data/p315-labels.json` from matches TSV
- `experiment/match_transcripts.py` — matches p315 transcriptions against VCTK reference corpus
- `experiment/baseline_retrieval.py` — text sentence-transformers and CLAP baselines
- `experiment/compare_evals.py` — prints a summary table across models

## Host volume layout (GPU server)

| Host path | Container path |
|---|---|
| `/home/joregan/merged_tts/workspace` | `/data` |
| `/home/joregan/merged_tts/models` | `/models` |
| `/home/joregan/merged_tts/results` | `/results` |

## Data paths (inside container)

| Purpose | Path |
|---|---|
| Training data | `/data/training-data` |
| Training targets | `/data/training-data-targets` |
| p315 audio | `/data/p315` |
| p315 query labels | `/data/p315-labels.json` |
| Models | `/models/wavlm-multiaxis`, `/models/wavlm-semantic` |
| OSR audio | `/data/osr-audio` |
| OSR segments | `/data/osr-segments` |
| OSR HF dataset | `/data/osr-dataset` |
| Utterance map | `/data/utterance-map.json` |
| OSR + rehasp index | `/data/osr-rehasp-mixed` |
| Rehasp query audio | `/data/rehasp-query-segments` |
| Rehasp query labels | `/data/rehasp-labels.json` |

## Docker commands

### Training (example — semantic, GPUs 4-7)

```bash
docker run -d --gpus '"device=4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/results:/results \
  -e DATASET_DIR=/data/training-data \
  -e TARGETS_DIR=/data/training-data-targets \
  -e OUTPUT_DIR=/models/wavlm-semantic \
  -e NPROC=4 \
  sst bash /workspace/experiment/run_train_semantic.sh
```

### p315 CLAP + text baselines (CPU is fine, mount live experiment scripts)

```bash
docker run --rm --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  sst \
  bash -c "
    python /workspace/experiment/baseline_retrieval.py \
      --corrected   /workspace/experiment/p315.tsv \
      --reference   /workspace/experiment/vctk-all.tsv \
      --labels      /data/p315-labels.json \
      --utterance_map /data/utterance-map.json \
      --output_json /data/p315-text-baseline.json && \
    python /workspace/experiment/baseline_retrieval.py \
      --corrected   /workspace/experiment/p315.tsv \
      --reference   /workspace/experiment/vctk-all.tsv \
      --labels      /data/p315-labels.json \
      --utterance_map /data/utterance-map.json \
      --audio_dir   /data/p315 \
      --clap        laion \
      --output_json /data/p315-clap-laion-baseline.json && \
    pip install msclap && \
    python /workspace/experiment/baseline_retrieval.py \
      --corrected   /workspace/experiment/p315.tsv \
      --reference   /workspace/experiment/vctk-all.tsv \
      --labels      /data/p315-labels.json \
      --utterance_map /data/utterance-map.json \
      --audio_dir   /data/p315 \
      --clap        ms \
      --output_json /data/p315-clap-ms-baseline.json
  "
```

### OSR eval (GPUs 0-3, mount live experiment scripts)

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  sst \
  bash -c "
    BASE_DIR=/models \
    INDEX_DATASET=/data/osr-dataset \
    QUERY_DIR=/data/osr-segments \
    bash /workspace/experiment/run_all_evals.sh
  "
```

### p315 eval (GPUs 0-3, mount live experiment scripts)

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/results:/results \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  sst \
  bash -c "
    python /workspace/experiment/make_p315_labels.py \
      --matches       /workspace/experiment/p315-matches.tsv \
      --audio_dir     /data/p315 \
      --utterance_map /data/utterance-map.json \
      --output        /data/p315-labels.json && \
    BASE_DIR=/models \
    INDEX_DATASET=/data/training-data \
    P315_AUDIO_DIR=/data/p315 \
    P315_LABELS=/data/p315-labels.json \
    bash /workspace/experiment/run_all_p315_evals.sh
  "
```

### Rehasp bootstrap (needs curl, no GPU required)

```bash
docker run --rm \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/spoken-sentence-transformers:/workspace \
  sst \
  bash -c "apt update && apt install -y curl unzip && bash /workspace/experiment/bootstrap_rehasp.sh"
```

### Rehasp eval (GPUs 0-3)

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  sst \
  bash -c "
    BASE_DIR=/models \
    OSR_DATASET=/data/osr-dataset \
    OSR_REHASP_DATASET=/data/osr-rehasp-mixed \
    QUERY_DIR=/data/rehasp-query-segments \
    QUERY_LABELS=/data/rehasp-labels.json \
    bash /workspace/experiment/run_all_rehasp_evals.sh
  "
```
