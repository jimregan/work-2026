# Training Plan

## Volume layout (same as eval-plan)

| Host path | Container path |
|---|---|
| `/home/joregan/merged_tts/workspace` | `/data` |
| `/home/joregan/merged_tts/models` | `/models` |
| `/home/joregan/merged_tts/spoken-sentence-transformers/experiment` | `/workspace/experiment` |

## Configs

Training parameters live in `experiment/configs/`.  Pass `CONFIG_FILE` and `NPROC`; everything else is in the JSON.

All runs hold out speaker `s5` (British accent, excluded from dialect classifier training) as a validation speaker.  At the end of each epoch the trainer reports `semantic_recall@10` on s5 queries vs. a corpus of training-speaker utterances with matching sentence IDs.  Results are appended to `held_out_s5.jsonl` in the model output directory.

> **Note:** verify that `"s5"` matches the `speaker_id` value in the training dataset (may be `"gbi_s5"` or similar) and adjust the JSON before running.

## Batch 1: Ablation runs (configs in `configs/old/`)

These models explore alternatives to the original `wavlm-multiaxis` (sem:384, spk:512, dial:12, gen:2).

### 1. wavlm-multiaxis-v1 — restore original dims (256-128-64-32)

Retrains the hub model architecture with the current pipeline and correct teacher targets.
Axes: `semantic:256 speaker_id:128 dialect:64 gender:32`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/old/wavlm-multiaxis-v1.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 2. wavlm-multiaxis-spk384 — equal dims for semantic and speaker

Speaker and semantic axes both at 384, dialect compressed to 12 and gender to 2.
Tests whether the large speaker dim (512 → 384) reduces speaker dominance in combined retrieval.
Axes: `semantic:384 speaker_id:384 dialect:12 gender:2`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/old/wavlm-multiaxis-spk384.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 3. wavlm-semantic-256 — semantic-only at 256-d

Tests whether a 256-dim semantic projection generalises better cross-corpus than the
384-dim version (which matches teacher output exactly).
Axis: `semantic:256`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/old/wavlm-semantic-256.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 4. wavlm-multiaxis-spkweight — downweight speaker loss

Same dims as run 1 (256-128-64-32), but `speaker_id` axis loss weighted at 0.5.
Forces the backbone to prioritise content over speaker identity during training.

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/old/wavlm-multiaxis-spkweight.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 5. wavlm-multiaxis-grl — gradient reversal on speaker axis

Same dims as run 1 (256-128-64-32), with a gradient reversal layer (λ=1.0)
before the speaker projection head.  The backbone is penalised for encoding
speaker-identifiable information — standard domain-adversarial training.

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/old/wavlm-multiaxis-grl.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

## Batch 2: Clean equal-dim runs (current configs)

These models drop gender, use equal 256-d projections for semantic and speaker,
and test whether adding a dialect axis (via softmax teacher) helps or hurts.
The dialect axis uses `jimregan/merged-tts-dialect-classification` softmax probabilities as the distillation target.

### 6. wavlm-sem256-spk256 — semantic + speaker, equal dims, no dialect

The clean baseline: two axes only, both at 256-d, no softmax classification axes.
Axes: `semantic:256 speaker_id:256`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem256-spk256.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 7. wavlm-sem256-spk256-dial-softmax — + dialect via softmax teacher

Same as run 6 plus a 12-d dialect axis trained from softmax probabilities.
Tests whether dialect supervision helps or hurts semantic/speaker retrieval quality.
Axes: `semantic:256 speaker_id:256 dialect:12`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem256-spk256-dial-softmax.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

## Watching training logs

```bash
docker logs -f <container_id>
```

## Eval after training

Run the full eval suite for all models under `/models` at once:

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e BASE_DIR=/models \
  -e DATA_DIR=/data \
  -e VCTK_INDEX_DIR=/data/vctk-index \
  sst bash /workspace/experiment/bootstrap_full_eval.sh
```

This runs OSR eval, p315 → VCTK eval, rehasp preference-flip eval, and the speaker weight sweep for every model directory under `/models`.

To evaluate a single model only (substitute model name):

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  sst bash -c "
    MODEL_DIR=/models/wavlm-sem256-spk256 \
    BASE_DIR=/models \
    INDEX_DATASET=/data/vctk-index \
    RESULTS_SUFFIX=-vctk \
    bash /workspace/experiment/run_eval_p315.sh && \
    MODEL_DIR=/models/wavlm-sem256-spk256 \
    bash /workspace/experiment/run_rehasp_eval.sh
  "
```
