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

## Runs

### 1. wavlm-multiaxis-v1 — restore original dims (256-128-64-32) [needs rerun with new config]

Retrains the hub model architecture with the current pipeline and correct teacher targets.
Axes: `semantic:256 speaker_id:128 dialect:64 gender:32`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-multiaxis-v1.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 2. wavlm-multiaxis-spk384 — speaker bottleneck at semantic dim

Speaker axis projected to 384 (same as semantic), forcing the fixed orthogonal alignment
matrix to compress 512-dim xvectors.  Hypothesis: less speaker capacity → less speaker
dominance in combined retrieval.
Axes: `semantic:384 speaker_id:384 dialect:12 gender:2`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-multiaxis-spk384.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

### 3. wavlm-semantic-256 — semantic-only with bottleneck projection

Tests whether the 256-dim semantic projection generalises better cross-corpus than the
current 384-dim (which matches teacher output exactly and may overfit to training speaker
acoustics).
Axis: `semantic:256`

```bash
docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-semantic-256.json \
  -e NPROC=8 \
  sst bash /workspace/experiment/run_train.sh
```

## Watching training logs

```bash
docker logs -f <container_id>
```

## Eval after training

Run the full eval for a single new model (substitute model name):

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  sst \
  bash -c "
    MODEL_DIR=/models/wavlm-multiaxis-v1 \
    BASE_DIR=/models \
    INDEX_DATASET=/data/vctk-index \
    RESULTS_SUFFIX=-vctk \
    bash /workspace/experiment/run_eval_p315.sh && \
    MODEL_DIR=/models/wavlm-multiaxis-v1 \
    BASE_DIR=/models \
    OSR_DATASET=/data/osr-dataset \
    OSR_REHASP_DATASET=/data/osr-rehasp-mixed \
    QUERY_DIR=/data/rehasp-query-segments \
    QUERY_LABELS=/data/rehasp-labels.json \
    bash /workspace/experiment/run_rehasp_eval.sh
  "
```
