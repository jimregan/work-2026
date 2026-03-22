# Training Plan

## Volume layout

| Host path | Container path |
|---|---|
| `/home/joregan/merged_tts/workspace` | `/data` |
| `/home/joregan/merged_tts/models` | `/models` |
| `/home/joregan/merged_tts/spoken-sentence-transformers/experiment` | `/workspace/experiment` |

## Setup (fresh machine)

Run inside the sst Docker container:

```bash
docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e HF_DATASET=jimregan/merged-vctk-cmuarctic-gbi \
  -e NPROC=8 \
  sst bash /workspace/experiment/bootstrap_fresh.sh
```

Then on the **host**, build the VCTK index:

```bash
VCTK_DIR=~/merged_tts/vctk/wav48_silence_trimmed \
UTTERANCE_MAP=/home/joregan/merged_tts/workspace/utterance-map.json \
OUTPUT_DIR=/home/joregan/merged_tts/workspace/vctk-index \
bash experiment/prepare_vctk_index.sh
```

## Targets datasets

| Path | Speaker backend | Axes |
|---|---|---|
| `/data/training-data-targets-resem` | Resemblyzer (256-d) | semantic(384) + speaker_id(256) + dialect(12) |
| `/data/training-data-targets-xvec` | WavLM-SV x-vector (768-d) | semantic(384) + speaker_id(768) |
| `/data/training-data-targets-xvec-pca256` | x-vector → PCA(256-d) | semantic(384) + speaker_id(256) |

## Ablation runs

All models: no gender axis, `semantic:384` (matches teacher exactly, no alignment matrix).

### Resemblyzer speaker targets — no alignment matrix on either axis

Run all four in parallel (2 GPUs each):

#### 1. wavlm-sem384 — semantic only (baseline)

```bash
docker run -d --gpus '"device=0,1"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384.json \
  -e NPROC=2 \
  sst bash /workspace/experiment/run_train.sh
```

#### 2. wavlm-sem384-spk256-resem — + speaker (resemblyzer)

```bash
docker run -d --gpus '"device=2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384-spk256-resem.json \
  -e NPROC=2 \
  sst bash /workspace/experiment/run_train.sh
```

#### 3. wavlm-sem384-spk256-resem-dial — + speaker + dialect

```bash
docker run -d --gpus '"device=4,5"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384-spk256-resem-dial.json \
  -e NPROC=2 \
  sst bash /workspace/experiment/run_train.sh
```

#### 4. wavlm-sem384-spk256-resem-grl — + speaker + gradient reversal

```bash
docker run -d --gpus '"device=6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384-spk256-resem-grl.json \
  -e NPROC=2 \
  sst bash /workspace/experiment/run_train.sh
```

### X-vector speaker targets — alignment matrix comparison

Run after the resemblyzer batch (reuse GPUs).

#### 5. wavlm-sem384-spk768-xvec — speaker:768, no alignment matrix

```bash
docker run -d --gpus '"device=0,1,2,3"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384-spk768-xvec.json \
  -e NPROC=4 \
  sst bash /workspace/experiment/run_train.sh
```

#### 7. wavlm-sem384-spk256-xvec-pca — speaker:256, PCA-projected x-vector targets (no alignment matrix)

```bash
docker run -d --gpus '"device=4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384-spk256-xvec-pca.json \
  -e NPROC=4 \
  sst bash /workspace/experiment/run_train.sh
```

#### 6. wavlm-sem384-spk256-xvec — speaker:256, with alignment matrix (256→768)

```bash
docker run -d --gpus '"device=4,5,6,7"' --ipc=host \
  -v /home/joregan/merged_tts/workspace:/data \
  -v /home/joregan/merged_tts/models:/models \
  -v /home/joregan/merged_tts/spoken-sentence-transformers/experiment:/workspace/experiment \
  -e CONFIG_FILE=/workspace/experiment/configs/wavlm-sem384-spk256-xvec.json \
  -e NPROC=4 \
  sst bash /workspace/experiment/run_train.sh
```

## Watching training

```bash
docker logs -f <container_id>
```

Check loss progression across all models:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('/models/*/checkpoint-*/trainer_state.json')):
    d = json.load(open(f))
    log = d.get('log_history', [])
    first = next((e for e in log if 'loss' in e), {})
    last = next((e for e in reversed(log) if 'loss' in e), {})
    name = f.split('/')[-3]
    print(f'{name}: {first.get(\"loss\",\"?\"):.4f} -> {last.get(\"loss\",\"?\"):.4f}  (epoch {last.get(\"epoch\",\"?\"):.1f})')
"
```

## Eval after training

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
