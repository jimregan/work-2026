# Minimal Training Runbook

This document is for private runs on the training server.  It assumes the
current branch has already been pulled directly over SSH and does not require
pushing the branch to GitHub.

## Build the Docker image

From the repo checkout on the training server:

```bash
cd /sbt-temp/sst2/spoken-sentence-transformers
docker build -t sst .
```

## Start the container

The training config uses `/data` for datasets and `/models` for checkpoints.
On the current server setup, `/sbt-temp/sst2` is the host directory mounted as
`/data` inside Docker.

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /sbt-temp/sst2:/data \
  -v /sbt-temp/sst2/models:/models \
  -v /shared:/shared:ro \
  sst bash
```

## English Common Voice run

The first comparison run mirrors the best non-Resemblyzer paper setup:

- `semantic:384`
- `speaker_id:256`
- semantic teacher: `all-MiniLM-L6-v2`
- speaker teacher: `microsoft/wavlm-base-plus-sv`
- full x-vector targets, no PCA

Config:

```text
/workspace/experiment/configs/cv-en-sem384-spk256-xvec.json
```

### Prepare the dataset

All splits are built from `validated.tsv`.  `test.tsv` and `dev.tsv` are used
only as sentence/speaker seeds.  Single-word prompts and prompts with fewer
than two recordings are filtered from every split.

```bash
mkdir -p /data/raw/common-voice

tar -xf /shared/datasets/Common_Voice_English_25/cv-corpus-25.0-2026-03-09-en.tar.gz \
  -C /data/raw/common-voice

python /workspace/experiment/prepare_cv.py \
  --cv_dir /data/raw/common-voice/cv-corpus-25.0-2026-03-09/en \
  --language en \
  --output_dir /data/cv-en
```

### Precompute targets

Use `path` as the utterance ID for Common Voice because it is unique per
recording.

```bash
python /workspace/experiment/precompute_targets.py \
  --dataset_dir /data/cv-en \
  --output_dir /data/cv-en-targets-xvec \
  --axes semantic speaker_id \
  --semantic_model all-MiniLM-L6-v2 \
  --speaker_id_model microsoft/wavlm-base-plus-sv \
  --speaker_id_backend xvector \
  --utterance_id_column path \
  --batch_size 16 \
  --device cuda
```

### Train

```bash
CONFIG_FILE=/workspace/experiment/configs/cv-en-sem384-spk256-xvec.json \
NPROC=8 \
bash /workspace/experiment/run_train.sh
```

The model output is written to:

```text
/models/cv-en-sem384-spk256-xvec
```

## Swedish placeholders

The Swedish run should use the same general flow, but with:

- prepared `cv-sv-plus` dataset
- semantic teacher: `KBLab/sentence-bert-swedish-cased`
- the same speaker teacher/backend unless there is a reason to change it

Do not add a Swedish dialect axis until the Swedish dialect classifier and its
label set are prepared.
