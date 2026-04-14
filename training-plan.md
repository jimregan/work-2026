# Training Plan

## Goal

Replace the old multi-run Docker ablation plan with two ASR training runs on one Slurm node with 8 GPUs:

1. English Common Voice
2. Swedish Common Voice plus extra Swedish data from a second dataset

Both runs should use Hugging Face datasets as the training input.

## Scope

This plan covers:

- dataset preparation
- split hygiene and test-set filtering
- conversion to Hugging Face datasets
- Slurm launch on 1 machine with 8 GPUs

It does not assume that the two prepared datasets are already on the Hugging Face Hub. If they remain local, the downstream training code must support local Hugging Face dataset loading, for example via `load_from_disk(...)` or direct Parquet/JSONL ingestion. If not, publish them to the Hub and train from there.

## Runs

| Run | Training data | Validation | Test |
|---|---|---|---|
| `cv-en` | English Common Voice `train` + cleaned `validated` remainder if needed | held-out English validation split | ASR-targeted English test set |
| `cv-sv-plus` | Swedish Common Voice plus second Swedish dataset | held-out Swedish validation split | ASR-targeted Swedish test set |

## Working assumptions

- Training is distributed across 8 GPUs on a single Slurm node.
- Common Voice provides an ASR-oriented test split that should be treated as the seed for the final test set.
- The validated split must be scrubbed so no utterance with the same sentence or speaker ID leaks into test.
- Single-word sentences are removed for both languages.
- For the second Swedish dataset, remove cases where the transcript is only one word spelled out.

If any of those assumptions are wrong for a specific release, fix the filtering script first and keep the run structure unchanged.

## Directory Layout

Use a layout like this on the cluster filesystem:

| Purpose | Path |
|---|---|
| raw datasets | `/scratch/$USER/asr/raw` |
| cleaned manifests | `/scratch/$USER/asr/manifests` |
| Hugging Face datasets | `/scratch/$USER/asr/hf` |
| model outputs | `/scratch/$USER/asr/models` |
| logs | `/scratch/$USER/asr/logs` |
| data-prep repo checkout | `/scratch/$USER/asr/work-2026` |
| training repo checkout | `/scratch/$USER/asr/spoken-sentence-transformers` |

Recommended dataset output directories:

- `/scratch/$USER/asr/hf/cv-en`
- `/scratch/$USER/asr/hf/cv-sv-plus`

## Dataset Preparation Rules

### 1. Common Voice English

Source inputs:

- Common Voice English `train`
- Common Voice English `validated`
- Common Voice English ASR-targeted `test`

Filtering:

1. Start from the ASR-targeted `test` split as the basis of the final test set.
2. Build two exclusion sets from that test split:
   - normalized sentence text
   - speaker/client ID
3. Remove from `validated` any row whose normalized sentence matches test.
4. Remove from `validated` any row whose speaker/client ID matches test.
5. Remove any row whose transcript is a single word.
6. Keep the resulting cleaned `validated` rows available for validation and, if needed, for train expansion after a second holdout.

Normalization for sentence matching should be deterministic and conservative:

- Unicode normalize
- lowercase
- trim whitespace
- collapse internal whitespace
- strip outer punctuation only if this is already standard in the pipeline

Do not use aggressive text normalization that could merge clearly distinct prompts.

### 2. Common Voice Swedish

Source inputs:

- Common Voice Swedish `train`
- Common Voice Swedish `validated`
- Common Voice Swedish ASR-targeted `test`

Filtering:

1. Use the ASR-targeted `test` split as the final test seed.
2. Build exclusion sets from test:
   - normalized sentence text
   - speaker/client ID
3. Remove from `validated` any row with matching sentence text.
4. Remove from `validated` any row with matching speaker/client ID.
5. Remove any row whose transcript is a single word.

### 3. Extra Swedish Dataset

Source inputs:

- the second Swedish dataset, referred to below as `SWEDISH_EXTRA`

Filtering:

1. Remove any transcript that is a single word.
2. Remove any transcript that is only a spelled-out single word.
3. If speaker IDs exist, preserve them as metadata.
4. If the dataset has no explicit speaker ID, create a stable surrogate speaker field if possible from filename/path/source metadata.
5. If there is any overlap risk with the Common Voice Swedish test material, exclude rows with matching normalized sentence text before merge.

The "single word spelled out" rule should catch transcripts of the form:

- one lexical item rendered as spelled letters
- orthographic variants separated by spaces or punctuation

Examples to exclude:

- `s o s`
- `S-O-S`
- `bvc`

Examples to keep:

- `det är sos`
- `jag sa bvc igår`

The exact regex or heuristic should be documented beside the filtering script.

## Split Construction

### English

Target output:

- `train`
- `validation`
- `test`

Construction:

1. `test` = Common Voice ASR-targeted English test
2. `validation` = held-out sample from cleaned English `validated`
3. `train` = Common Voice English train plus remaining cleaned English `validated` if needed

If `validated` is large enough, prefer:

- `validation` from cleaned `validated`
- keep original `train` as train

Only fold cleaned `validated` into `train` if data volume is otherwise too small.

### Swedish

Target output:

- `train`
- `validation`
- `test`

Construction:

1. `test` = Common Voice ASR-targeted Swedish test
2. `validation` = held-out sample from cleaned Swedish Common Voice `validated`
3. `train` = Common Voice Swedish train plus filtered `SWEDISH_EXTRA`

Keep the Swedish validation set Common Voice-only unless there is a strong reason to validate on the merged distribution.

## Hugging Face Dataset Schema

Prepare both datasets with a common schema:

| Column | Type | Notes |
|---|---|---|
| `audio` | `Audio()` | path or decoded audio |
| `text` | `string` | training transcript |
| `speaker_id` | `string` | original or surrogate |
| `sentence_id` | `string` | if available |
| `source` | `string` | `cv-en`, `cv-sv`, `swedish-extra` |
| `split_origin` | `string` | original source split |
| `language` | `string` | `en` or `sv` |

Recommended split packaging:

- create a `DatasetDict`
- store one directory per run
- keep the three final splits only: `train`, `validation`, `test`

Example output names:

- `cv-en`
- `cv-sv-plus`

## Packaging Options

Choose one of these and keep the training command consistent with it.

### Option A: Save locally

Use:

```python
dataset_dict.save_to_disk("/scratch/$USER/asr/hf/cv-en")
dataset_dict.save_to_disk("/scratch/$USER/asr/hf/cv-sv-plus")
```

The downstream trainer must then accept a local dataset path and load it correctly. This is the cleanest option for cluster-local work.

### Option B: Export loadable data files

Write each split to Parquet or JSONL and load with a local dataset builder or `load_dataset(...)`.

This avoids depending on `save_to_disk(...)`, but needs a stable file convention and a matching training command.

### Option C: Push both datasets to the Hugging Face Hub

Use private repos if needed, then train with:

```bash
--dataset_name your-org/cv-en
--dataset_name your-org/cv-sv-plus
```

This is the simplest option if the external training code already expects Hub datasets.

## Training Launch

Dataset preparation happens in this repo. Training happens from `~/Playing/spoken-sentence-transformers/`, or from its cluster checkout at `/scratch/$USER/asr/spoken-sentence-transformers`, one run at a time on a single node with 8 GPUs.

### Environment

- 1 node
- 8 GPUs
- 8 tasks
- 1 process per GPU

### Suggested Slurm Script Shape

Create a launcher script in `spoken-sentence-transformers` and parameterize the dataset/run name via environment variables.

```bash
#!/bin/bash
#SBATCH --job-name=asr-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/asr/logs/%x-%j.out

set -euo pipefail

cd /scratch/$USER/asr/spoken-sentence-transformers

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  path/to/train.py \
  --output_dir /scratch/$USER/asr/models/$RUN_NAME \
  --dataset_name_or_path $DATASET_SPEC \
  --train_split train \
  --eval_split validation \
  --audio_column audio \
  --text_column text \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --save_steps 500 \
  --eval_steps 500 \
  --gradient_checkpointing
```

Replace `--dataset_name_or_path` with whatever the real trainer expects. The important part is that it resolves to one of the two prepared Hugging Face datasets.

## Concrete Run Names

### Run 1: English Common Voice

Suggested values:

- `RUN_NAME=cv-en-asr`
- `DATASET_SPEC=<english dataset name or local dataset path>`

Launch:

```bash
sbatch --export=ALL,RUN_NAME=cv-en-asr,DATASET_SPEC=<english_dataset> /scratch/$USER/asr/spoken-sentence-transformers/slurm/train_launcher.sh
```

### Run 2: Swedish Common Voice + extra Swedish data

Suggested values:

- `RUN_NAME=cv-sv-plus-asr`
- `DATASET_SPEC=<swedish dataset name or local dataset path>`

Launch:

```bash
sbatch --export=ALL,RUN_NAME=cv-sv-plus-asr,DATASET_SPEC=<swedish_dataset> /scratch/$USER/asr/spoken-sentence-transformers/slurm/train_launcher.sh
```

## Validation Checklist Before Launch

For each run, confirm:

1. no test speaker appears in validation
2. no test speaker appears in train
3. no test sentence appears in validation
4. no test sentence appears in train
5. no single-word transcripts remain
6. for `SWEDISH_EXTRA`, no spelled-out single-word transcripts remain
7. `audio` decodes correctly in the Hugging Face dataset
8. `text` is non-empty
9. train and validation contain enough hours to justify the run

## Recommended Implementation Order

1. Write one preparation script for Common Voice split cleaning.
2. Write one preparation script for the second Swedish dataset filter.
3. Build `cv-en` as a final `DatasetDict`.
4. Build `cv-sv-plus` as a final `DatasetDict`.
5. Decide whether training will use Hub datasets or local Hugging Face dataset loading.
6. Update the external trainer if it cannot read the chosen dataset packaging.
7. Add the Slurm launcher script in `spoken-sentence-transformers`.
8. Run English first, then Swedish merged.

## Short Version

- English run: cleaned Common Voice English only
- Swedish run: cleaned Common Voice Swedish plus filtered second Swedish dataset
- test set basis: ASR-targeted Common Voice test
- leakage prevention: remove matching sentence text and speaker IDs from `validated`
- transcript cleanup: remove single-word items everywhere, plus spelled-out single-word items in the second Swedish dataset
- training target: Hugging Face dataset per run
- compute target: 1 Slurm node, 8 GPUs
