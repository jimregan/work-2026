# Training Plan

## Goal

Replace the old multi-run Docker ablation plan with two ASR training runs on one Slurm node with 8 GPUs:

1. English Common Voice
2. Swedish Common Voice plus extra Swedish data from a second dataset

Both runs should use local Hugging Face datasets saved with `save_to_disk(...)` as the training input.

## Scope

This plan covers:

- dataset preparation
- split hygiene and test-set filtering
- conversion to Hugging Face datasets
- Slurm launch on 1 machine with 8 GPUs

The training code in `~/Playing/spoken-sentence-transformers/experiment/train_wavlm.py` already loads datasets with `load_from_disk(...)`, so local Hugging Face datasets are the default and preferred path here.

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
- The second Swedish dataset is `KTH/nst`.
- The English run will use the same semantic teacher model as the existing English setup.
- The Swedish semantic teacher model is `KBLab/sentence-bert-swedish-cased`.
- The same speaker ID model/backend choices will be used across both runs.
- A Swedish dialect classifier model will need to be trained for the Swedish run.

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

## Source Archives

Common Voice release archives are available on the NFS mount visible to the training machine:

- Swedish Common Voice 25: `/shared/datasets/Common_Voice_Swedish_25/cv-corpus-25.0-2026-03-09-sv-SE.tar.gz`
- English Common Voice 25: `/shared/datasets/Common_Voice_English_25/cv-corpus-25.0-2026-03-09-en.tar.gz`

Recommended extraction targets on scratch:

- `/scratch/$USER/asr/raw/common-voice/sv-SE/`
- `/scratch/$USER/asr/raw/common-voice/en/`

## Dataset Preparation Rules

### 1. Common Voice English

Source inputs:

- archive: `/shared/datasets/Common_Voice_English_25/cv-corpus-25.0-2026-03-09-en.tar.gz`
- extracted Common Voice English `train`
- extracted Common Voice English `validated`
- extracted Common Voice English ASR-targeted `test`

Known `validated.tsv` columns:

- `client_id`
- `path`
- `sentence_id`
- `sentence`
- `sentence_domain`
- `up_votes`
- `down_votes`
- `age`
- `gender`
- `accents`
- `variant`
- `locale`
- `segment`

Known `test.tsv` columns:

- `client_id`
- `path`
- `sentence_id`
- `sentence`
- `sentence_domain`
- `up_votes`
- `down_votes`
- `age`
- `gender`
- `accents`
- `variant`
- `locale`
- `segment`

Observed metadata notes:

- `variant` is empty in English Common Voice 25
- `locale` is always `en` in English Common Voice 25

Filtering:

1. Start from the ASR-targeted `test` split as the basis of the final test set.
2. Build two exclusion sets from that test split:
   - normalized sentence text from `sentence`
   - speaker ID from `client_id`
3. Remove from `validated` any row whose normalized `sentence` matches test.
4. Remove from `validated` any row whose `client_id` matches test.
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

- archive: `/shared/datasets/Common_Voice_Swedish_25/cv-corpus-25.0-2026-03-09-sv-SE.tar.gz`
- extracted Common Voice Swedish `train`
- extracted Common Voice Swedish `validated`
- extracted Common Voice Swedish ASR-targeted `test`

Known `validated.tsv` columns:

- `client_id`
- `path`
- `sentence_id`
- `sentence`
- `sentence_domain`
- `up_votes`
- `down_votes`
- `age`
- `gender`
- `accents`
- `variant`
- `locale`
- `segment`

Known `test.tsv` columns:

- `client_id`
- `path`
- `sentence_id`
- `sentence`
- `sentence_domain`
- `up_votes`
- `down_votes`
- `age`
- `gender`
- `accents`
- `variant`
- `locale`
- `segment`

Filtering:

1. Use the ASR-targeted `test` split as the final test seed.
2. Build exclusion sets from test:
   - normalized sentence text from `sentence`
   - speaker ID from `client_id`
3. Remove from `validated` any row with matching normalized `sentence`.
4. Remove from `validated` any row with matching `client_id`.
5. Remove any row whose transcript is a single word.
6. Inspect which regional or accent metadata fields are actually available and usable for a Swedish dialect-classifier training set.

### 3. Extra Swedish Dataset

Source inputs:

- `KTH/nst`

Known columns:

- `speaker_id`
- `age`
- `gender`
- `region_of_birth`
- `region_of_youth`
- `text`
- `path`
- `audio`
- `text_normalised`

Filtering:

1. Use `text_normalised` as the primary transcript field for filtering and training text unless inspection shows it is unsuitable.
2. Keep `text` as auxiliary metadata so the original orthography is not lost.
3. Remove any transcript whose normalized text is a single word.
4. Remove any transcript whose normalized text is only a single spelled-out word.
5. Preserve `speaker_id` as the speaker metadata field.
6. Preserve `age`, `gender`, `region_of_birth`, and `region_of_youth` as optional metadata columns if they are useful downstream.
7. Exclude rows with normalized sentence text matching the Common Voice Swedish test material before merge.
8. For dialect-classifier supervision, keep only rows where `region_of_birth == region_of_youth`.
9. Use the agreed regional field as the dialect label source for those retained rows.

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

Implement that rule against `text_normalised`. The exact regex or heuristic should be documented beside the filtering script.

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
3. `train` = Common Voice Swedish train plus filtered `KTH/nst`

Keep the Swedish validation set Common Voice-only unless there is a strong reason to validate on the merged distribution.

### Swedish dialect labels

The Swedish run needs a dialect classifier model for the dialect axis.

Label plan:

1. For `KTH/nst`, derive dialect labels only from rows where `region_of_birth == region_of_youth`.
2. Use the shared region value from those rows as the dialect label.
3. Exclude rows with mismatched `region_of_birth` and `region_of_youth` from the dialect-classifier training set.
4. For Common Voice Swedish, inspect what dialect-, accent-, or region-like metadata is actually present before deciding whether it can contribute to the Swedish dialect-classifier training data.
5. If Common Voice Swedish does not provide usable region labels, train the dialect classifier from `KTH/nst` only and use it as the teacher for the Swedish run.

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

For `KTH/nst`, also retain when practical:

- `text_normalised`
- `age`
- `gender`
- `region_of_birth`
- `region_of_youth`
- `path`

Recommended split packaging:

- create a `DatasetDict`
- store one directory per run
- keep the three final splits only: `train`, `validation`, `test`

Example output names:

- `cv-en`
- `cv-sv-plus`

## Packaging

Preferred format:

Use:

```python
dataset_dict.save_to_disk("/scratch/$USER/asr/hf/cv-en")
dataset_dict.save_to_disk("/scratch/$USER/asr/hf/cv-sv-plus")
```

This matches `spoken-sentence-transformers/experiment/train_wavlm.py`, which expects:

- `--dataset_dir`
- a dataset saved with `save_to_disk(...)`

Optional multi-axis target precomputation also already exists in:

- `~/Playing/spoken-sentence-transformers/experiment/precompute_targets.py`

That script writes a companion targets dataset for use as:

- `--targets_dir`

If you later want Hub publication, treat that as optional distribution, not as the primary training input path.

## Shared Model Choices

- Use the same speaker ID model/backend setup across English and Swedish.
- Keep the existing English semantic teacher choice for the English run.
- Use `KBLab/sentence-bert-swedish-cased` as the Swedish semantic teacher.
- Add a Swedish dialect classifier teacher once its label set has been prepared.

## Unresolved Choices

These are still open and should not be treated as decided:

- whether these runs should be semantic-only or multi-axis
- which existing English semantic teacher model name should be written explicitly into the final config
- what usable dialect or region metadata Common Voice Swedish actually exposes
- whether the Swedish dialect classifier will be trained from `KTH/nst` only or from `KTH/nst` plus a labeled Common Voice subset

## Training Launch

Dataset preparation happens in this repo. Training happens from `~/Playing/spoken-sentence-transformers/`, or from its cluster checkout at `/scratch/$USER/asr/spoken-sentence-transformers`, one run at a time on a single node with 8 GPUs.

### Environment

- 1 node
- 8 GPUs
- 8 tasks
- 1 process per GPU

### Suggested Slurm Script Shape

Create `slurm/train_launcher.sh` in `spoken-sentence-transformers` and parameterize the config path via environment variables.

```bash
#!/bin/bash
#SBATCH --job-name=sst-train
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
export NPROC=8

bash experiment/run_train.sh
```

This wrapper expects `CONFIG_FILE` and calls:

```bash
torchrun --nproc_per_node="$NPROC" experiment/train_wavlm.py "$CONFIG_FILE"
```

### Suggested Config Pattern

Create one JSON config per run in `spoken-sentence-transformers/experiment/configs/`.

Use concrete absolute paths in JSON. Do not rely on `$USER` or other shell variables inside the config file itself.

Template pattern only:

- replace the axis list with the real run design
- set `semantic_model` to the teacher chosen for the language
- include `targets_dir` only if you are precomputing teacher outputs
- add the Swedish dialect target model only after the classifier has been trained

Example skeleton:

```json
{
  "model_id": "microsoft/wavlm-base-plus",
  "encoder_dim": 768,
  "axes": ["<decide-per-run>"],
  "semantic_model": "<choose-for-language-if-used>",
  "dataset_dir": "/scratch/<USER>/asr/hf/<dataset-name>",
  "output_dir": "/scratch/<USER>/asr/models/<run-name>",
  "per_device_train_batch_size": 16,
  "num_train_epochs": 10,
  "learning_rate": 0.0001,
  "warmup_ratio": 0.1,
  "fp16": true,
  "save_strategy": "epoch",
  "eval_strategy": "epoch",
  "logging_steps": 100,
  "report_to": "tensorboard",
  "dataloader_num_workers": 4,
  "gradient_checkpointing": true
}
```

If using precomputed targets, add:

- `targets_dir`

If using speaker contrastive behavior or held-out speaker retrieval evaluation, ensure the dataset includes `speaker_id` and set:

- `val_speaker_ids`
- `val_speaker_column`
- `val_eval_k`

## Concrete Run Names

### Run 1: English Common Voice

Suggested values:

- `RUN_NAME=cv-en-<variant>`
- `DATASET_DIR=/scratch/$USER/asr/hf/cv-en`
- `CONFIG_FILE=/scratch/$USER/asr/spoken-sentence-transformers/experiment/configs/cv-en-<variant>.json`

Launch:

```bash
sbatch --export=ALL,CONFIG_FILE=/scratch/$USER/asr/spoken-sentence-transformers/experiment/configs/cv-en-<variant>.json /scratch/$USER/asr/spoken-sentence-transformers/slurm/train_launcher.sh
```

### Run 2: Swedish Common Voice + extra Swedish data

Suggested values:

- `RUN_NAME=cv-sv-plus-<variant>`
- `DATASET_DIR=/scratch/$USER/asr/hf/cv-sv-plus`
- `CONFIG_FILE=/scratch/$USER/asr/spoken-sentence-transformers/experiment/configs/cv-sv-plus-<variant>.json`

For Swedish semantic distillation, set:

- `semantic_model`: `KBLab/sentence-bert-swedish-cased`

Launch:

```bash
sbatch --export=ALL,CONFIG_FILE=/scratch/$USER/asr/spoken-sentence-transformers/experiment/configs/cv-sv-plus-<variant>.json /scratch/$USER/asr/spoken-sentence-transformers/slurm/train_launcher.sh
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
2. Write one preparation script for `KTH/nst` using `text_normalised` as the primary filtering field.
3. Choose the semantic teacher model for English, if semantic distillation is part of the design.
4. Build `cv-en` as a final `DatasetDict`.
5. Build `cv-sv-plus` as a final `DatasetDict`.
6. Inspect Common Voice Swedish metadata to see whether it can contribute labels for a Swedish dialect classifier.
7. Build the Swedish dialect-classifier training set, using `KTH/nst` rows where `region_of_birth == region_of_youth`.
8. Train the Swedish dialect classifier model.
9. Save both ASR training datasets with `save_to_disk(...)`.
10. If needed, run `experiment/precompute_targets.py` to build matching `targets_dir` datasets.
11. Add run-specific JSON configs in `spoken-sentence-transformers/experiment/configs/`.
12. Add the Slurm launcher script in `spoken-sentence-transformers`.
13. Run English first, then Swedish merged.

## Short Version

- English run: cleaned Common Voice English only
- Swedish run: cleaned Common Voice Swedish plus filtered `KTH/nst`
- test set basis: ASR-targeted Common Voice test
- leakage prevention: remove matching sentence text and speaker IDs from `validated`
- transcript cleanup: remove single-word items everywhere; for `KTH/nst`, apply the spelled-out single-word filter against `text_normalised`
- training target: local Hugging Face `save_to_disk(...)` dataset per run
- training code: `spoken-sentence-transformers/experiment/train_wavlm.py`
- shared speaker targets: use the same speaker ID models across both runs
- Swedish dialect teacher: train from labeled Swedish regional data, with `KTH/nst` labels gated by `region_of_birth == region_of_youth`
- compute target: 1 Slurm node, 8 GPUs
