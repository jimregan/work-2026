# Adding a Dataset

Datasets are HuggingFace `Dataset` objects with columns that follow a naming
convention.  The trainer does not interpret the semantics of the data — it only
inspects column names to determine how tensors should be routed to the model.

Each dataset row represents a **single anchor example** plus optional positive
samples for different axes.  The dataset is not symmetric; each row is
anchor-centric.

---

## Column naming convention

Every column name has the form `{role}_{suffix}`:

```
{role}_{suffix}

role:    anchor
         {axis}_pos

suffix:  input_features       (raw audio / mel spectrogram)
         input_ids            (tokenised text)
         sentence_embedding   (pre-computed dense vector)
```

`{axis}_pos` means a sample that **shares the value of that axis with the
anchor**.  For example, `speaker_id_pos_input_features` is an utterance spoken
by the same speaker as the anchor; `semantic_pos_sentence_embedding` is an
embedding whose text matches the anchor's transcript.

Examples for a model with axes `semantic`, `speaker_id`, `gender`:

| Column | Meaning |
|---|---|
| `anchor_input_features` | Mel spectrogram of the anchor utterance |
| `semantic_pos_sentence_embedding` | Pre-cached text embedding of the transcript |
| `speaker_id_pos_input_features` | Mel spectrogram of a different utterance by the same speaker |
| `gender_pos_sentence_embedding` | Pre-cached embedding of an utterance with the same gender label |

For each suffix used in a row (e.g. `input_features`), the corresponding
`anchor_*` column must be present.  Positives are optional — axes whose
`{axis}_pos_*` columns are missing in a row are skipped for that example.
Other axes still contribute to the loss.

---

## How columns are routed

The suffix determines how a tensor reaches the projection module:

```
anchor_input_features          → encoder → pooling → projection
semantic_pos_sentence_embedding → projection          (skips encoder)
speaker_id_pos_input_features  → encoder → pooling → projection
```

`_sentence_embedding` columns bypass the acoustic encoder entirely and are fed
directly into the `MultiAxisProjection`.  This is the recommended path for
pre-computed content positives (see below).

---

## Label columns (for the batch sampler)

`MultiAxisNoDuplicatesBatchSampler` prevents false negatives by ensuring that
two samples with the same `{axis}_label` never appear in the same batch unless
they are explicitly paired as positives.  Name label columns `{axis}_label`:

```python
dataset = dataset.add_column("semantic_label",   transcript_ids)
dataset = dataset.add_column("speaker_id_label", speaker_ids)
dataset = dataset.add_column("gender_label",     gender_labels)
```

Pass these to the trainer:

```python
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from spoken_sentence_transformers import MultiAxisNoDuplicatesBatchSampler

args = SentenceTransformerTrainingArguments(
    output_dir="...",
    batch_sampler=MultiAxisNoDuplicatesBatchSampler,
)
trainer = MultiAxisProjectionTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss_fn,
)
```

---

## Using pre-cached text embeddings for semantic content

The cleanest way to supply semantic content positives is to run transcripts
through a text sentence transformer offline and store the result as
`semantic_pos_sentence_embedding`.  This bypasses the acoustic encoder entirely
for that column.

```python
from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer("all-MiniLM-L6-v2")

# dataset has columns: audio, transcript, speaker_id, gender
transcripts = dataset["transcript"]
text_embs = text_model.encode(transcripts, batch_size=256, show_progress_bar=True)

dataset = dataset.add_column("anchor_input_features",          audio_features)
dataset = dataset.add_column("semantic_pos_sentence_embedding", text_embs.tolist())
dataset = dataset.add_column("semantic_label",                 transcripts)   # text = content ID
dataset = dataset.add_column("speaker_id_pos_input_features",  same_speaker_features)
dataset = dataset.add_column("speaker_id_label",               speaker_ids)
```

> **Important**
>
> `_sentence_embedding` columns bypass the encoder and are fed directly into
> the projection module.  Therefore the embedding dimension **must match the
> projection input dimension** (`in_features` of `MultiAxisProjection`, i.e.
> the output of the acoustic encoder + pooling).  If dimensions differ, add a
> linear adapter or choose a text model whose output dimension matches.

---

## Adding a new input modality

If your encoder produces a feature type not in the default list
(`input_features`, `input_ids`, `sentence_embedding`, `pixel_values`), register
the new suffix with the trainer:

```python
trainer = MultiAxisProjectionTrainer(
    ...,
    feature_suffixes=MultiAxisProjectionTrainer.DEFAULT_FEATURE_SUFFIXES
        + ("pitch_values",),
)
```

Then name your columns `anchor_pitch_values`, `semantic_pos_pitch_values`, etc.

---

## Minimal working example

```python
import numpy as np
from datasets import Dataset
from spoken_sentence_transformers import (
    MultiAxisProjection,
    MultiAxisSentenceTransformer,
    MultiAxisInfoNCELoss,
    MultiAxisProjectionTrainer,
    MultiAxisNoDuplicatesBatchSampler,
)
from sentence_transformers.models import Pooling
from spoken_sentence_transformers.encoders import HFAcousticEncoder
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

AXES = {"semantic": 256, "speaker_id": 128}
ENCODER_DIM = 1024  # e.g. WavLM-large hidden size

encoder = HFAcousticEncoder("microsoft/wavlm-large")
pooling  = Pooling(ENCODER_DIM, pooling_mode="mean")
proj     = MultiAxisProjection(in_features=ENCODER_DIM, axes=AXES)
model    = MultiAxisSentenceTransformer(modules=[encoder, pooling, proj])

dataset = Dataset.from_dict({
    "anchor_input_features":           [...],  # list of np.ndarray, shape (samples,)
    "semantic_pos_sentence_embedding": [...],  # list of list[float], length ENCODER_DIM
    "speaker_id_pos_input_features":   [...],  # list of np.ndarray, shape (samples,)
    "semantic_label":                  [...],  # list of str  — transcript text
    "speaker_id_label":                [...],  # list of str  — speaker ID
})

loss = MultiAxisInfoNCELoss(model)
args = SentenceTransformerTrainingArguments(
    output_dir="output",
    batch_sampler=MultiAxisNoDuplicatesBatchSampler,
    per_device_train_batch_size=32,
    num_train_epochs=10,
)
trainer = MultiAxisProjectionTrainer(model=model, args=args,
                                     train_dataset=dataset, loss=loss)
trainer.train()
```
