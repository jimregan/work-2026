# Adding a Dataset

Datasets are HuggingFace `Dataset` objects with columns that follow a naming
convention.  The trainer does not care what the data *means* — it only looks at
column names to decide how to route tensors to the model.

---

## Column naming convention

Every column name has the form `{role}_{suffix}`:

| Part | Values | Meaning |
|---|---|---|
| `role` | `anchor`, `{axis}_pos` | Which position in the contrastive pair |
| `suffix` | `input_features`, `input_ids`, `sentence_embedding` | How the data is encoded |

Examples for a model with axes `semantic`, `speaker_id`, `gender`:

| Column | Meaning |
|---|---|
| `anchor_input_features` | Mel spectrogram of the anchor utterance |
| `semantic_pos_sentence_embedding` | Pre-cached text embedding of the transcript |
| `speaker_id_pos_input_features` | Mel spectrogram of a different utterance by the same speaker |
| `gender_pos_sentence_embedding` | Pre-cached embedding of an utterance with the same gender label |

All roles for a given suffix must be present for a training step to work.  You
do not need a positive for every axis in every row — `MultiAxisInfoNCELoss`
skips axes whose `{axis}_pos` key is absent.

---

## Label columns (for the batch sampler)

`MultiAxisNoDuplicatesBatchSampler` needs one label column per axis to avoid
false negatives in a batch.  Name them `{axis}_label`:

```python
dataset = dataset.add_column("semantic_label",   transcript_ids)
dataset = dataset.add_column("speaker_id_label", speaker_ids)
dataset = dataset.add_column("gender_label",     gender_labels)
```

Pass these to the trainer:

```python
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
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

The cleanest way to supply semantic content labels is to run transcripts through
a text sentence transformer offline and store the result as
`semantic_pos_sentence_embedding`.  This bypasses the acoustic encoder entirely
for that column — the trainer routes `_sentence_embedding` columns directly to
the projection module.

```python
from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer("all-MiniLM-L6-v2")

# dataset has columns: audio, transcript, speaker_id, gender
transcripts = dataset["transcript"]
text_embs = text_model.encode(transcripts, batch_size=256, show_progress_bar=True)

dataset = dataset.add_column("anchor_input_features",         audio_features)
dataset = dataset.add_column("semantic_pos_sentence_embedding", text_embs.tolist())
dataset = dataset.add_column("semantic_label",                transcripts)   # text = content ID
dataset = dataset.add_column("speaker_id_pos_input_features", same_speaker_features)
dataset = dataset.add_column("speaker_id_label",              speaker_ids)
```

The pre-computed embedding dimension must match the `in_features` of the
`MultiAxisProjection` (i.e. the output of the acoustic encoder + pooling), since
both paths feed into the same projection heads.  If dimensions differ, add a
linear adapter or choose a text model whose output matches.

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
import torch
from datasets import Dataset
from spoken_sentence_transformers import (
    MultiAxisProjection,
    MultiAxisSentenceTransformer,
    MultiAxisInfoNCELoss,
    MultiAxisProjectionTrainer,
    MultiAxisNoDuplicatesBatchSampler,
)
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from spoken_sentence_transformers.encoders import HFAcousticEncoder
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments, BatchSamplers,
)

AXES = {"semantic": 256, "speaker_id": 128}
ENCODER_DIM = 1024  # e.g. WavLM-large hidden size

encoder  = HFAcousticEncoder("microsoft/wavlm-large")
pooling  = Pooling(ENCODER_DIM, pooling_mode="mean")
proj     = MultiAxisProjection(in_features=ENCODER_DIM, axes=AXES)
model    = MultiAxisSentenceTransformer(modules=[encoder, pooling, proj])

dataset = Dataset.from_dict({
    "anchor_input_features":           [...],  # list of np.ndarray waveforms
    "semantic_pos_sentence_embedding": [...],  # list of float lists (text embs)
    "speaker_id_pos_input_features":   [...],  # same-speaker waveforms
    "semantic_label":                  [...],  # transcript strings (content IDs)
    "speaker_id_label":                [...],  # speaker ID strings
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
