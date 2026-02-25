# Adding a New Encoder

Encoders bridge an audio backend into the SentenceTransformer pipeline.
All encoders must subclass `AcousticEncoder` from
`spoken_sentence_transformers.encoders`.

```
YourEncoder → Pooling → MultiAxisProjection
              ↑
              writes token_embeddings [B, T, H]
              and attention_mask [B, T]
```

The `Pooling` module (from `sentence_transformers`) collapses the per-frame
representations to a single vector.  Your encoder does not pool — it only
produces frame-level hidden states.

---

## The five abstract methods

### `tokenize(audio_list, **kwargs) -> dict[str, Tensor]`

Called once per batch before `forward()`.  Converts raw audio into whatever
tensors your model needs.

- Input: list of `np.ndarray` waveforms, or HuggingFace audio dicts
  `{"array": np.ndarray, "sampling_rate": int}`
- Output: a dict of tensors that will be passed to `forward()`

```python
def tokenize(self, audio_list, **kwargs):
    # example: return codec token IDs
    codes = [self.codec.encode(w) for w in audio_list]
    return {"codec_codes": torch.nn.utils.rnn.pad_sequence(codes, batch_first=True)}
```

### `forward(features, **kwargs) -> dict[str, Tensor]`

Called during the model forward pass.  Must write two keys into `features`:

- `token_embeddings`: shape `[B, T, H]` — per-frame hidden states
- `attention_mask`: shape `[B, T]` — 1 for valid frames, 0 for padding

```python
def forward(self, features, **kwargs):
    codes = features["codec_codes"]                  # [B, T]
    hidden = self.embedding_table(codes)             # [B, T, H]
    mask = (codes != self.pad_id).long()             # [B, T]
    features["token_embeddings"] = hidden
    features["attention_mask"] = mask
    return features
```

### `get_word_embedding_dimension() -> int`

Returns the size of the last dimension of `token_embeddings` (H above).
Used by `Pooling` and `MultiAxisProjection` to set up their layer sizes.

```python
def get_word_embedding_dimension(self):
    return self.hidden_size
```

### `save(output_path, *args, safe_serialization=True, **kwargs) -> None`

Save whatever is needed to reconstruct this encoder from `output_path`.

```python
def save(self, output_path, *args, safe_serialization=True, **kwargs):
    torch.save(self.state_dict(),
               os.path.join(output_path, "codec_encoder.pt"))
    with open(os.path.join(output_path, "codec_encoder_config.json"), "w") as f:
        json.dump({"hidden_size": self.hidden_size, "vocab_size": self.vocab_size}, f)
    self.save_config(output_path)   # saves sentence_bert_config.json
```

### `load(cls, model_name_or_path, **kwargs) -> Self`

Reconstruct from a saved directory.

```python
@classmethod
def load(cls, model_name_or_path, **kwargs):
    with open(os.path.join(model_name_or_path, "codec_encoder_config.json")) as f:
        cfg = json.load(f)
    enc = cls(**cfg)
    enc.load_state_dict(
        torch.load(os.path.join(model_name_or_path, "codec_encoder.pt"),
                   map_location="cpu", weights_only=True)
    )
    return enc
```

---

## Full skeleton

```python
from __future__ import annotations
import json, os
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import Tensor, nn

from spoken_sentence_transformers.encoders import AcousticEncoder


class CodecEncoder(AcousticEncoder):
    """Wraps a discrete codec (e.g. EnCodec) as an ST-compatible encoder.

    Tokenizes audio into codec codes, embeds them with a learned table,
    and returns per-frame hidden states for downstream pooling.
    """

    config_keys = ["hidden_size", "vocab_size", "pad_id"]

    def __init__(self, hidden_size: int, vocab_size: int, pad_id: int = 0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embedding_table = nn.Embedding(vocab_size, hidden_size,
                                            padding_idx=pad_id)

    # --- tokenize -----------------------------------------------------------

    def tokenize(self, audio_list: list[np.ndarray | dict[str, Any]],
                 **kwargs) -> dict[str, Tensor]:
        waveforms = [
            np.asarray(a["array"] if isinstance(a, dict) else a, dtype=np.float32)
            for a in audio_list
        ]
        # Replace with real codec call:
        codes = [torch.zeros(50, dtype=torch.long) for _ in waveforms]
        padded = torch.nn.utils.rnn.pad_sequence(
            codes, batch_first=True, padding_value=self.pad_id
        )
        return {"codec_codes": padded}

    # --- forward ------------------------------------------------------------

    def forward(self, features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        codes = features["codec_codes"]               # [B, T]
        hidden = self.embedding_table(codes)          # [B, T, H]
        mask = (codes != self.pad_id).long()          # [B, T]
        features["token_embeddings"] = hidden
        features["attention_mask"] = mask
        return features

    # --- dimension ----------------------------------------------------------

    def get_word_embedding_dimension(self) -> int:
        return self.hidden_size

    # --- save / load --------------------------------------------------------

    def save(self, output_path: str, *args,
             safe_serialization: bool = True, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(output_path, "codec_encoder.pt"))
        with open(os.path.join(output_path, "codec_encoder_config.json"), "w") as f:
            json.dump({
                "hidden_size": self.hidden_size,
                "vocab_size":  self.vocab_size,
                "pad_id":      self.pad_id,
            }, f)
        self.save_config(output_path)

    @classmethod
    def load(cls, model_name_or_path: str, **kwargs) -> Self:
        with open(os.path.join(model_name_or_path,
                               "codec_encoder_config.json")) as f:
            cfg = json.load(f)
        enc = cls(**cfg)
        enc.load_state_dict(
            torch.load(os.path.join(model_name_or_path, "codec_encoder.pt"),
                       map_location="cpu", weights_only=True)
        )
        return enc
```

---

## Registering a new input column suffix

If your `tokenize()` uses a key other than `input_features`, `input_ids`,
`sentence_embedding`, or `pixel_values`, tell the trainer:

```python
trainer = MultiAxisProjectionTrainer(
    ...,
    feature_suffixes=MultiAxisProjectionTrainer.DEFAULT_FEATURE_SUFFIXES
        + ("codec_codes",),
)
```

Then name your dataset columns `anchor_codec_codes`,
`speaker_id_pos_codec_codes`, etc.

---

## Augmenting an existing encoder with derived features

To add a derived signal (pitch contours, formant trajectories, classifier
logits) alongside acoustic frames, wrap an existing encoder:

```python
class PitchAugmentedEncoder(AcousticEncoder):
    def __init__(self, base: HFAcousticEncoder, pitch_dim: int) -> None:
        super().__init__()
        self.base = base
        self.pitch_proj = nn.Linear(pitch_dim, base.get_word_embedding_dimension())

    def tokenize(self, audio_list, **kwargs):
        # audio_list entries are dicts with keys "array", "sampling_rate", "pitch"
        base_feats = self.base.tokenize(
            [{"array": a["array"], "sampling_rate": a["sampling_rate"]}
             for a in audio_list], **kwargs
        )
        pitch = torch.stack([torch.tensor(a["pitch"]) for a in audio_list])
        base_feats["pitch_values"] = pitch
        return base_feats

    def forward(self, features, **kwargs):
        features = self.base.forward(features, **kwargs)
        pitch_emb = self.pitch_proj(features["pitch_values"])  # [B, T, H]
        features["token_embeddings"] = features["token_embeddings"] + pitch_emb
        return features

    def get_word_embedding_dimension(self):
        return self.base.get_word_embedding_dimension()

    # save/load: delegate to base + save pitch_proj weights
    ...
```
