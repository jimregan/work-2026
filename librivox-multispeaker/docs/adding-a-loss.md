# Adding a New Loss / Comparison Method

A loss function is any `nn.Module` whose `forward` method accepts
`(named_features, labels)` and returns either a scalar `Tensor` or a
`dict[str, Tensor]` of per-axis losses.

Returning a dict is preferred: the trainer logs each component individually
and sums them for backprop.

---

## The `named_features` structure

`named_features` is produced by `MultiAxisProjectionTrainer.collect_features`
from the batch.  Its keys match the role prefixes in your dataset columns:

```python
named_features = {
    "anchor":        {"input_features": Tensor[B, ...]},   # or sentence_embedding, etc.
    "semantic_pos":  {"sentence_embedding": Tensor[B, D]},
    "speaker_id_pos":{"input_features": Tensor[B, ...]},
    # ... one entry per role present in the batch
}
```

To get an embedding out of a role's feature dict, run it through the model:

```python
anchor_out = self.model(named_features["anchor"])
# anchor_out["sentence_embedding"]     → pooled, projected, concatenated
# anchor_out["embedding_semantic"]     → L2-normalised semantic subspace
# anchor_out["embedding_speaker_id"]   → L2-normalised speaker subspace
```

---

## Minimal example: per-axis triplet loss

```python
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spoken_sentence_transformers import MultiAxisSentenceTransformer


class MultiAxisTripletLoss(nn.Module):
    """Per-axis triplet loss with random in-batch negatives.

    For each axis the loss is max(0, margin - sim(a,p) + sim(a,n))
    where the negative is a randomly permuted positive from the same axis pool.
    """

    def __init__(
        self,
        model: "MultiAxisSentenceTransformer",
        margin: float = 0.2,
        axis_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.margin = margin
        self.axis_weights = axis_weights or {}

    def forward(
        self,
        named_features: dict[str, dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        anchor_out = self.model(named_features["anchor"])
        losses: dict[str, Tensor] = {}

        for axis in self.model.axes:
            pos_key = f"{axis}_pos"
            if pos_key not in named_features:
                continue

            pos_out = self.model(named_features[pos_key])

            a = anchor_out[f"embedding_{axis}"]   # [B, D]  unit vectors
            p = pos_out[f"embedding_{axis}"]       # [B, D]

            # Random in-batch negatives: roll the positive matrix by 1
            n = torch.roll(p, shifts=1, dims=0)

            sim_ap = F.cosine_similarity(a, p, dim=-1)   # [B]
            sim_an = F.cosine_similarity(a, n, dim=-1)   # [B]

            loss = F.relu(self.margin - sim_ap + sim_an).mean()
            losses[axis] = self.axis_weights.get(axis, 1.0) * loss

        if not losses:
            raise ValueError(
                f"No axis positives found in named_features: "
                f"{list(named_features.keys())}"
            )
        return losses
```

Usage is identical to `MultiAxisInfoNCELoss`:

```python
loss = MultiAxisTripletLoss(model, margin=0.3,
                            axis_weights={"semantic": 1.0, "speaker_id": 0.5})
trainer = MultiAxisProjectionTrainer(model=model, ..., loss=loss)
```

---

## Example: comparing against classical speaker embeddings

A loss that regularises the speaker axis to match a pre-trained speaker
embedding system (e.g. x-vectors stored in the dataset as
`speaker_id_pos_sentence_embedding`):

```python
class SpeakerAlignmentLoss(nn.Module):
    """Pull speaker axis towards a reference embedding (x-vector, ECAPA, …).

    The reference embeddings are stored in the dataset as pre-cached
    sentence_embeddings (suffix _sentence_embedding) under the speaker_id_pos
    role.  This loss minimises cosine distance between the model's speaker
    axis projection and the reference.
    """

    def __init__(self, model: "MultiAxisSentenceTransformer",
                 temperature: float = 0.05) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, named_features, labels=None) -> dict[str, Tensor]:
        if "speaker_id_pos" not in named_features:
            return {}

        anchor_out = self.model(named_features["anchor"])
        ref_out    = self.model(named_features["speaker_id_pos"])

        a = anchor_out["embedding_speaker_id"]    # [B, D]  unit vector
        r = ref_out["sentence_embedding"]         # [B, D_ref] pre-cached
        r = F.normalize(r.float(), p=2, dim=-1)   # ensure unit norm

        # If dims differ, project r down; simpler to match dims at dataset build time.
        logits  = a @ r.T / self.temperature      # [B, B]
        targets = torch.arange(len(a), device=a.device)
        loss    = F.cross_entropy(logits, targets)
        return {"speaker_alignment": loss}
```

---

## Combining multiple losses

For multi-dataset training, pass a dict keyed by dataset name:

```python
trainer = MultiAxisProjectionTrainer(
    model=model,
    train_dataset=DatasetDict({
        "librispeech": librispeech_ds,
        "voxceleb":    voxceleb_ds,
    }),
    loss={
        "librispeech": MultiAxisInfoNCELoss(model),
        "voxceleb":    SpeakerAlignmentLoss(model),
    },
)
```

Each dataset's batch is routed to its paired loss.  Both contribute to the
same model parameters.

---

## Checklist for a new loss

- [ ] Subclass `nn.Module`
- [ ] Accept `(named_features: dict[str, dict[str, Tensor]], labels: Tensor | None)`
- [ ] Return `dict[str, Tensor]` (preferred) or scalar `Tensor`
- [ ] Access model via `self.model` passed at construction
- [ ] Use `self.model.axes` to iterate axes — do not hardcode axis names
- [ ] Embeddings in `embedding_{axis}` are already L2-normalised unit vectors;
      do not normalise again (idempotent but wastes compute)
- [ ] Skip axes where `{axis}_pos` is absent in `named_features` (not every
      dataset supplies every axis)
