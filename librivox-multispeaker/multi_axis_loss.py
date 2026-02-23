"""Per-axis InfoNCE loss for MultiAxisSentenceTransformer.

Each axis is trained independently: its positive pool is the only source of
in-batch negatives for that axis's comparison.  This prevents cross-axis
contamination (a speaker-axis positive from another anchor appearing as a
negative in the content-axis comparison).

Model usage
-----------
The anchor is run through the model **once**; all axis projections are
computed simultaneously by ``MultiAxisProjection`` and cached in the features
dict.  Each axis's positive is run through the model separately, and only
that axis's projection is extracted.  This is the minimum number of forward
passes required.

For pre-cached embeddings the model should contain only the
``MultiAxisProjection`` module (no ``Transformer`` or ``Pooling``), so the
features dict ``{"sentence_embedding": Tensor[B, D]}`` is passed directly to
the projection heads.  For end-to-end audio/text training the full pipeline
(``Transformer → Pooling → MultiAxisProjection``) is used instead; the loss
function is identical in both cases.

Dataset / batch requirements
-----------------------------
* ``named_features["anchor"]``        — anchor feature dict
* ``named_features["{axis}_pos"]``    — positive feature dict for each axis

The ``MultiAxisNoDuplicatesBatchSampler`` should be used to guarantee that no
two anchors in the same batch share a label for any axis, so every off-diagonal
entry in each B×B similarity matrix is a true negative.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from multi_axis_sentence_transformer import MultiAxisSentenceTransformer


class MultiAxisInfoNCELoss(nn.Module):
    """Per-axis InfoNCE loss for multi-axis contrastive speech training.

    For each axis *k* the loss is:

    .. math::

        \\mathcal{L}_k = \\text{CrossEntropy}\\!\\left(
            \\frac{z^{(a)}_k \\cdot Z^{(k)\\top}}{\\tau},\\;
            \\mathbf{I}
        \\right)

    where :math:`z^{(a)}_k` is the L2-normalised anchor projection for axis
    *k* (shape ``[B, D_k]``) and :math:`Z^{(k)}` is the matrix of
    L2-normalised projections for axis *k*'s positives (same shape).  The
    diagonal of the ``[B, B]`` logit matrix is the true positive; all other
    entries are in-batch negatives drawn exclusively from that axis's positive
    pool.

    The total loss returned to the trainer is a ``dict[str, Tensor]`` so that
    each axis component is logged individually via
    :meth:`~multi_axis_trainer.MultiAxisProjectionTrainer.track_loss_components`.

    Args:
        model: The :class:`~multi_axis_sentence_transformer.MultiAxisSentenceTransformer`
            whose ``MultiAxisProjection`` module produces per-axis embeddings.
        temperature: Softmax temperature *τ* applied to the logit matrix.
            Smaller values make the distribution sharper.  Defaults to 0.05.
        axis_weights: Optional mapping of axis name to loss weight.  Missing
            axes default to weight 1.0.  Weights scale the loss *before*
            summing, so they affect both the gradient magnitude and the logged
            per-axis values.
    """

    def __init__(
        self,
        model: MultiAxisSentenceTransformer,
        temperature: float = 0.05,
        axis_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.axis_weights: dict[str, float] = axis_weights or {}

    def forward(
        self,
        named_features: dict[str, dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute per-axis InfoNCE losses.

        Args:
            named_features: Dict produced by
                :meth:`~multi_axis_trainer.MultiAxisProjectionTrainer.collect_features`.
                Must contain ``"anchor"`` and ``"{axis}_pos"`` for at least
                one axis.
            labels: Unused; present to satisfy the trainer's loss interface.

        Returns:
            ``dict[str, Tensor]`` mapping each axis name to its (weighted)
            scalar InfoNCE loss.  The trainer sums these for backprop and logs
            each component individually.
        """
        # --- Run anchor through the model ONCE. ---
        # MultiAxisProjection computes all axis projections simultaneously and
        # stores them as features["embedding_{axis}"] in the returned dict.
        anchor_out = self.model(named_features["anchor"])

        losses: dict[str, Tensor] = {}

        for axis in self.model.axes:
            pos_key = f"{axis}_pos"
            if pos_key not in named_features:
                continue

            # Run this axis's positives through the model.
            # We only use the projection for axis k, but all heads run;
            # there is no per-head computational saving to be had.
            pos_out = self.model(named_features[pos_key])

            # Normalise: each row becomes a unit vector on the hypersphere.
            anchor_proj = F.normalize(anchor_out[f"embedding_{axis}"], p=2, dim=-1)
            pos_proj    = F.normalize(pos_out[f"embedding_{axis}"],    p=2, dim=-1)

            # B×B logit matrix.  Diagonal = true positive.
            # Off-diagonal = in-batch negatives from this axis's pool only.
            loss = self._info_nce(anchor_proj, pos_proj)

            weight = self.axis_weights.get(axis, 1.0)
            losses[axis] = weight * loss

        if not losses:
            raise ValueError(
                "MultiAxisInfoNCELoss: no axis positives found in named_features. "
                f"Expected keys like 'content_pos', 'speaker_pos', …; "
                f"got: {list(named_features.keys())}"
            )

        return losses

    def _info_nce(self, anchor: Tensor, positive: Tensor) -> Tensor:
        """Symmetric InfoNCE over a B×B similarity matrix.

        ``anchor[i]`` is the true positive for ``positive[i]``; all other
        ``positive[j]`` (j ≠ i) are negatives for anchor ``i``.

        Args:
            anchor:   L2-normalised embeddings, shape ``[B, D]``.
            positive: L2-normalised embeddings, shape ``[B, D]``.

        Returns:
            Scalar cross-entropy loss.
        """
        logits = anchor @ positive.T / self.temperature
        targets = torch.arange(len(anchor), device=anchor.device)
        return F.cross_entropy(logits, targets)
