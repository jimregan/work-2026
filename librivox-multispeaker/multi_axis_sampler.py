"""Batch sampler for multi-axis contrastive training.

For InfoNCE to work correctly, every item in a batch that shares an axis label
with another item in the same batch is a potential false negative for that
axis's comparison.  ``MultiAxisNoDuplicatesBatchSampler`` prevents this by
ensuring that, for every axis, all label values within a batch are unique.

Dataset format
--------------
The dataset must contain one label column per axis, named according to the
convention ``{axis}_label`` (e.g. ``content_label``, ``speaker_label``).
These columns are passed as ``valid_label_columns`` to the sampler (matching
the keyword the :class:`~multi_axis_trainer.MultiAxisProjectionTrainer` passes
when calling ``get_batch_sampler``).

Example dataset row::

    {
        "anchor_sentence_embedding":      Tensor[D],
        "content_pos_sentence_embedding": Tensor[D],
        "speaker_pos_sentence_embedding": Tensor[D],
        "content_label": "sentence_42",   # content class — same text across speakers
        "speaker_label": "reader_007",    # speaker identity
    }
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from sentence_transformers.sampler import DefaultBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset

logger = logging.getLogger(__name__)


class MultiAxisNoDuplicatesBatchSampler(DefaultBatchSampler):
    """Batch sampler that guarantees no false negatives across all axes.

    For each axis, maintains a set of labels already present in the
    current batch.  A candidate sample is admitted only if its label
    for **every** axis is absent from the batch.  This ensures that
    for any axis-k InfoNCE comparison, every non-diagonal entry in the
    B×B similarity matrix is a true negative.

    The sampler plugs directly into
    :class:`~multi_axis_trainer.MultiAxisProjectionTrainer` via
    ``SentenceTransformerTrainingArguments(batch_sampler=MultiAxisNoDuplicatesBatchSampler)``.
    The trainer calls ``get_batch_sampler`` with
    ``valid_label_columns=["{axis}_label", ...]``; this sampler reuses
    that argument as its axis label column list.

    Args:
        dataset: HuggingFace :class:`~datasets.Dataset` with axis label columns.
        batch_size: Number of samples per batch.
        drop_last: Drop the last incomplete batch if ``True``.
        valid_label_columns: List of axis label column names,
            e.g. ``["content_label", "speaker_label"]``.  All must be
            present in the dataset.  If ``None`` or empty, the sampler
            degrades to random batching with a warning.
        generator: Optional :class:`torch.Generator` for reproducibility.
        seed: Seed applied as ``seed + epoch`` at the start of each epoch.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )

        self.axis_label_columns: list[str] = []
        self._axis_labels: list[list] = []

        for col in valid_label_columns or []:
            if col not in dataset.column_names:
                logger.warning(
                    "MultiAxisNoDuplicatesBatchSampler: label column %r not found "
                    "in dataset (columns: %s). Skipping.",
                    col,
                    dataset.column_names,
                )
                continue
            self.axis_label_columns.append(col)
            self._axis_labels.append(dataset[col])

        if not self.axis_label_columns:
            logger.warning(
                "MultiAxisNoDuplicatesBatchSampler: no valid axis label columns "
                "found — batches will not be checked for false negatives."
            )

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        # Dict preserves random order while allowing O(1) removal.
        remaining_indices = dict.fromkeys(
            torch.randperm(len(self.dataset), generator=self.generator).tolist()
        )

        while remaining_indices:
            # One seen-label set per axis for the batch being assembled.
            batch_seen: list[set] = [set() for _ in self.axis_label_columns]
            batch_indices: list[int] = []

            for index in remaining_indices:
                # Reject if this sample's label for any axis is already in the batch.
                if any(
                    self._axis_labels[ax][index] in batch_seen[ax]
                    for ax in range(len(self.axis_label_columns))
                ):
                    continue

                # Admit the sample and record its per-axis labels.
                batch_indices.append(index)
                for ax, labels in enumerate(self._axis_labels):
                    batch_seen[ax].add(labels[index])

                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    break

            else:
                # Exhausted remaining_indices before filling a full batch.
                if not self.drop_last and batch_indices:
                    yield batch_indices

            for index in batch_indices:
                del remaining_indices[index]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
