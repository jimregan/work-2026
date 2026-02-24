from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer

from .projection import MultiAxisProjection


class MultiAxisSentenceTransformer(SentenceTransformer):
    """A SentenceTransformer that exposes per-axis embeddings.

    This subclass adds ``encode_axis``, ``encode_all_axes``, and
    ``similarity_vector`` on top of the standard ``encode()`` API.
    The standard ``encode()`` returns a single vector (concatenated or
    default-axis, depending on the ``MultiAxisProjection`` config), so
    existing evaluators like ``InformationRetrievalEvaluator`` keep
    working unchanged.
    """

    @property
    def axes(self) -> list[str]:
        """Return axis names in declaration order."""
        return self._get_projection_module().axis_names

    @property
    def axis_slices(self) -> dict[str, tuple[int, int]]:
        """Return {axis: (start, end)} byte offsets into the concatenated embedding."""
        return self._get_projection_module().axis_slices

    def _get_projection_module(self) -> MultiAxisProjection:
        for module in self.modules():
            if isinstance(module, MultiAxisProjection):
                return module
        raise ValueError(
            "No MultiAxisProjection module found in the model."
        )

    def encode_axis(
        self,
        sentences: str | list[str],
        axis: str,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | Tensor:
        """Encode sentences and return embeddings for a single axis.

        Args:
            sentences: Input sentence(s).
            axis: Which axis to extract.
            **kwargs: Forwarded to ``encode()``.

        Returns:
            Array of shape ``[N, axis_dim]`` (numpy by default).
        """
        features_list = self.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=None,
            convert_to_numpy=False,
            convert_to_tensor=False,
            device=device,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )
        input_was_string = isinstance(sentences, str)
        if input_was_string:
            features_list = [features_list]

        key = f"embedding_{axis}"
        embeddings = torch.stack([f[key] for f in features_list])

        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if input_was_string:
            embeddings = embeddings[0]

        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    def encode_all_axes(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        device: str | None = None,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Encode sentences and return embeddings for all axes.

        Args:
            sentences: Input sentence(s).
            **kwargs: Forwarded to ``encode()``.

        Returns:
            Dict mapping axis name to ``[N, axis_dim]`` numpy array.
        """
        features_list = self.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=None,
            convert_to_numpy=False,
            convert_to_tensor=False,
            device=device,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )
        input_was_string = isinstance(sentences, str)
        if input_was_string:
            features_list = [features_list]

        result: dict[str, np.ndarray] = {}
        for axis in self.axes:
            key = f"embedding_{axis}"
            stacked = torch.stack([f[key] for f in features_list])
            if normalize_embeddings:
                stacked = torch.nn.functional.normalize(stacked, p=2, dim=1)
            result[axis] = stacked.cpu().numpy()

        return result

    def similarity_vector(
        self,
        a: dict[str, np.ndarray | Tensor],
        b: dict[str, np.ndarray | Tensor],
    ) -> dict[str, Tensor]:
        """Compute per-axis similarity matrices.

        Args:
            a: Dict from ``encode_all_axes()``, axis → ``[N, D]``.
            b: Dict from ``encode_all_axes()``, axis → ``[M, D]``.

        Returns:
            Dict mapping axis name to ``[N, M]`` similarity tensor.
        """
        sim_fn = self.similarity
        result: dict[str, Tensor] = {}
        for axis in self.axes:
            result[axis] = sim_fn(a[axis], b[axis])
        return result
