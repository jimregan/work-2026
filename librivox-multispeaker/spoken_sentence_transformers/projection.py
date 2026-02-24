from __future__ import annotations

from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PretrainedConfig

from sentence_transformers.models.Module import Module


class MultiAxisProjectionConfig(PretrainedConfig):
    """HuggingFace-compatible config for :class:`MultiAxisProjection`.

    Storing axes in ``config.json`` makes projection heads loadable via
    ``AutoConfig.from_pretrained`` and pushable to the Hub without any
    custom code on the loading side.
    """

    model_type = "multi_axis_projection"

    def __init__(
        self,
        in_features: int = 0,
        axes: dict[str, int] | list[list] | None = None,
        hidden_dim: int | None = None,
        default_axis: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_features = in_features
        # Store as ordered list of [name, dim] pairs so that
        # PretrainedConfig's sort_keys=True JSON serialization does not
        # reorder the axes.  A JSON array preserves insertion order.
        if isinstance(axes, dict):
            self.axes = list(axes.items())
        else:
            self.axes = axes or []
        self.hidden_dim = hidden_dim
        self.default_axis = default_axis


class MultiAxisProjection(Module):
    """Projects a pooled embedding into multiple named axes.

    Each axis gets its own linear head (or 2-layer MLP if ``hidden_dim``
    is set).  Every axis output is **L2-normalised to unit norm** before
    being stored as ``embedding_{axis_name}`` and before concatenation.
    This gives a factor-partitioned embedding whose subspaces each live
    on a unit hypersphere.

    The module sets ``sentence_embedding`` to either a single requested
    axis, the ``default_axis``, or the concatenation of all axes.

    **Geometry note**: because each block is unit-normalised, cosine
    similarity over the full concatenated vector equals the unweighted
    mean of per-axis cosines (full vector norm = √A where A = number of
    axes).  Axis *dimensionality* controls representational capacity, not
    similarity weight.  For weighted retrieval, use
    :meth:`~MultiAxisSentenceTransformer.encode_all_axes` and combine
    per-axis cosines explicitly.

    Args:
        in_features: Dimension of the incoming ``sentence_embedding``.
        axes: Mapping of axis name to output dimension,
            e.g. ``{"semantic": 256, "speaker_id": 256}``.
            Insertion order is preserved through save/load.
        hidden_dim: If set, use a 2-layer MLP (Linear → ReLU → Linear)
            per axis instead of a single Linear.
        default_axis: Axis whose projection becomes the default
            ``sentence_embedding``.  If ``None``, all axis projections
            are concatenated.
    """

    config_keys: list[str] = [
        "in_features",
        "axes",
        "hidden_dim",
        "default_axis",
    ]

    forward_kwargs: set[str] = {"axis"}

    def __init__(
        self,
        in_features: int,
        axes: dict[str, int],
        hidden_dim: int | None = None,
        default_axis: str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        # Preserve user-specified axis order.  Accept both dict and list-of-pairs
        # (the latter is returned by MultiAxisProjectionConfig on load).
        self.axes: dict[str, int] = dict(axes) if not isinstance(axes, dict) else axes
        self.hidden_dim = hidden_dim
        self.default_axis = default_axis

        if default_axis is not None and default_axis not in axes:
            raise ValueError(
                f"default_axis {default_axis!r} not in axes {list(axes)}"
            )

        self.heads = nn.ModuleDict()
        for name, out_dim in self.axes.items():
            if hidden_dim is not None:
                self.heads[name] = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                )
            else:
                self.heads[name] = nn.Linear(in_features, out_dim)

    def forward(
        self,
        features: dict[str, Tensor | Any],
        axis: str | None = None,
        **kwargs,
    ) -> dict[str, Tensor | Any]:
        embedding = features["sentence_embedding"]

        projections: dict[str, Tensor] = {}
        for name, head in self.heads.items():
            proj = F.normalize(head(embedding), p=2, dim=-1)
            projections[name] = proj
            features[f"embedding_{name}"] = proj

        if axis is not None:
            if axis not in projections:
                raise ValueError(
                    f"Unknown axis {axis!r}. Available: {list(self.axes)}"
                )
            features["sentence_embedding"] = projections[axis]
        elif self.default_axis is not None:
            features["sentence_embedding"] = projections[self.default_axis]
        else:
            features["sentence_embedding"] = torch.cat(
                list(projections.values()), dim=-1
            )

        return features

    def get_sentence_embedding_dimension(self) -> int:
        if self.default_axis is not None:
            return self.axes[self.default_axis]
        return sum(self.axes.values())

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "axes": self.axes,
            "hidden_dim": self.hidden_dim,
            "default_axis": self.default_axis,
        }

    def save(
        self,
        output_path: str,
        *args,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        config = MultiAxisProjectionConfig(
            in_features=self.in_features,
            axes=self.axes,
            hidden_dim=self.hidden_dim,
            default_axis=self.default_axis,
        )
        config.save_pretrained(output_path)
        self.save_torch_weights(
            output_path, safe_serialization=safe_serialization
        )

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        config = MultiAxisProjectionConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_dir=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        # config.axes is a list of [name, dim] pairs; dict() preserves order.
        axes = dict(config.axes) if not isinstance(config.axes, dict) else config.axes
        model = cls(
            in_features=config.in_features,
            axes=axes,
            hidden_dim=config.hidden_dim,
            default_axis=config.default_axis,
        )
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model = cls.load_torch_weights(
            model_name_or_path=model_name_or_path, model=model, **hub_kwargs
        )
        return model

    def __repr__(self) -> str:
        return f"MultiAxisProjection({self.get_config_dict()})"
