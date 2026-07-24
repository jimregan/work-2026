"""Layer-specific artifact identity.

There is no global "same thing" relation between artifacts. Two objects that
plausibly refer to the same real-world event (two radio captures of one
broadcast, an ASR transcript and the audio it was produced from) are only
comparable through an explicit correspondence artifact, never through shared
identity. Each layer therefore gets its own `LayerId` subclass, and instances
of different subclasses are never equal even if `local_id` matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LayerId:
    """Base class for a layer-scoped local identifier.

    Not meant to be instantiated directly; each layer defines a subclass
    that sets `layer`. New layers are added by subclassing and registering
    with `register_layer_id_class`, not by editing a closed set here.
    """

    local_id: str
    layer: ClassVar[str]


_LAYER_ID_CLASSES: dict[str, type[LayerId]] = {}


def register_layer_id_class(cls: type[LayerId]) -> type[LayerId]:
    """Class decorator: make a LayerId subclass resolvable by its `layer`
    name, e.g. for deserializing storage records back into the right type."""
    _LAYER_ID_CLASSES[cls.layer] = cls
    return cls


def resolve_layer_id_class(layer: str) -> type[LayerId]:
    try:
        return _LAYER_ID_CLASSES[layer]
    except KeyError as exc:
        raise KeyError(f"no LayerId subclass registered for layer {layer!r}") from exc


@register_layer_id_class
class AcquisitionId(LayerId):
    layer: ClassVar[str] = "acquisition"


@register_layer_id_class
class TranscriptId(LayerId):
    layer: ClassVar[str] = "transcript"


@register_layer_id_class
class AlignmentId(LayerId):
    layer: ClassVar[str] = "alignment"


@register_layer_id_class
class SegmentationId(LayerId):
    layer: ClassVar[str] = "segmentation"


@register_layer_id_class
class CorrespondenceId(LayerId):
    layer: ClassVar[str] = "correspondence"


@register_layer_id_class
class MetadataId(LayerId):
    layer: ClassVar[str] = "metadata"


@dataclass(frozen=True)
class TransformationId:
    """Identifier for a transformation record. Not layer-scoped: a
    transformation is not itself an artifact and does not participate in
    layer identity."""

    value: str
