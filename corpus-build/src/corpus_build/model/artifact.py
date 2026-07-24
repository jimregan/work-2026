"""The Artifact type: a generic, versioned unit in the corpus graph.

Recordings, transcripts, alignments, segmentations, metadata records, and
correspondence mappings are all `Artifact` instances distinguished by
`artifact_type` and by the `LayerId` subclass of their `id`. No layer or
media type is privileged in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Mapping, TypeVar

from corpus_build.model.identity import LayerId

IdT = TypeVar("IdT", bound=LayerId)


@dataclass(frozen=True)
class Artifact(Generic[IdT]):
    """A single versioned artifact.

    `content_hash` and `content_locator` are independently optional: an
    artifact can be a declared node in the graph (e.g. the planned output of
    a transformation that has not been run yet) before any content exists,
    and an ephemeral artifact can lose its materialized content entirely
    while remaining fully describable via its provenance.

    `id.local_id` should itself be content-addressed wherever possible (see
    `content_addressing`): `content_address(bytes)` once bytes exist, or
    `locator_address(origin, date)` for an intake node declared ahead of a
    fetch. `content_hash` then records the hash of the bytes currently
    materialized at `content_locator`, which coincides with `id.local_id`
    for a content-addressed artifact but is independently re-derivable as an
    integrity check.

    `persistent` is a stored decision, never derived from graph depth or
    from whether content currently exists.
    """

    id: IdT
    artifact_type: str
    persistent: bool
    content_hash: str | None = None
    content_locator: str | None = None
    acquisition_time: datetime | None = None
    document_time: datetime | None = None
    processing_time: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
