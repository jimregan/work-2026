"""Transformation records: first-class objects, not functions.

A `Transformation` carries everything needed to know how an artifact came to
be, including the case where it has not been executed yet: `executed_at` and
`execution_metadata` describe the run, everything else describes the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from corpus_build.model.identity import LayerId, TransformationId


@dataclass(frozen=True)
class Transformation:
    """A record of (or plan for) producing `output_ids` from `input_ids`.

    `input_ids` and `output_ids` are heterogeneous: a transformation
    routinely consumes artifacts from more than one layer (e.g. a
    transcript and an audio recording) to produce one of a third (e.g. a
    correspondence). Multi-parent is the default shape, not a special case.

    `implementation_ref` identifies the source (e.g. a git commit);
    `container_image` identifies the executable behaviour and should prefer
    an immutable digest over a mutable tag. Both may be known before the
    transformation has run.
    """

    id: TransformationId
    implementation_ref: str
    container_image: str | None
    parameters: Mapping[str, Any]
    input_ids: tuple[LayerId, ...]
    output_ids: tuple[LayerId, ...]
    executed_at: datetime | None = None
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_run(self) -> bool:
        return self.executed_at is not None
