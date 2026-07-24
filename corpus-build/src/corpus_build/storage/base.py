"""The storage backend contract.

The core model must not import any specific backend. A backend stores
`Artifact` and `Transformation` records; it does not execute transformations
and does not assume anything about where content bytes live beyond a
`content_locator` string on the artifact (typically a filesystem path).

Artifacts and transformations are append-only records: `put_artifact` and
`put_transformation` reject an id that already exists rather than silently
overwriting provenance history. A new version of a "thing" is a new
artifact with a new id and a transformation linking it back, not a mutation
of the old record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from corpus_build.model.artifact import Artifact
from corpus_build.model.identity import LayerId, TransformationId
from corpus_build.model.provenance import assert_acyclic
from corpus_build.model.transformation import Transformation


class ArtifactNotFoundError(KeyError):
    def __init__(self, artifact_id: LayerId) -> None:
        self.artifact_id = artifact_id
        super().__init__(f"no artifact {artifact_id.layer}:{artifact_id.local_id}")


class DuplicateArtifactError(ValueError):
    def __init__(self, artifact_id: LayerId) -> None:
        self.artifact_id = artifact_id
        super().__init__(f"artifact {artifact_id.layer}:{artifact_id.local_id} already exists")


class TransformationNotFoundError(KeyError):
    def __init__(self, transformation_id: TransformationId) -> None:
        self.transformation_id = transformation_id
        super().__init__(f"no transformation {transformation_id.value}")


class DuplicateTransformationError(ValueError):
    def __init__(self, transformation_id: TransformationId) -> None:
        self.transformation_id = transformation_id
        super().__init__(f"transformation {transformation_id.value} already exists")


class StorageBackend(ABC):
    """Backend contract. Concrete backends run against the shared
    conformance suite in `tests/storage_conformance.py`."""

    @abstractmethod
    def put_artifact(self, artifact: Artifact[Any]) -> None:
        """Store a new artifact. Raises DuplicateArtifactError if its id is
        already present."""

    @abstractmethod
    def get_artifact(self, artifact_id: LayerId) -> Artifact[Any]:
        """Raises ArtifactNotFoundError if absent."""

    @abstractmethod
    def list_artifacts(self, layer: str | None = None) -> Iterator[Artifact[Any]]:
        """All artifacts, optionally restricted to one layer name."""

    @abstractmethod
    def put_transformation(self, transformation: Transformation) -> None:
        """Store a new transformation. Raises DuplicateTransformationError
        if its id is already present, or CycleError if accepting it would
        make an artifact its own ancestor."""

    @abstractmethod
    def get_transformation(self, transformation_id: TransformationId) -> Transformation:
        """Raises TransformationNotFoundError if absent."""

    @abstractmethod
    def all_transformations(self) -> Iterator[Transformation]: ...

    def transformations_producing(self, artifact_id: LayerId) -> Iterator[Transformation]:
        """Transformations whose output_ids include artifact_id. Default
        implementation is a linear scan; backends may override with an
        indexed query."""
        for transformation in self.all_transformations():
            if artifact_id in transformation.output_ids:
                yield transformation

    def transformations_consuming(self, artifact_id: LayerId) -> Iterator[Transformation]:
        """Transformations whose input_ids include artifact_id."""
        for transformation in self.all_transformations():
            if artifact_id in transformation.input_ids:
                yield transformation

    def _check_acyclic_with(self, candidate: Transformation) -> None:
        """Call from put_transformation before persisting `candidate`."""
        assert_acyclic([*self.all_transformations(), candidate])
