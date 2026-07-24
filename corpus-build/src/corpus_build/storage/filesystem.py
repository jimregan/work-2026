"""Filesystem-conventions storage backend.

One JSON sidecar per record, laid out by convention:

    <root>/artifacts/<layer>/<local_id>.json
    <root>/transformations/<transformation_id>.json

This backend never stores content bytes itself; `content_locator` on an
artifact is just a string it round-trips, conventionally a path elsewhere on
disk.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from corpus_build.model.artifact import Artifact
from corpus_build.model.identity import LayerId, TransformationId
from corpus_build.model.transformation import Transformation
from corpus_build.storage.base import (
    ArtifactNotFoundError,
    DuplicateArtifactError,
    DuplicateTransformationError,
    StorageBackend,
    TransformationNotFoundError,
)
from corpus_build.storage.serialization import (
    artifact_from_dict,
    artifact_to_dict,
    transformation_from_dict,
    transformation_to_dict,
)


class FilesystemBackend(StorageBackend):
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._artifacts_dir = self.root / "artifacts"
        self._transformations_dir = self.root / "transformations"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._transformations_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: LayerId) -> Path:
        return self._artifacts_dir / artifact_id.layer / f"{artifact_id.local_id}.json"

    def _transformation_path(self, transformation_id: TransformationId) -> Path:
        return self._transformations_dir / f"{transformation_id.value}.json"

    def put_artifact(self, artifact: Artifact[Any]) -> None:
        path = self._artifact_path(artifact.id)
        if path.exists():
            raise DuplicateArtifactError(artifact.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact_to_dict(artifact), indent=2))

    def get_artifact(self, artifact_id: LayerId) -> Artifact[Any]:
        path = self._artifact_path(artifact_id)
        if not path.exists():
            raise ArtifactNotFoundError(artifact_id)
        return artifact_from_dict(json.loads(path.read_text()))

    def list_artifacts(self, layer: str | None = None) -> Iterator[Artifact[Any]]:
        if layer is not None:
            layer_dirs = [self._artifacts_dir / layer]
        else:
            layer_dirs = sorted(p for p in self._artifacts_dir.iterdir() if p.is_dir())
        for layer_dir in layer_dirs:
            if not layer_dir.is_dir():
                continue
            for path in sorted(layer_dir.glob("*.json")):
                yield artifact_from_dict(json.loads(path.read_text()))

    def put_transformation(self, transformation: Transformation) -> None:
        path = self._transformation_path(transformation.id)
        if path.exists():
            raise DuplicateTransformationError(transformation.id)
        self._check_acyclic_with(transformation)
        path.write_text(json.dumps(transformation_to_dict(transformation), indent=2))

    def get_transformation(self, transformation_id: TransformationId) -> Transformation:
        path = self._transformation_path(transformation_id)
        if not path.exists():
            raise TransformationNotFoundError(transformation_id)
        return transformation_from_dict(json.loads(path.read_text()))

    def all_transformations(self) -> Iterator[Transformation]:
        for path in sorted(self._transformations_dir.glob("*.json")):
            yield transformation_from_dict(json.loads(path.read_text()))
