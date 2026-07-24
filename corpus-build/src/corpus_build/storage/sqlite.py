"""SQLite storage backend.

Structured metadata only: artifacts, transformations, and the input/output
edges between them. `content_locator` is stored as plain TEXT — a path to a
real file elsewhere on disk. Content bytes are never stored as a BLOB column;
software that wants to read an artifact's content should be able to open it
as an ordinary file, not need this library to extract it from a database.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from corpus_build.model.artifact import Artifact
from corpus_build.model.identity import LayerId
from corpus_build.model.transformation import Transformation, TransformationId
from corpus_build.storage.base import (
    ArtifactNotFoundError,
    DuplicateArtifactError,
    DuplicateTransformationError,
    StorageBackend,
    TransformationNotFoundError,
)
from corpus_build.storage.serialization import artifact_from_dict, artifact_to_dict, transformation_from_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    layer TEXT NOT NULL,
    local_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    persistent INTEGER NOT NULL,
    content_hash TEXT,
    content_locator TEXT,
    acquisition_time TEXT,
    document_time TEXT,
    processing_time TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (layer, local_id)
);

CREATE TABLE IF NOT EXISTS transformations (
    id TEXT PRIMARY KEY,
    implementation_ref TEXT NOT NULL,
    container_image TEXT,
    parameters_json TEXT NOT NULL,
    executed_at TEXT,
    execution_metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transformation_inputs (
    transformation_id TEXT NOT NULL REFERENCES transformations(id),
    position INTEGER NOT NULL,
    layer TEXT NOT NULL,
    local_id TEXT NOT NULL,
    PRIMARY KEY (transformation_id, position)
);

CREATE TABLE IF NOT EXISTS transformation_outputs (
    transformation_id TEXT NOT NULL REFERENCES transformations(id),
    position INTEGER NOT NULL,
    layer TEXT NOT NULL,
    local_id TEXT NOT NULL,
    PRIMARY KEY (transformation_id, position)
);
"""

_ARTIFACT_SELECT = (
    "SELECT layer, local_id, artifact_type, persistent, content_hash, content_locator, "
    "acquisition_time, document_time, processing_time, metadata_json FROM artifacts"
)

_TRANSFORMATION_SELECT = (
    "SELECT id, implementation_ref, container_image, parameters_json, executed_at, "
    "execution_metadata_json FROM transformations"
)


def _artifact_row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    (
        layer,
        local_id,
        artifact_type,
        persistent,
        content_hash,
        content_locator,
        acquisition_time,
        document_time,
        processing_time,
        metadata_json,
    ) = row
    return {
        "layer": layer,
        "local_id": local_id,
        "artifact_type": artifact_type,
        "persistent": bool(persistent),
        "content_hash": content_hash,
        "content_locator": content_locator,
        "acquisition_time": acquisition_time,
        "document_time": document_time,
        "processing_time": processing_time,
        "metadata": json.loads(metadata_json),
    }


class SQLiteBackend(StorageBackend):
    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def put_artifact(self, artifact: Artifact[Any]) -> None:
        exists = self._conn.execute(
            "SELECT 1 FROM artifacts WHERE layer = ? AND local_id = ?",
            (artifact.id.layer, artifact.id.local_id),
        ).fetchone()
        if exists is not None:
            raise DuplicateArtifactError(artifact.id)
        data = artifact_to_dict(artifact)
        self._conn.execute(
            "INSERT INTO artifacts (layer, local_id, artifact_type, persistent, content_hash, "
            "content_locator, acquisition_time, document_time, processing_time, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                data["layer"],
                data["local_id"],
                data["artifact_type"],
                int(data["persistent"]),
                data["content_hash"],
                data["content_locator"],
                data["acquisition_time"],
                data["document_time"],
                data["processing_time"],
                json.dumps(data["metadata"]),
            ),
        )
        self._conn.commit()

    def get_artifact(self, artifact_id: LayerId) -> Artifact[Any]:
        row = self._conn.execute(
            f"{_ARTIFACT_SELECT} WHERE layer = ? AND local_id = ?",
            (artifact_id.layer, artifact_id.local_id),
        ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(artifact_id)
        return artifact_from_dict(_artifact_row_to_dict(row))

    def list_artifacts(self, layer: str | None = None) -> Iterator[Artifact[Any]]:
        if layer is not None:
            rows = self._conn.execute(
                f"{_ARTIFACT_SELECT} WHERE layer = ? ORDER BY local_id", (layer,)
            ).fetchall()
        else:
            rows = self._conn.execute(f"{_ARTIFACT_SELECT} ORDER BY layer, local_id").fetchall()
        for row in rows:
            yield artifact_from_dict(_artifact_row_to_dict(row))

    def put_transformation(self, transformation: Transformation) -> None:
        exists = self._conn.execute(
            "SELECT 1 FROM transformations WHERE id = ?", (transformation.id.value,)
        ).fetchone()
        if exists is not None:
            raise DuplicateTransformationError(transformation.id)
        self._check_acyclic_with(transformation)

        self._conn.execute(
            "INSERT INTO transformations (id, implementation_ref, container_image, "
            "parameters_json, executed_at, execution_metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                transformation.id.value,
                transformation.implementation_ref,
                transformation.container_image,
                json.dumps(dict(transformation.parameters)),
                transformation.executed_at.isoformat() if transformation.executed_at else None,
                json.dumps(dict(transformation.execution_metadata)),
            ),
        )
        for position, input_id in enumerate(transformation.input_ids):
            self._conn.execute(
                "INSERT INTO transformation_inputs (transformation_id, position, layer, local_id) "
                "VALUES (?, ?, ?, ?)",
                (transformation.id.value, position, input_id.layer, input_id.local_id),
            )
        for position, output_id in enumerate(transformation.output_ids):
            self._conn.execute(
                "INSERT INTO transformation_outputs (transformation_id, position, layer, local_id) "
                "VALUES (?, ?, ?, ?)",
                (transformation.id.value, position, output_id.layer, output_id.local_id),
            )
        self._conn.commit()

    def get_transformation(self, transformation_id: TransformationId) -> Transformation:
        row = self._conn.execute(
            f"{_TRANSFORMATION_SELECT} WHERE id = ?", (transformation_id.value,)
        ).fetchone()
        if row is None:
            raise TransformationNotFoundError(transformation_id)
        return self._build_transformation(row)

    def all_transformations(self) -> Iterator[Transformation]:
        rows = self._conn.execute(f"{_TRANSFORMATION_SELECT} ORDER BY id").fetchall()
        for row in rows:
            yield self._build_transformation(row)

    def _build_transformation(self, row: tuple[Any, ...]) -> Transformation:
        (tid, implementation_ref, container_image, parameters_json, executed_at, execution_metadata_json) = row
        inputs = self._conn.execute(
            "SELECT layer, local_id FROM transformation_inputs WHERE transformation_id = ? ORDER BY position",
            (tid,),
        ).fetchall()
        outputs = self._conn.execute(
            "SELECT layer, local_id FROM transformation_outputs WHERE transformation_id = ? ORDER BY position",
            (tid,),
        ).fetchall()
        data = {
            "id": tid,
            "implementation_ref": implementation_ref,
            "container_image": container_image,
            "parameters": json.loads(parameters_json),
            "input_ids": [{"layer": layer, "local_id": local_id} for (layer, local_id) in inputs],
            "output_ids": [{"layer": layer, "local_id": local_id} for (layer, local_id) in outputs],
            "executed_at": executed_at,
            "execution_metadata": json.loads(execution_metadata_json),
        }
        return transformation_from_dict(data)
