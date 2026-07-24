"""Plain-dict encoding of model objects, shared by every backend.

Kept separate from the backends themselves so the filesystem and SQLite
implementations encode/decode records identically; only where the dicts are
put (files vs. rows) differs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from corpus_build.model.artifact import Artifact
from corpus_build.model.identity import LayerId, TransformationId, resolve_layer_id_class
from corpus_build.model.transformation import Transformation


def _dt_to_str(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dt_from_str(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def layer_id_to_dict(layer_id: LayerId) -> dict[str, str]:
    return {"layer": layer_id.layer, "local_id": layer_id.local_id}


def layer_id_from_dict(data: dict[str, str]) -> LayerId:
    cls = resolve_layer_id_class(data["layer"])
    return cls(local_id=data["local_id"])


def artifact_to_dict(artifact: Artifact[Any]) -> dict[str, Any]:
    return {
        "layer": artifact.id.layer,
        "local_id": artifact.id.local_id,
        "artifact_type": artifact.artifact_type,
        "persistent": artifact.persistent,
        "content_hash": artifact.content_hash,
        "content_locator": artifact.content_locator,
        "acquisition_time": _dt_to_str(artifact.acquisition_time),
        "document_time": _dt_to_str(artifact.document_time),
        "processing_time": _dt_to_str(artifact.processing_time),
        "metadata": dict(artifact.metadata),
    }


def artifact_from_dict(data: dict[str, Any]) -> Artifact[Any]:
    cls = resolve_layer_id_class(data["layer"])
    return Artifact(
        id=cls(local_id=data["local_id"]),
        artifact_type=data["artifact_type"],
        persistent=data["persistent"],
        content_hash=data.get("content_hash"),
        content_locator=data.get("content_locator"),
        acquisition_time=_dt_from_str(data.get("acquisition_time")),
        document_time=_dt_from_str(data.get("document_time")),
        processing_time=_dt_from_str(data.get("processing_time")),
        metadata=data.get("metadata") or {},
    )


def transformation_to_dict(transformation: Transformation) -> dict[str, Any]:
    return {
        "id": transformation.id.value,
        "implementation_ref": transformation.implementation_ref,
        "container_image": transformation.container_image,
        "parameters": dict(transformation.parameters),
        "input_ids": [layer_id_to_dict(i) for i in transformation.input_ids],
        "output_ids": [layer_id_to_dict(o) for o in transformation.output_ids],
        "executed_at": _dt_to_str(transformation.executed_at),
        "execution_metadata": dict(transformation.execution_metadata),
    }


def transformation_from_dict(data: dict[str, Any]) -> Transformation:
    return Transformation(
        id=TransformationId(data["id"]),
        implementation_ref=data["implementation_ref"],
        container_image=data.get("container_image"),
        parameters=data.get("parameters") or {},
        input_ids=tuple(layer_id_from_dict(i) for i in data["input_ids"]),
        output_ids=tuple(layer_id_from_dict(o) for o in data["output_ids"]),
        executed_at=_dt_from_str(data.get("executed_at")),
        execution_metadata=data.get("execution_metadata") or {},
    )
