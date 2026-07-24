from __future__ import annotations

from datetime import datetime
from typing import Any

from corpus_build.model.artifact import Artifact
from corpus_build.model.identity import LayerId
from corpus_build.model.transformation import Transformation, TransformationId


def artifact(id_: LayerId, artifact_type: str = "test/artifact", persistent: bool = True, **overrides: Any) -> Artifact[Any]:
    fields: dict[str, Any] = {"id": id_, "artifact_type": artifact_type, "persistent": persistent}
    fields.update(overrides)
    return Artifact(**fields)


def transformation(
    id_: str,
    inputs: tuple[LayerId, ...],
    outputs: tuple[LayerId, ...],
    implementation_ref: str = "git:deadbeef",
    container_image: str | None = "sha256:example",
    parameters: dict[str, Any] | None = None,
    executed_at: datetime | None = None,
) -> Transformation:
    return Transformation(
        id=TransformationId(id_),
        implementation_ref=implementation_ref,
        container_image=container_image,
        parameters=parameters or {},
        input_ids=inputs,
        output_ids=outputs,
        executed_at=executed_at,
    )
