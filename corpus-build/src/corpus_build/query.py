"""The query surface: answering "how was this artifact produced".

Deliberately thin. Provenance is derived from `Transformation` records
already in a `StorageBackend`; this module just walks that graph.
"""

from __future__ import annotations

from corpus_build.model.identity import LayerId, TransformationId
from corpus_build.model.provenance import ProvenanceGraph
from corpus_build.model.transformation import Transformation
from corpus_build.storage.base import StorageBackend


def provenance_of(backend: StorageBackend, artifact_id: LayerId) -> ProvenanceGraph:
    """The transitive ancestry of `artifact_id`: every transformation and
    every artifact that contributed to it, directly or indirectly.

    An artifact with no producing transformation is a source node — its
    provenance graph is itself with no transformations.
    """
    seen_artifacts: set[LayerId] = {artifact_id}
    seen_transformations: dict[TransformationId, Transformation] = {}
    frontier: list[LayerId] = [artifact_id]

    while frontier:
        current = frontier.pop()
        for transformation in backend.transformations_producing(current):
            if transformation.id in seen_transformations:
                continue
            seen_transformations[transformation.id] = transformation
            for parent in transformation.input_ids:
                if parent not in seen_artifacts:
                    seen_artifacts.add(parent)
                    frontier.append(parent)

    return ProvenanceGraph(
        root=artifact_id,
        transformations=tuple(seen_transformations.values()),
        artifacts=tuple(seen_artifacts),
    )
