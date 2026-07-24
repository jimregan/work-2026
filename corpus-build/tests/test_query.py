from __future__ import annotations

from pathlib import Path

from conftest import artifact, transformation
from corpus_build.model.identity import AcquisitionId, TranscriptId
from corpus_build.query import provenance_of
from corpus_build.storage.filesystem import FilesystemBackend


def test_provenance_of_walks_full_ancestry(tmp_path: Path) -> None:
    backend = FilesystemBackend(tmp_path)
    backend.put_artifact(artifact(AcquisitionId("rec-1")))
    backend.put_artifact(artifact(TranscriptId("tx-1")))
    t1 = transformation("t1", inputs=(AcquisitionId("rec-1"),), outputs=(TranscriptId("tx-1"),))
    backend.put_transformation(t1)

    graph = provenance_of(backend, TranscriptId("tx-1"))

    assert graph.root == TranscriptId("tx-1")
    assert AcquisitionId("rec-1") in graph.artifacts
    assert t1 in graph.transformations


def test_source_artifact_has_no_transformations(tmp_path: Path) -> None:
    backend = FilesystemBackend(tmp_path)
    backend.put_artifact(artifact(AcquisitionId("rec-1")))

    graph = provenance_of(backend, AcquisitionId("rec-1"))

    assert graph.transformations == ()
    assert graph.artifacts == (AcquisitionId("rec-1"),)


def test_ephemeral_artifact_provenance_survives_content_loss(tmp_path: Path) -> None:
    """Persistence is a property, not a position: an ephemeral artifact can
    lose its materialized content while its provenance stays fully
    answerable from the graph."""
    backend = FilesystemBackend(tmp_path)
    backend.put_artifact(artifact(AcquisitionId("rec-1")))
    ephemeral = artifact(
        TranscriptId("tx-ephemeral"),
        persistent=False,
        content_hash=None,
        content_locator=None,
    )
    backend.put_artifact(ephemeral)
    t1 = transformation("t1", inputs=(AcquisitionId("rec-1"),), outputs=(TranscriptId("tx-ephemeral"),))
    backend.put_transformation(t1)

    graph = provenance_of(backend, TranscriptId("tx-ephemeral"))

    assert t1 in graph.transformations
    assert AcquisitionId("rec-1") in graph.artifacts
    stored = backend.get_artifact(TranscriptId("tx-ephemeral"))
    assert stored.content_hash is None
    assert stored.persistent is False


def test_persistence_is_independent_of_graph_depth(tmp_path: Path) -> None:
    """A deep artifact may be persistent and a shallow one ephemeral;
    nothing infers persistence from position in the graph."""
    backend = FilesystemBackend(tmp_path)
    backend.put_artifact(artifact(AcquisitionId("rec-shallow"), persistent=False))
    backend.put_artifact(artifact(TranscriptId("tx-deep"), persistent=True))
    t1 = transformation("t1", inputs=(AcquisitionId("rec-shallow"),), outputs=(TranscriptId("tx-deep"),))
    backend.put_transformation(t1)

    assert backend.get_artifact(AcquisitionId("rec-shallow")).persistent is False
    assert backend.get_artifact(TranscriptId("tx-deep")).persistent is True
