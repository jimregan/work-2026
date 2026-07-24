"""Shared behavioural contract every StorageBackend must satisfy.

Not collected by pytest directly (no test_ prefix): subclass
StorageConformanceSuite in a test_storage_<backend>.py module and provide a
`backend` fixture. Any behaviour that passes for one backend and not another
is a leak in the StorageBackend interface, not something to special-case.
"""

from __future__ import annotations

import pytest

from conftest import artifact, transformation
from corpus_build.model.identity import AcquisitionId, AlignmentId, CorrespondenceId, TranscriptId, TransformationId
from corpus_build.model.provenance import CycleError
from corpus_build.storage.base import (
    ArtifactNotFoundError,
    DuplicateArtifactError,
    DuplicateTransformationError,
    StorageBackend,
    TransformationNotFoundError,
)


class StorageConformanceSuite:
    @pytest.fixture
    def backend(self) -> StorageBackend:
        raise NotImplementedError("subclasses must override the backend fixture")

    def test_put_and_get_artifact_roundtrips(self, backend: StorageBackend) -> None:
        a = artifact(AcquisitionId("rec-1"), content_hash="deadbeef")
        backend.put_artifact(a)
        assert backend.get_artifact(AcquisitionId("rec-1")) == a

    def test_get_missing_artifact_raises(self, backend: StorageBackend) -> None:
        with pytest.raises(ArtifactNotFoundError):
            backend.get_artifact(AcquisitionId("missing"))

    def test_duplicate_artifact_raises(self, backend: StorageBackend) -> None:
        a = artifact(AcquisitionId("rec-1"))
        backend.put_artifact(a)
        with pytest.raises(DuplicateArtifactError):
            backend.put_artifact(a)

    def test_list_artifacts_filters_by_layer(self, backend: StorageBackend) -> None:
        backend.put_artifact(artifact(AcquisitionId("rec-1")))
        backend.put_artifact(artifact(TranscriptId("tx-1")))
        acquisitions = list(backend.list_artifacts(layer="acquisition"))
        assert [a.id for a in acquisitions] == [AcquisitionId("rec-1")]
        assert len(list(backend.list_artifacts())) == 2

    def test_put_and_get_transformation_roundtrips(self, backend: StorageBackend) -> None:
        backend.put_artifact(artifact(AcquisitionId("rec-1")))
        backend.put_artifact(artifact(TranscriptId("tx-1")))
        t1 = transformation("t1", inputs=(AcquisitionId("rec-1"),), outputs=(TranscriptId("tx-1"),))
        backend.put_transformation(t1)
        assert backend.get_transformation(t1.id) == t1

    def test_get_missing_transformation_raises(self, backend: StorageBackend) -> None:
        with pytest.raises(TransformationNotFoundError):
            backend.get_transformation(TransformationId("missing"))

    def test_duplicate_transformation_raises(self, backend: StorageBackend) -> None:
        backend.put_artifact(artifact(AcquisitionId("rec-1")))
        backend.put_artifact(artifact(TranscriptId("tx-1")))
        t1 = transformation("t1", inputs=(AcquisitionId("rec-1"),), outputs=(TranscriptId("tx-1"),))
        backend.put_transformation(t1)
        with pytest.raises(DuplicateTransformationError):
            backend.put_transformation(t1)

    def test_transformation_with_multiple_parents_roundtrips(self, backend: StorageBackend) -> None:
        backend.put_artifact(artifact(TranscriptId("tx-1")))
        backend.put_artifact(artifact(AlignmentId("al-1")))
        backend.put_artifact(artifact(CorrespondenceId("corr-1")))
        t1 = transformation(
            "align-corr",
            inputs=(TranscriptId("tx-1"), AlignmentId("al-1")),
            outputs=(CorrespondenceId("corr-1"),),
        )
        backend.put_transformation(t1)
        fetched = backend.get_transformation(t1.id)
        assert fetched.input_ids == (TranscriptId("tx-1"), AlignmentId("al-1"))

    def test_transformations_producing_and_consuming(self, backend: StorageBackend) -> None:
        backend.put_artifact(artifact(AcquisitionId("rec-1")))
        backend.put_artifact(artifact(TranscriptId("tx-1")))
        t1 = transformation("t1", inputs=(AcquisitionId("rec-1"),), outputs=(TranscriptId("tx-1"),))
        backend.put_transformation(t1)

        assert list(backend.transformations_producing(TranscriptId("tx-1"))) == [t1]
        assert list(backend.transformations_consuming(AcquisitionId("rec-1"))) == [t1]
        assert list(backend.transformations_producing(AcquisitionId("rec-1"))) == []

    def test_put_transformation_rejects_cycle(self, backend: StorageBackend) -> None:
        backend.put_artifact(artifact(AcquisitionId("a")))
        backend.put_artifact(artifact(TranscriptId("b")))
        t1 = transformation("t1", inputs=(AcquisitionId("a"),), outputs=(TranscriptId("b"),))
        backend.put_transformation(t1)
        t2 = transformation("t2", inputs=(TranscriptId("b"),), outputs=(AcquisitionId("a"),))
        with pytest.raises(CycleError):
            backend.put_transformation(t2)
