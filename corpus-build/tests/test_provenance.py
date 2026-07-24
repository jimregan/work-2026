import pytest

from conftest import transformation
from corpus_build.model.identity import AcquisitionId, AlignmentId, CorrespondenceId, TranscriptId
from corpus_build.model.provenance import CycleError, assert_acyclic, find_cycle


def test_acyclic_graph_passes() -> None:
    t1 = transformation("t1", inputs=(AcquisitionId("rec-1"),), outputs=(TranscriptId("tx-1"),))
    assert_acyclic([t1])
    assert find_cycle([t1]) is None


def test_multi_parent_is_the_normal_case() -> None:
    t1 = transformation(
        "align-corr",
        inputs=(TranscriptId("tx-1"), AlignmentId("al-1")),
        outputs=(CorrespondenceId("corr-1"),),
    )
    assert_acyclic([t1])
    assert len(t1.input_ids) == 2
    assert {type(i) for i in t1.input_ids} == {TranscriptId, AlignmentId}


def test_cycle_is_detected() -> None:
    t1 = transformation("t1", inputs=(AcquisitionId("a"),), outputs=(TranscriptId("b"),))
    t2 = transformation("t2", inputs=(TranscriptId("b"),), outputs=(AcquisitionId("a"),))
    cycle = find_cycle([t1, t2])
    assert cycle is not None
    with pytest.raises(CycleError):
        assert_acyclic([t1, t2])


def test_unrelated_transformations_do_not_form_a_false_cycle() -> None:
    t1 = transformation("t1", inputs=(AcquisitionId("a"),), outputs=(TranscriptId("b"),))
    t2 = transformation("t2", inputs=(AcquisitionId("c"),), outputs=(TranscriptId("d"),))
    assert_acyclic([t1, t2])
