from corpus_build.model.identity import AcquisitionId, TranscriptId


def test_same_local_id_different_layer_is_not_equal() -> None:
    acquisition = AcquisitionId(local_id="x")
    transcript = TranscriptId(local_id="x")
    assert acquisition != transcript
    assert len({acquisition, transcript}) == 2


def test_same_layer_same_local_id_is_equal() -> None:
    assert AcquisitionId(local_id="x") == AcquisitionId(local_id="x")


def test_layer_class_var_matches_registered_name() -> None:
    assert AcquisitionId.layer == "acquisition"
    assert TranscriptId.layer == "transcript"
