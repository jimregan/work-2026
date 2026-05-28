from dysfluent_wfst.alignment import (
    AlignmentSegment,
    UtteranceAlignment,
    load_alignment,
    save_alignment,
)


def test_alignment_round_trip(tmp_path):
    path = tmp_path / "alignment.json"
    alignment = UtteranceAlignment(
        utterance_id="utt-1",
        audio_path="audio.wav",
        ref_text="hello world",
        sample_rate=16000,
        frame_shift_ms=17.5,
        ref_phonemes=["h", "e"],
        decoded_phonemes=["h", "e"],
        segments=[
            AlignmentSegment(
                phoneme="h",
                ref_phoneme="h",
                start_frame=3,
                end_frame=5,
                start_time_s=0.0525,
                end_time_s=0.0875,
                variation_type="normal",
                ref_state=0,
                lattice_score=-1.2,
            )
        ],
        variation_info=[{"phoneme": "h", "start_state": 0, "end_state": 1}],
    )

    save_alignment(alignment, str(path))
    loaded = load_alignment(str(path))

    assert loaded.utterance_id == "utt-1"
    assert loaded.frame_shift_ms == 17.5
    assert len(loaded.segments) == 1
    assert loaded.segments[0].phoneme == "h"
    assert loaded.segments[0].start_time_s == 0.0525
