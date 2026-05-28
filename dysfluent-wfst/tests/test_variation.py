from dysfluent_wfst.variation import (
    build_state_trajectory,
    collapse_label_runs,
    merge_trans_markers,
)


class FakeSymbolTable:
    def __init__(self, mapping):
        self.mapping = mapping

    def find(self, idx):
        return self.mapping.get(idx, "")


def test_collapse_label_runs_preserves_frame_spans():
    syms = FakeSymbolTable(
        {
            1: "a",
            2: "0<trans>2",
            3: "b",
            4: "<pad>",
        }
    )

    runs = collapse_label_runs([1, 1, 2, 2, 3, 3, 4], syms)

    assert runs == [
        {"symbol": "a", "start_frame": 0, "end_frame": 2},
        {"symbol": "0<trans>2", "start_frame": 2, "end_frame": 4},
        {"symbol": "b", "start_frame": 4, "end_frame": 6},
    ]


def test_merge_and_classify_substitution_and_deletion():
    merged = merge_trans_markers(
        [
            {"symbol": "0<trans>2", "start_frame": 2, "end_frame": 3},
            {"symbol": "2<trans>3", "start_frame": 3, "end_frame": 4},
            {"symbol": "z", "start_frame": 4, "end_frame": 7},
        ]
    )

    assert merged == [
        {"symbol": "0<trans>3", "start_frame": 2, "end_frame": 4},
        {"symbol": "z", "start_frame": 4, "end_frame": 7},
    ]

    trajectory = build_state_trajectory(merged, ["a", "b", "c", "d"])

    assert trajectory[0] == {
        "phoneme": "<del>",
        "start_state": -1,
        "end_state": 3,
        "start_frame": 4,
        "end_frame": 4,
        "variation_type": "deletion",
    }
    assert trajectory[1]["phoneme"] == "z"
    assert trajectory[1]["start_state"] == 3
    assert trajectory[1]["end_state"] == 4
    assert trajectory[1]["start_frame"] == 4
    assert trajectory[1]["end_frame"] == 7
    assert trajectory[1]["variation_type"] == "substitution"
