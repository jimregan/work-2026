import sys
import types

import torch

from dysfluent_wfst import decoder as decoder_mod


class FakeDenseFsaVec:
    def to(self, _device):
        return self


class FakeK2Fsa:
    def to(self, _device):
        return self


def test_candidate_phoneme_sequences_prefers_rule_expanded_paths(monkeypatch):
    dec = object.__new__(decoder_mod.Decoder)
    dec.rules_fst = object()
    dec.lexicon_fst = object()
    dec._get_phoneme_ids = lambda phonemes: [len(p) for p in phonemes]
    dec._enumerate_label_paths = lambda fst: [[7, 8], [7, 8], [9, 10]]

    captured = {}

    def fake_build_utterance_fst(words, lexicon_fst, rules_fst):
        captured["words"] = words
        captured["lexicon_fst"] = lexicon_fst
        captured["rules_fst"] = rules_fst
        return object()

    monkeypatch.setattr(decoder_mod, "build_utterance_fst", fake_build_utterance_fst)

    sequences = dec._candidate_phoneme_sequences(["aa"], "Hello, world!")

    assert captured["words"] == ["hello", "world"]
    assert sequences == [[7, 8], [9, 10]]


def test_decode_utterance_uses_frame_shift_for_segment_times(monkeypatch):
    dec = object.__new__(decoder_mod.Decoder)
    dec.device = "cpu"
    dec.input_syms = object()
    dec.ctc_topo = types.SimpleNamespace(copy=lambda: types.SimpleNamespace(arcsort=lambda *_args: None))
    dec.rules_fst = None
    dec.similarity_matrix = None
    dec.phn2idx = None
    dec.lexicon_list = ["", "a", "b"]
    dec._candidate_phoneme_sequences = lambda ref_phonemes, ref_text: [[1, 2]]

    monkeypatch.setattr(decoder_mod, "create_dense_fsa_vec", lambda emission, lengths: FakeDenseFsaVec())
    monkeypatch.setattr(decoder_mod, "build_output_symbol_table", lambda input_syms: object())

    class FakeRef:
        def arcsort(self, *_args):
            return None

        def optimize(self):
            return self

    fake_ref = FakeRef()
    monkeypatch.setattr(decoder_mod, "build_ref_fst", lambda **kwargs: fake_ref)

    monkeypatch.setattr(decoder_mod.pynini, "compose", lambda left, right: object())
    monkeypatch.setattr(decoder_mod.pynini, "union", lambda *fsts: fsts[0])
    monkeypatch.setattr(decoder_mod, "fst_to_k2_str", lambda fst, to_log_probs=True: "fake")

    fake_k2 = types.SimpleNamespace(
        Fsa=types.SimpleNamespace(from_str=lambda fst_str, acceptor=False: FakeK2Fsa()),
        arc_sort=lambda fsa: fsa,
        intersect_dense=lambda composed_k2, dense_fsa, output_beam=25.0: object(),
        shortest_path=lambda lattice, use_double_scores=True: [
            types.SimpleNamespace(aux_labels=torch.tensor([11, 11, 12, 12]))
        ],
    )
    monkeypatch.setitem(sys.modules, "k2", fake_k2)

    monkeypatch.setattr(
        decoder_mod,
        "collapse_label_runs",
        lambda labels, output_syms: [
            {"symbol": "a", "start_frame": 1, "end_frame": 3},
            {"symbol": "b", "start_frame": 3, "end_frame": 5},
        ],
    )
    monkeypatch.setattr(decoder_mod, "merge_trans_markers", lambda runs: runs)
    monkeypatch.setattr(
        decoder_mod,
        "build_state_trajectory",
        lambda merged, ref_phonemes: [
            {
                "phoneme": "a",
                "start_state": 0,
                "end_state": 1,
                "start_frame": 1,
                "end_frame": 3,
                "variation_type": "normal",
            },
            {
                "phoneme": "b",
                "start_state": 1,
                "end_state": 2,
                "start_frame": 3,
                "end_frame": 5,
                "variation_type": "normal",
            },
        ],
    )

    alignment = dec.decode_utterance(
        logits=torch.zeros(6, 4),
        length=6,
        ref_phonemes=["a", "b"],
        ref_text="ignored",
        frame_shift_ms=12.5,
    )

    assert alignment.frame_shift_ms == 12.5
    assert alignment.segments[0].start_frame == 1
    assert alignment.segments[0].end_frame == 3
    assert alignment.segments[0].start_time_s == 0.0125
    assert alignment.segments[0].end_time_s == 0.0375
    assert alignment.segments[1].start_time_s == 0.0375
    assert alignment.segments[1].end_time_s == 0.0625
