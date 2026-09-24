"""End-to-end FST pipeline tests against real pynini.

These exercise the parts the mocked unit tests cannot: actual FST
construction, composition, lexicon lookup, rule application, and k2
text export. They are skipped automatically when only the fake pynini
shim (see conftest.py) is available, i.e. when real pynini is not
installed on the host.
"""

import pynini
import pytest

REAL_PYNINI = (
    getattr(pynini, "__version__", None) is not None
    and hasattr(pynini, "cross")
    and hasattr(pynini, "Fst")
)

pytestmark = pytest.mark.skipif(
    not REAL_PYNINI, reason="requires real pynini (use the dev container)"
)

from dysfluent_wfst.symbols import build_symbol_table, build_output_symbol_table
from dysfluent_wfst.lexicon import build_lexicon_fst, lookup_word, build_utterance_fst
from dysfluent_wfst.rules import compile_rules
from dysfluent_wfst.ctc_topo import build_ctc_topo
from dysfluent_wfst.ref_fst import build_ref_fst
from dysfluent_wfst.k2_bridge import fst_to_k2_str


VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
    "h": 5, "ʉː": 6, "s": 7, "ə": 8, "t": 9,
    "ɡ": 10, "ɑː": 11, "a": 12, "ɣ": 13, "ö": 14, "r": 15, "n": 16,
}


@pytest.fixture
def syms():
    return build_symbol_table(VOCAB)


@pytest.fixture
def lexicon_fst(syms):
    entries = [
        ("huset", "h ʉː s ə t"),
        ("gata", "ɡ ɑː t a"),
        ("hörn", "h ö r n"),
    ]
    return build_lexicon_fst(entries, input_token_type="utf8", output_syms=syms)


def _output_label_paths(fst):
    """Enumerate output-label id paths through an acyclic FST."""
    zero = pynini.Weight.zero(fst.weight_type())
    paths = []

    def dfs(state, acc):
        if fst.final(state) != zero:
            paths.append(list(acc))
        for arc in fst.arcs(state):
            lab = arc.olabel if arc.olabel > 0 else arc.ilabel
            dfs(arc.nextstate, acc + ([lab] if lab > 0 else []))

    if fst.start() != -1:
        dfs(fst.start(), [])
    return paths


def _path_symbols(paths, syms):
    return {tuple(syms.find(i) for i in p) for p in paths}


def test_ctc_topo_builds(syms):
    """build_ctc_topo must not rely on the non-existent pynini.Compiler."""
    topo = build_ctc_topo(syms.num_symbols(), syms)
    assert topo.start() == 0
    assert topo.num_states() == syms.num_symbols()


def test_ref_fst_builds_and_composes_with_topo(syms):
    phoneme_ids = [syms.find(s) for s in ["h", "ʉː", "s", "ə", "t"]]
    osyms = build_output_symbol_table(syms)
    ref = build_ref_fst(
        phoneme_ids, beta=2.0, input_syms=syms, output_syms=osyms,
        skip=True, back=True, sub=False,
    )
    assert ref.start() == 0
    assert ref.num_states() == len(phoneme_ids) + 1

    topo = build_ctc_topo(syms.num_symbols(), syms)
    ref.arcsort("ilabel")
    composed = pynini.compose(topo.copy(), ref)
    assert composed.start() != -1, "topo ∘ ref must be non-empty"

    k2_str = fst_to_k2_str(composed, to_log_probs=True)
    lines = k2_str.splitlines()
    # final line is the superfinal state id; arc lines have 5 fields
    assert lines[-1].isdigit()
    assert all(len(ln.split()) in (1, 5) for ln in lines)


def test_lookup_ascii_word(lexicon_fst, syms):
    fst = lookup_word("huset", lexicon_fst).project("output")
    assert _path_symbols(_output_label_paths(fst), syms) == {
        ("h", "ʉː", "s", "ə", "t")
    }


def test_lookup_non_ascii_word(lexicon_fst, syms):
    """B3 regression: utf8 query must match the utf8 lexicon input side."""
    fst = lookup_word("hörn", lexicon_fst)
    assert fst.start() != -1, "non-ASCII lookup must not be empty"
    proj = fst.project("output")
    assert _path_symbols(_output_label_paths(proj), syms) == {
        ("h", "ö", "r", "n")
    }


def test_rules_expand_pronunciation_lattice(lexicon_fst, syms, tmp_path):
    """B2 regression: rule FSTs must share the SymbolTable label space.

    With byte-based rule FSTs the composition was silently empty; here
    the g→ɣ lenition rule must add a variant path alongside the citation.
    """
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text(
        "rules:\n"
        "  - segment: 'ɡ'\n"
        "    replacement: 'ɣ'\n"
        "    preceding_context: ''\n"
        "    following_context: ''\n",
        encoding="utf-8",
    )
    rules_fst = compile_rules(str(rules_yaml), syms)
    assert rules_fst is not None

    utt = build_utterance_fst(["gata"], lexicon_fst, rules_fst)
    got = _path_symbols(_output_label_paths(utt), syms)
    assert ("ɡ", "ɑː", "t", "a") in got, "citation form must survive"
    assert ("ɣ", "ɑː", "t", "a") in got, "rule variant must be present"


def test_rules_without_rules_is_citation_only(lexicon_fst, syms):
    utt = build_utterance_fst(["gata"], lexicon_fst, rules_fst=None)
    assert _path_symbols(_output_label_paths(utt), syms) == {
        ("ɡ", "ɑː", "t", "a")
    }
