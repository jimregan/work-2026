"""End-to-end test with the real parsers and the real intergaelic API.

Runs only where the full stack is installed (i.e. inside the devcontainer);
skipped elsewhere. No component is simulated.
"""
import importlib.util

import pytest

needs_stack = pytest.mark.skipif(
    importlib.util.find_spec("stanza") is None
    or importlib.util.find_spec("ufal") is None,
    reason="stanza/ufal.udpipe not installed — run inside the devcontainer",
)


@needs_stack
def test_full_pipeline_on_real_sentence(tmp_path):
    import parse_irish
    from irish_parse import conllu

    src = tmp_path / "in.txt"
    src.write_text(
        "Bhí Áindrías an Ime na chomhnaidhe i mBaile ui Mún i nGleann an Bhaile Dhuibh.\n"
    )
    out = tmp_path / "out"
    rc = parse_irish.main([str(src), "--out", str(out)])
    assert rc == 0

    primary = conllu.parse((tmp_path / "out.conllu").read_text())
    assert len(primary) == 1
    sent = primary[0]

    # the standardized form lives only in text_standard
    assert sent.meta_get("text") == (
        "Bhí Áindrías an Ime na chomhnaidhe i mBaile ui Mún i nGleann an Bhaile Dhuibh."
    )
    assert "chónaí" in (sent.meta_get("text_standard") or "")

    # FORM is the original surface, exactly as written
    forms = [t.form for t in sent.tokens if "-" not in t.id]
    assert "chomhnaidhe" in forms

    # LEMMA is the parser's lemmatisation of the standardized text
    tok = next(t for t in sent.tokens if t.form == "chomhnaidhe")
    assert tok.lemma == "cónaí"
    bhi = next(t for t in sent.tokens if t.form == "Bhí")
    assert bhi.lemma == "bí"

    # a real dependency structure came through
    assert any(t.deprel == "root" for t in sent.tokens)

    # the cross-check parse and the diff report were written
    udpipe = conllu.parse((tmp_path / "out.udpipe.conllu").read_text())
    assert len(udpipe) == 1
    assert (tmp_path / "out.diff.md").read_text().startswith("# Stanza vs UDPipe")
