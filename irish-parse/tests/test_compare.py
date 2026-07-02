from irish_parse import compare
from irish_parse.conllu import Sentence, Token


def _sent(rows):
    toks = []
    for i, (form, upos, head, deprel, lemma) in enumerate(rows, start=1):
        toks.append(
            Token(id=str(i), form=form, lemma=lemma, upos=upos, head=head, deprel=deprel)
        )
    return Sentence(tokens=toks)


def test_detects_field_disagreements():
    st = _sent([("Bhí", "VERB", "0", "root", "bí"), ("sé", "PRON", "1", "nsubj", "sé")])
    ud = _sent([("Bhí", "VERB", "0", "root", "bí"), ("sé", "PRON", "1", "obj", "sé")])
    d = compare.compare("1", "Bhí sé", st, ud)
    assert not d.misaligned
    assert len(d.diffs) == 1
    assert d.diffs[0].field == "deprel"
    assert (d.diffs[0].stanza, d.diffs[0].udpipe) == ("nsubj", "obj")
    assert d.n_tokens_affected == 1


def test_flags_tokenization_mismatch():
    st = _sent([("a", "X", "0", "root", "a"), ("b", "X", "1", "dep", "b")])
    ud = _sent([("a", "X", "0", "root", "a")])
    d = compare.compare("1", "a b", st, ud)
    assert d.misaligned


def test_markdown_reports_clean_run():
    st = _sent([("a", "X", "0", "root", "a")])
    ud = _sent([("a", "X", "0", "root", "a")])
    md = compare.render_markdown([compare.compare("1", "a", st, ud)])
    assert "No disagreements" in md


def test_markdown_includes_diff_table():
    st = _sent([("sé", "PRON", "1", "nsubj", "sé")])
    ud = _sent([("sé", "PRON", "1", "obj", "sé")])
    md = compare.render_markdown([compare.compare("1", "sé", st, ud)])
    assert "| Field |" in md.replace("  ", " ") or "Field" in md
    assert "nsubj" in md and "obj" in md
