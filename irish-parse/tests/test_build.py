from irish_parse import modernize
from irish_parse.build import build_primary
from irish_parse.conllu import Sentence, Token


def _parsed(rows):
    toks = []
    for i, (form, lemma, head, deprel) in enumerate(rows, start=1):
        toks.append(
            Token(id=str(i), form=form, lemma=lemma, upos="X", head=head, deprel=deprel)
        )
    return Sentence(tokens=toks)


def test_form_is_original_lemma_from_parser():
    pairs = [("chomhnaidhe", "chónaí")]
    parser = _parsed([("chónaí", "cónaí", "0", "root")])
    al = modernize.align(pairs, ["chónaí"])
    out = build_primary(parser, pairs, al)
    tok = out.tokens[0]
    assert tok.form == "chomhnaidhe"  # exactly as written, pre-standard
    assert tok.lemma == "cónaí"  # the lemmatizer's output


def test_split_becomes_multiword_token():
    # original "d'ith" -> standard "a d' ith" (gold: `2-4 d'ith`)
    pairs = [("Nuair", "Nuair"), ("d'ith", "a d' ith"), ("siad", "siad")]
    parser = _parsed(
        [
            ("Nuair", "nuair", "4", "mark"),
            ("a", "a", "4", "mark:prt"),
            ("d'", "do", "4", "mark:prt"),
            ("ith", "ith", "0", "root"),
            ("siad", "siad", "4", "nsubj"),
        ]
    )
    al = modernize.align(pairs, [t.form for t in parser.tokens])
    out = build_primary(parser, pairs, al)
    assert [t.id for t in out.tokens] == ["1", "2-4", "2", "3", "4", "5"]
    assert out.tokens[1].form == "d'ith"  # range line carries the original
    # word rows keep the standard forms and their analyses
    assert [t.form for t in out.tokens[2:5]] == ["a", "d'", "ith"]
    assert out.tokens[0].head == "4"
    assert out.tokens[5].head == "4"


def test_deleted_original_gets_placeholder_row_and_heads_renumber():
    pairs = [("Baile", "Baile"), ("ui", ""), ("Mún", "Mún")]
    parser = _parsed(
        [
            ("Baile", "Baile", "0", "root"),
            ("Mún", "Mún", "1", "flat:name"),
        ]
    )
    al = modernize.align(pairs, ["Baile", "Mún"])
    out = build_primary(parser, pairs, al)
    assert [t.form for t in out.tokens] == ["Baile", "ui", "Mún"]
    assert [t.id for t in out.tokens] == ["1", "2", "3"]
    placeholder = out.tokens[1]
    assert placeholder.misc_dict().get("Skip") == "Standard"
    assert placeholder.lemma == "_" and placeholder.head == "_"
    # Mún's head still points at Baile (id 1) after renumbering
    assert out.tokens[2].head == "1"
