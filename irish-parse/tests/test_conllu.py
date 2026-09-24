from irish_parse import conllu

SAMPLE = """# sent_id = 1
# text = Áindrías an Ime.
# text_standard = Aindrias an Ime.
1\tÁindrías\tAindrias\tPROPN\tNoun\tDefinite=Def|Gender=Masc\t0\troot\t_\tNamedEntity=Yes
2\tan\tan\tDET\tArt\t_\t3\tdet\t_\t_
3\tIme\tim\tNOUN\tNoun\tCase=Gen\t1\tnmod\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_
"""


def test_round_trip_preserves_metadata_order_and_tokens():
    sents = conllu.parse(SAMPLE)
    assert len(sents) == 1
    s = sents[0]
    assert [k for k, _ in s.metadata] == ["sent_id", "text", "text_standard"]
    assert s.meta_get("text") == "Áindrías an Ime."
    assert len(s.tokens) == 4
    assert s.tokens[0].form == "Áindrías"
    # dump + reparse is stable
    again = conllu.parse(conllu.dump(sents))
    assert conllu.dump(again) == conllu.dump(sents)


def test_misc_helpers():
    tok = conllu.Token(misc="NamedEntity=Yes|SpaceAfter=No")
    d = tok.misc_dict()
    assert d == {"NamedEntity": "Yes", "SpaceAfter": "No"}
    tok.add_misc("Orig", "Áindrías")
    assert tok.misc_dict()["Orig"] == "Áindrías"
    empty = conllu.Token()
    assert empty.misc_dict() == {}
    empty.add_misc("Align", "Check")
    assert empty.misc == "Align=Check"


def test_multiple_sentences_split_on_blank_line():
    text = SAMPLE + "\n" + SAMPLE
    assert len(conllu.parse(text)) == 2
