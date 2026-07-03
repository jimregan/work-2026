from irish_parse.modernize import detokenize


def test_space_before_sentence_punctuation_removed():
    assert detokenize("Aindrias an Ime .") == "Aindrias an Ime."
    assert detokenize("cad é a dhéanfaidh mé ?") == "cad é a dhéanfaidh mé?"


def test_space_around_commas_and_brackets():
    assert detokenize("Ba leis Baile uí Mún , áit fiche bó .") == (
        "Ba leis Baile uí Mún, áit fiche bó."
    )
    assert detokenize("( rud )") == "(rud)"


def test_words_untouched():
    assert detokenize("i mBaile uí Mún") == "i mBaile uí Mún"
