"""Merging analytic verb+pronoun splits of synthetic forms.

Fixture shapes are taken from real pipeline output: the API renders
gheánfad -> "déanfaidh mé" and arsé -> "ar sé" (verified live), and
build_primary turns those into two-word multiword tokens.
"""
from irish_parse.build import merge_synthetic_pronouns
from irish_parse.conllu import Sentence, Token


def _tok(id, form, lemma, upos, feats, head, deprel, misc="_"):
    return Token(id=id, form=form, lemma=lemma, upos=upos, xpos="_",
                 feats=feats, head=head, deprel=deprel, deps="_", misc=misc)


def test_synthetic_verb_merges_and_renumbers():
    # Ní gheánfad -> standard "Ní déanfaidh mé" (API, verified live)
    s = Sentence(tokens=[
        _tok("1", "Ní", "ní", "PART", "_", "3", "advmod"),
        _tok("2-3", "gheánfad", "_", "_", "_", "_", "_"),
        _tok("2", "déanfaidh", "déan", "VERB", "Mood=Ind|Tense=Fut", "0", "root"),
        _tok("3", "mé", "mé", "PRON", "Number=Sing|Person=1", "2", "nsubj"),
        _tok("4", ".", ".", "PUNCT", "_", "2", "punct"),
    ])
    merge_synthetic_pronouns(s)
    assert [t.id for t in s.tokens] == ["1", "2", "3"]
    verb = s.tokens[1]
    assert verb.form == "gheánfad"  # original synthetic surface
    assert verb.lemma == "déan"
    assert verb.feats == "Mood=Ind|Number=Sing|Person=1|Tense=Fut"
    # heads renumbered: Ní and the full stop now point at the merged verb
    assert s.tokens[0].head == "2"
    assert s.tokens[2].head == "2"


def test_quotative_ar_se_is_not_merged():
    # arsé -> "ar sé" is a deliberate split (pre-standard rule), keep it
    s = Sentence(tokens=[
        _tok("1-2", "arsé", "_", "_", "_", "_", "_"),
        _tok("1", "ar", "ar", "VERB", "Mood=Ind|Tense=Past", "0", "root"),
        _tok("2", "sé", "sé", "PRON", "Gender=Masc|Number=Sing|Person=3", "1", "nsubj"),
    ])
    merge_synthetic_pronouns(s)
    assert [t.id for t in s.tokens] == ["1-2", "1", "2"]


def test_analytic_original_untouched():
    # chonnairc mé: two original tokens, no multiword token -> no merge
    s = Sentence(tokens=[
        _tok("1", "chonnairc", "feic", "VERB", "Mood=Ind|Tense=Past", "0", "root"),
        _tok("2", "mé", "mé", "PRON", "Number=Sing|Person=1", "1", "nsubj"),
    ])
    merge_synthetic_pronouns(s)
    assert [t.id for t in s.tokens] == ["1", "2"]


def test_non_verb_pronoun_group_untouched():
    # d'ith -> a d' ith: three-row group, not a verb+pronoun pair
    s = Sentence(tokens=[
        _tok("1-3", "d'ith", "_", "_", "_", "_", "_"),
        _tok("1", "a", "a", "PART", "_", "3", "mark:prt"),
        _tok("2", "d'", "do", "PART", "_", "3", "mark:prt"),
        _tok("3", "ith", "ith", "VERB", "Mood=Ind|Tense=Past", "0", "root"),
    ])
    merge_synthetic_pronouns(s)
    assert [t.id for t in s.tokens] == ["1-3", "1", "2", "3"]


def test_range_misc_carried_onto_merged_verb():
    s = Sentence(tokens=[
        _tok("1-2", "bhéidheas", "_", "_", "_", "_", "_", misc="Align=Check"),
        _tok("1", "bheidh", "bí", "VERB", "Mood=Ind|Tense=Fut", "0", "root"),
        _tok("2", "mé", "mé", "PRON", "Number=Sing|Person=1", "1", "nsubj"),
    ])
    merge_synthetic_pronouns(s)
    assert len(s.tokens) == 1
    assert s.tokens[0].misc_dict().get("Align") == "Check"
