from irish_parse.modernize import MATCH, UNCERTAIN, align


def _origs(al):
    return [m.original for m in al.mapped]


def test_one_to_one_alignment():
    pairs = [("Áindrías", "Aindrias"), ("an", "an"), ("Ime", "Ime")]
    forms = ["Aindrias", "an", "Ime"]
    al = align(pairs, forms)
    assert _origs(al) == ["Áindrías", "an", "Ime"]
    assert all(m.status == MATCH for m in al.mapped)
    assert al.dropped == []


def test_expansion_groups_words_under_one_original():
    pairs = [("d'ith", "a d' ith"), ("siad", "siad")]
    forms = ["a", "d'", "ith", "siad"]
    al = align(pairs, forms)
    assert _origs(al) == ["d'ith", "d'ith", "d'ith", "siad"]
    assert al.mapped[0].orig_index == al.mapped[2].orig_index
    assert al.mapped[3].orig_index != al.mapped[0].orig_index


def test_identical_surfaces_stay_distinct_originals():
    # two different originals standardize to the same word
    pairs = [("a's", "is"), ("is", "is")]
    forms = ["is", "is"]
    al = align(pairs, forms)
    assert _origs(al) == ["a's", "is"]
    assert al.mapped[0].orig_index != al.mapped[1].orig_index


def test_deletion_recorded_for_reinsertion():
    # standard drops the patronymic particle (empty standard form)
    pairs = [("Baile", "Baile"), ("ui", ""), ("Mún", "Mún")]
    forms = ["Baile", "Mún"]
    al = align(pairs, forms)
    assert _origs(al) == ["Baile", "Mún"]
    assert len(al.dropped) == 1
    assert al.dropped[0].original == "ui"
    assert al.dropped[0].after_index == 0


def test_unmatched_parser_token_joins_neighbouring_original():
    pairs = [("dhó", "dhó")]
    forms = ["do", "dhó"]  # parser produced an extra token
    al = align(pairs, forms)
    assert _origs(al) == ["dhó", "dhó"]
    assert al.mapped[0].orig_index == al.mapped[1].orig_index
    assert al.mapped[0].status == UNCERTAIN


def test_single_divergence_does_not_desync_rest():
    pairs = [("a", "a"), ("bszz", "xyz"), ("c", "c")]
    forms = ["a", "zzz", "c"]
    al = align(pairs, forms)
    assert al.mapped[0].original == "a" and al.mapped[0].status == MATCH
    assert al.mapped[2].original == "c" and al.mapped[2].status == MATCH
    assert al.mapped[1].original == "bszz"
    assert al.mapped[1].status == UNCERTAIN
