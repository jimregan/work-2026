from irish_parse.prestandard import apply, load_rules


def _rules(tmp_path, text):
    p = tmp_path / "rules.tsv"
    p.write_text(text, encoding="utf-8")
    return load_rules(str(p))


def test_load_rules_skips_comments_and_sorts_longest_first(tmp_path):
    rules = _rules(tmp_path, "# comment\ngo\tcad\ngo dé\tcad é\n\nacht\tach\n")
    assert rules[0] == (["go", "dé"], ["cad", "é"])  # longest first
    assert (["acht"], ["ach"]) in rules


def test_multiword_rule_rewrites_standard_side_only(tmp_path):
    rules = _rules(tmp_path, "go dé\tcad é\n")
    # the API leaves Ulster "go dé" unstandardised
    pairs = [("go", "go"), ("dé", "dé"), ("gheánfaidh", "dhéanfaidh"), ("mé", "mé")]
    out = apply(pairs, rules)
    assert out == [
        ("go", "cad"),
        ("dé", "é"),
        ("gheánfaidh", "dhéanfaidh"),
        ("mé", "mé"),
    ]


def test_single_token_rule_overrides_api(tmp_path):
    rules = _rules(tmp_path, "acht\tach\n")
    pairs = [("acht", "acht"), ("ní", "ní")]
    assert apply(pairs, rules)[0] == ("acht", "ach")


def test_initial_capital_preserved(tmp_path):
    rules = _rules(tmp_path, "acht\tach\n")
    pairs = [("Acht", "Acht")]
    assert apply(pairs, rules)[0] == ("Acht", "Ach")


def test_shorter_replacement_deletes_tail_originals(tmp_path):
    # two originals collapse to one standard word: the second original's
    # standard side becomes empty -> Skip=Standard placeholder downstream
    rules = _rules(tmp_path, "fá dtaobh\tfaoi\n")
    pairs = [("fá", "fá"), ("dtaobh", "dtaobh"), ("de", "de")]
    out = apply(pairs, rules)
    assert out == [("fá", "faoi"), ("dtaobh", ""), ("de", "de")]


def test_longer_replacement_expands_last_original(tmp_path):
    rules = _rules(tmp_path, "go dé\tcad é a\n")
    pairs = [("go", "go"), ("dé", "dé")]
    out = apply(pairs, rules)
    assert out == [("go", "cad"), ("dé", "é a")]
