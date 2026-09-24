import yaml

from dysfluent_wfst.rule_learning import (
    align_phones,
    changes_from_alignment,
    induce_rules,
    load_phone_classes,
    load_timit_pair_manifest,
    write_rules_yaml,
)


def test_align_phones_extracts_substitution_and_deletion():
    edits = align_phones(["n", "t", "s", "aa"], ["n", "s", "ah"])

    assert [(e.source, e.target) for e in edits] == [
        ("n", "n"),
        ("t", None),
        ("s", "s"),
        ("aa", "ah"),
    ]


def test_induce_rules_counts_contextual_changes():
    pairs = [
        ("utt1", "n t s", "n s"),
        ("utt2", "n t s", "n s"),
        ("utt3", "k t aa", "k t aa"),
    ]

    rules = induce_rules(pairs, min_count=2)

    assert len(rules) == 1
    assert rules[0].segment == "t"
    assert rules[0].replacement == ""
    assert rules[0].preceding_context == "n"
    assert rules[0].following_context == "s"
    assert rules[0].count == 2
    assert rules[0].opportunities == 2
    assert rules[0].probability == 1.0


def test_changes_coalesce_insertions_and_multi_phone_edits():
    edits = align_phones(["a", "b", "c"], ["a", "x", "y", "c"])
    changes = changes_from_alignment(edits)

    assert len(changes) == 1
    assert changes[0].source == ("b",)
    assert changes[0].target == ("x", "y")


def test_induce_rules_preserves_pure_insertions():
    rules = induce_rules(
        [("one", "a b", "a x b"), ("two", "a b", "a x b")],
        min_count=2,
    )

    assert any(
        rule.segment == ""
        and rule.replacement == "x"
        and rule.preceding_context == "a"
        and rule.following_context == "b"
        for rule in rules
    )


def test_known_variants_are_removed_before_residual_inference():
    pairs = [
        ("known", "a n", "a m"),
        ("residual1", "a t", "a s"),
        ("residual2", "a t", "a s"),
    ]
    rules = induce_rules(
        pairs,
        known_variants={"known": [["a", "m"]]},
        min_count=2,
    )

    assert [(rule.segment, rule.replacement) for rule in rules] == [("t", "s")]
    assert rules[0].known_rule_coverage == 1


def test_unchanged_examples_count_as_overgeneration_opportunities():
    rules = induce_rules(
        [("changed", "a t b", "a s b"), ("same", "a t b", "a t b")],
        min_count=1,
    )

    rule = next(r for r in rules if r.segment == "t" and r.replacement == "s")
    assert rule.opportunities == 2
    assert rule.probability == 0.5


def test_phone_classes_generalize_contexts(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("classes:\n  vowel: [a, e]\n", encoding="utf-8")
    classes = load_phone_classes(str(path))
    rules = induce_rules(
        [("one", "a t", "a s"), ("two", "e t", "e s")],
        min_count=2,
        phone_classes=classes,
    )

    generalized = next(r for r in rules if r.preceding_context == "[ae]")
    assert generalized.count == 2
    assert generalized.opportunities == 2


def test_write_rules_yaml_matches_compiler_format(tmp_path):
    rules = induce_rules(
        [("utt1", "aa n", "ah n"), ("utt2", "aa n", "ah n")],
        min_count=2,
    )
    path = tmp_path / "learned.yaml"

    write_rules_yaml(rules, str(path), include_stats=False)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data == {
        "rules": [
            {
                "segment": "aa",
                "replacement": "ah",
                "preceding_context": "",
                "following_context": "n",
            }
        ]
    }


def test_load_timit_pair_manifest_reads_phn_intervals(tmp_path):
    phn = tmp_path / "utt.phn"
    phn.write_text("0 100 n\n100 200 s\n", encoding="utf-8")
    manifest = tmp_path / "pairs.tsv"
    manifest.write_text(
        f"utt1\tn t s\t{phn}\n",
        encoding="utf-8",
    )

    assert load_timit_pair_manifest(str(manifest)) == [
        ("utt1", "n t s", "n s")
    ]


def test_load_timit_pair_manifest_resolves_paths_from_manifest_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    phn = data_dir / "utt.phn"
    phn.write_text("0 100 aa\n100 200 n\n", encoding="utf-8")
    manifest = tmp_path / "pairs.tsv"
    manifest.write_text("utt1\taa n\tdata/utt.phn\n", encoding="utf-8")

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    assert load_timit_pair_manifest(str(manifest)) == [
        ("utt1", "aa n", "aa n")
    ]
