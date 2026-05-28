import yaml

from dysfluent_wfst.rule_learning import (
    align_phones,
    induce_rules,
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
