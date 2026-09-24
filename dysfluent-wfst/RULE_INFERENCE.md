# Residual phonological rule inference

`dysfluent-learn-rules` learns rules for variation that is not already
licensed by a known phonological grammar. The output uses the same YAML format
as the hand-written rules and can be compiled directly with
`dysfluent_wfst.rules.compile_rules`.

## Recommended workflow

Prepare a tab-separated file with an item id, citation pronunciation, and
observed pronunciation:

```text
utt-1\tn t s\tn s
utt-2\ta n b\ta m b
```

Then run:

```bash
dysfluent-learn-rules \
  --pairs pronunciations.tsv \
  --known-rules rules/swedish_hypo.yaml \
  --phone-classes rules/swedish_phone_classes.yaml \
  --min-count 2 \
  --min-probability 0.1 \
  --output learned.yaml
```

For each item, the known rules are compiled as a Pynini `cdrewrite` cascade.
The learner enumerates up to `--max-known-variants` licensed outputs and uses
the one nearest to the observation as its baseline. An exact licensed output
is considered explained and produces no learned rule. This bound prevents a
large cascade of optional rules from expanding without limit.

Residual alignments are coalesced, so substitutions, deletions, insertions,
and contiguous multi-phone changes can all become candidates. Contexts are
emitted both literally and, when a class file is supplied, as generalized
phone-class patterns. Unchanged opportunities provide negative evidence.

Candidates are ranked using occurrence count, empirical probability, an
overgeneration penalty, and a small penalty for generalized contexts. Use
`--overgeneration-penalty` and `--complexity-penalty` to tune these terms.
Counts, probability, score, examples, and known-rule coverage are retained as
YAML metadata. The compiler turns a learned probability into a tropical cost
of `-log(probability)` on the changed path.

Evaluate learned rules on held-out pronunciation pairs before adding them to
the decoding grammar. The score ranks hypotheses within the training sample;
it is not a substitute for held-out validation.
