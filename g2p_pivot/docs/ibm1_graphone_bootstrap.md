# IBM Model 1 Graphone Bootstrap

This is the first curriculum step for learning language-specific graphone evidence without trusting the original language tags too much.

The narrow starting assumption is:

> begin only with entries where grapheme-token count and phone-token count are equal, train a per-language IBM Model 1 table, and export high-confidence `P(phone | grapheme)` seeds.

This does not solve full graphone alignment. It gives a low-risk prior for later stages that admit multigraphs, deletions, insertions, vowel length, and compound-internal language changes.

## Input Format

Use CSV or TSV with at least these columns:

```text
language,word,phones
```

Phones should already be tokenized. By default, phone tokens are whitespace-separated:

```text
language,word,phones
swe,sol,s u: l
eng,west,w e s t
```

## Run

```bash
python3 scripts/ibm1_graphone_bootstrap.py examples/tiny_lexicon.tsv \
  --language-col language \
  --word-col word \
  --phones-col phones \
  --seeds-out /tmp/graphone_seeds.tsv \
  --scores-out /tmp/entry_scores.tsv \
  --training-out /tmp/one_to_one_training.tsv
```

For TSV input, the delimiter is inferred from `.tsv`. For CSV input, it is inferred from `.csv`. You can override it with `--delimiter "\\t"` or `--delimiter ","`.

For the current Braxen export:

```bash
python3 scripts/ibm1_graphone_bootstrap.py /Users/joregan/Playing/braxen/dict/braxen-sv.tsv \
  --no-header \
  --delimiter "\t" \
  --word-index 0 \
  --phones-index 1 \
  --language-index 3 \
  --id-index 26 \
  --language-filter swe \
  --language-filter eng \
  --language-filter dan \
  --language-filter fre \
  --language-filter spa \
  --language-filter ita \
  --language-filter lat \
  --casefold-words \
  --drop-phone-token "." \
  --drop-phone-token "|" \
  --strip-phone-prefixes "\"'," \
  --seeds-out /tmp/braxen_graphone_seeds.tsv \
  --scores-out /tmp/braxen_entry_scores.tsv \
  --training-out /tmp/braxen_one_to_one_training.tsv \
  --skip-scores
```

This skips lines beginning with `#` by default. The `--skip-scores` option is useful for the first full-corpus pass because entry scoring compares every entry against every learned language model.

## Outputs

`graphone_seeds.tsv` contains:

```text
language
grapheme
phone
probability
expected_count
entropy
accepted_seed
```

`accepted_seed=1` means the row passed the configured thresholds:

- `--min-seed-count`
- `--min-probability`
- `--max-entropy`

`entry_scores.tsv` scores each entry under every learned language model and reports:

- the best-scoring language
- the rank of the declared language
- the margin between best and second-best language
- all per-language scores

Use this as weak evidence, not as a final relabeling decision.

## Next Notebook Cells

1. Load `graphone_seeds.tsv` and inspect top accepted correspondences per language.
2. Plot entropy by grapheme and language.
3. Load `entry_scores.tsv` and inspect entries where `declared_rank` is not `1`.
4. For compounds, score candidate parts separately and compare part-level language evidence against the whole-word tag.
5. Use accepted seeds as priors for a second-stage monotonic graphone aligner that permits `2:1`, `1:2`, and epsilon alignments.
