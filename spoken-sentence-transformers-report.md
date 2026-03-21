# Multi-Axis Spoken Sentence Transformers: Internal Report

**Date:** March 2026

## Goal

The project investigated whether a single speech encoder could support retrieval along
multiple independent axes simultaneously — specifically semantic content, speaker identity,
gender, and dialect — by fine-tuning WavLM with per-axis projection heads trained on
contrastive objectives.

The intended application was audio retrieval where a user could weight different axes
(e.g. "find speech that says the same thing, by a similar-sounding speaker") without
requiring separate models.

## What Was Built

### Training data

The training dataset (`merged-vctk-cmuarctic-gbi`) combined VCTK, CMU Arctic, and
Google's Britain and Ireland (GBI) dataset. It provides the natural supervision signal
for multi-axis training: the same sentence read by different speakers, from different
corpora, with known gender labels.

### Models trained

Three models were trained, all starting from a WavLM base:

- **wavlm-semantic**: Single projection head trained for semantic (content) similarity only.
- **wavlm-multiaxis**: Multiple projection heads trained jointly for semantic, speaker identity, gender, and dialect axes.
- **wavlm-resemblyzer**: Multiaxis variant where the speaker head was replaced with embeddings from Resemblyzer (a pretrained speaker verification model) as a fixed target.

### Evaluation framework

A retrieval evaluation pipeline was built (`retrieval_eval.py`, `retrieval_eval.sh`,
`analyze_results.py`) supporting:

- Per-axis nearest-neighbour retrieval with configurable top-k
- Combined retrieval with weighted axis scores
- Preference-flip evaluation: does the model rank same-sentence items above
  same-speaker items when speaker is a distractor?
- Three test sets: OSR (in-distribution), p315 (held-out VCTK speaker),
  and rehasp (held-out speaker reading Harvard sentences)

## Results

### OSR retrieval (in-distribution)

All models perform near-perfectly on OSR semantic retrieval (R@1 98.7–100%). This
confirms the training objective is working and the evaluation pipeline is correct.

### p315 retrieval (cross-speaker, held-out speaker)

| Model | Sem R@1 | Sem R@5 |
|---|---|---|
| wavlm-semantic | 90.0% | 96.2% |
| wavlm-multiaxis | 51.9% | 72.5% |
| wavlm-resemblyzer | 0.6% | 0.6% |
| text baseline | 94.4% | 99.4% |
| CLAP baselines | 0.0% | 0.0% |

The semantic-only model achieves 90% R@1, close to the text-match ceiling of 94.4%.
The multiaxis model drops to 52%: the speaker axis is pulling retrieval toward training
speakers that sound similar to p315, overriding the semantic signal. The resemblyzer
model essentially fails (0.6%), suggesting the Resemblyzer-based speaker head dominates
and the semantic axis has no effective influence on retrieval.

Note: the index for p315 evaluation contained training speakers, which is the most
favourable possible condition for the multiaxis model. Under a clean evaluation (held-out
index speakers), performance would likely be lower.

### Rehasp retrieval (cross-speaker, cross-corpus preference-flip)

The preference-flip evaluation tests whether the model ranks same-sentence items (from
OSR) above same-speaker items (lucy/rehasp rep001) when both are in the index.

| Model | Sem R@1 (→OSR only) | PF P@1 (→mixed index) |
|---|---|---|
| wavlm-semantic | 63.9% | n/a (no speaker axis) |
| wavlm-multiaxis | 16.8% | 0.4% |
| wavlm-resemblyzer | 3.0% | 0.9% |

When querying against the OSR-only index (no same-speaker distractor), the semantic
model retrieves the correct sentence 64% of the time; multiaxis achieves only 17%.
When the same-speaker item is added to the index (mixed), the preference-flip P@1
falls to under 1% for both multiaxis models: the same-speaker item is ranked first
in essentially all cases, making the combined retrieval useless for semantic retrieval.

The mean rank of same-sentence items (~103 for multiaxis, ~173 for resemblyzer) versus
different-sentence items (~224 for both) confirms the semantic axis is doing something,
but the speaker axis overwhelms it in the combined score.

## Conclusions

**Combining axes in retrieval does not work with the current approach.** When the
query and index speakers differ — which is the normal retrieval scenario — the speaker
axis actively degrades semantic retrieval performance. This is not a tuning problem:
sweeping speaker weights from +1 to −2 does not recover performance.

**The semantic-only model works.** 90% cross-speaker R@1 on a held-out speaker is
a reasonable result and close to the text ceiling. Fine-tuning WavLM for semantic
retrieval is viable.

**The resemblyzer approach does not work at all for semantic retrieval.** The
Resemblyzer speaker embeddings dominate and the semantic axis produces near-zero
retrieval performance on out-of-distribution speakers.

**The multiaxis training objective works on in-distribution data** (OSR evaluation
is near-perfect) but does not generalise to cross-speaker retrieval in the way the
project intended.

## What Would Be Needed to Proceed

A publishable result would require at minimum:

1. A clean cross-speaker retrieval benchmark (index speakers not seen during training).
2. A method for combining axes that does not degrade semantic performance — e.g.
   semantic-first retrieval with speaker re-ranking on a shortlist.
3. Better speaker axis generalisation to unseen speakers (the current model effectively
   memorises training speaker identities).

These are non-trivial and were outside the scope of the feasibility study.
