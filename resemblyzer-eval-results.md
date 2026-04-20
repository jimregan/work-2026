# Resemblyzer Model Retrieval Eval (Sanity Check)

Model: `/data/wavlm-resemblyzer`
Index: `/data/osr-dataset`
Queries: `/data/osr-segments` (390 queries)
Command:
```bash
python /workspace/experiment/retrieval_eval.py \
    --model_dir /data/wavlm-resemblyzer \
    --index_dataset /data/osr-dataset \
    --query_dir /data/osr-segments \
    --index_cache /data/multiaxis-index.pt \
    --output_json /results/multiaxis-resemblyzer-eval.json
```

## Results

| Axis | Metric | Score |
|------|--------|-------|
| Semantic | Sentence-set R@1 | 0.990 |
| Semantic | Sentence-set R@5 | 0.995 |
| Speaker ID | Correct dialect in top hit | 1.000 |
| Dialect | Unambiguous R@1 | 1.000 |
| Dialect | Unambiguous R@5 | 0.917 |
| Gender | R@1 | 0.744 |

## Notes

### Semantic

Excellent — 99% of queries retrieve the correct sentence set at @1.

### Speaker ID

The index labels are speaker-level (`osr_uk_1`, `osr_us_3`, etc.), so the most meaningful
metric here is dialect separation. UK and US queries are perfectly separated at @1.
Individual speaker discrimination within the same dialect was not evaluated.

### Dialect

UK-only sentence sets (H43, H52, H58) retrieve other UK items at R@1 = 1.000.
US-only sentence sets (H23, H25, H27, H28, H30–H35, H57, H59, H72) get R@1 = 0.992,
R@5 = 1.000. The one failure at @1 is recovered by @5.

Overlapping sentence sets (H1–H5, H42) are ambiguous for dialect evaluation and were excluded.

### Gender

The gender axis has a clear bias problem:

| Query gender | R@1 |
|---|---|
| Male | 1.000 (190/190) |
| Female | 0.500 (100/200) |

R@5 == R@1 for gender, so it is not a ranking issue — the female queries consistently
return male as the top result across all 5 hits. The 2-dim gender embedding is collapsing
female onto male.

Likely cause: class imbalance in the training data (male speakers dominate), so the model
learns a biased decision boundary. Worth addressing before using the gender axis for retrieval.
