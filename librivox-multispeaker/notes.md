- [Accents Table](https://wiki.librivox.org/index.php?title=Accents_Table)

---

**Multi-Axis Speech Similarity Evaluation Checklist**

## Metrics
- [ ] Recall@K (K = {1, 5, 10})
- [ ] Mean Average Precision (MAP)
- [ ] nDCG (Normalized Discounted Cumulative Gain)
- [ ] Precision@K (K = {1, 5, 10})
- [ ] Optional: MRR (for comparison to existing work)

## Per-Axis Evaluation Setup

### Semantic Axis
- [ ] Define "correct": same sentence/content, different speaker
- [ ] Count correct answers per query in dataset
- [ ] Run retrieval evaluation with above metrics

### Speaker Identity Axis
- [ ] Define "correct": same speaker, different utterance
- [ ] Count correct answers per query in dataset
- [ ] Run retrieval evaluation with above metrics

### Gender Axis
- [ ] Define "correct": same gender, different speaker/utterance
- [ ] Count correct answers per query in dataset
- [ ] Run retrieval evaluation with above metrics

### Accent Axis
- [ ] Define "correct": same accent, different speaker/utterance
- [ ] Count correct answers per query in dataset
- [ ] Run retrieval evaluation with above metrics

### Fluency Axis (LibriVox)
- [ ] Decide: categorical bins OR continuous similarity?
- [ ] If categorical: define "correct" as same fluency bin
- [ ] If continuous: consider different evaluation approach (correlation? ranking?)

### Prosody Axis (LibriVox)
- [ ] Decide: categorical bins OR continuous similarity?
- [ ] If categorical: define "correct" as similar prosodic profile
- [ ] If continuous: consider different evaluation approach

## Implementation
- [ ] Adapt MSEB retrieval code for multi-correct-answer metrics
- [ ] Implement per-axis "correct answer" definitions
- [ ] Handle variable numbers of correct answers per query
- [ ] Validate metrics on small subset before full evaluation

## Paper/Documentation
- [ ] Justify metric choices: "All axes involve multiple valid targets"
- [ ] Report K values used and why
- [ ] State what "correct" means for each axis explicitly
- [ ] Compare to MSEB baselines where applicable