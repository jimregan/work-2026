

# Strategy for Multi-Axis Speech Search: High-Impact Revision

## 1. Phrasing the Dependency: Establishing Autonomy

To avoid the "salami slicing" perception, the phrasing must shift from "validating" the previous work to "interrogating its limits."

### Recommended Phrasing

> "While previous multi-axis conditioning [Anonymous, under review] demonstrated the feasibility of factored embeddings on controlled, high-fidelity TTS corpora, the stability of these factors under high intra-speaker variance remains unexplored. This work investigates the metric space's ability to maintain axis-orthogonality when exposed to the longitudinal acoustic shifts present in multi-year speech recordings (LibriVox) and the environmental noise of the Common Voice corpus."

## 2. Technical Contribution: Embedding Stability vs. Content Retrieval

Instead of just proving the model can find a sentence, we shift to proving the **Speaker Axis** is invariant to content, and the **Content Axis** is invariant to prosodic/temporal shifts.

### Proposed Experiments (The "Interesting" Path)

- **The Multi-Reader Stress Test (LibriVox):** Instead of a massive scrape, identify 5–10 readers who have recorded multiple books over several years.
    
    - _The Goal:_ Show that the Speaker Embedding for "Reader A" in 2015 is closer to "Reader A" in 2022 than it is to any other reader, _regardless_ of the text complexity.
        
- **Intra-Speaker Content Stability:** For a single speaker who repeats a phrase (found in CV or LibriVox), measure the Euclidean distance between content embeddings across different recording environments.
    

## 3. Training Data: The "Clean Sweep" vs. "Hard Negatives"

In the original paper, you had a natural held-out speaker. To make this paper a unique contribution, we move to a **Hard Negative** training curriculum.

### Exclusion & Sampling Strategy

1. **Exclude Both (Strict):** Maintain the disjoint Speaker and Sentence ID rule for the primary test set to ensure no "memorization."
    
2. **Hard Negative Mining:** During training, specifically sample "Same Speaker, Different Sentence" and "Different Speaker, Same Sentence" pairs for the contrastive loss. This forces the model to work harder to factorize the space than a random shuffle would.
    

## 4. Evaluation: Beyond Recall@K

If the dataset is larger, we need to look at the "topography" of the metric space:

- **Silhouette Score of Factors:** Measure how well the clusters form in the Speaker-subspace vs. the Content-subspace.
    
- **Inter-axis Correlation:** Calculate the mutual information between the factored embeddings. A "perfect" contribution shows near-zero mutual information while maintaining high retrieval performance.
    

## 5. Timeline: The 72-Hour "Sprint for Substance"

If you are ditching the "salami slice" (Swedish), here is the priority:

- **Thursday Morning:** Run a targeted scrape/process on 10 LibriVox readers with high volume (e.g., those who read many chapters of the same work or different works).
    
- **Thursday Evening:** Train the English CV model using the "Hard Negative" sampling logic.
    
- **Friday:** Evaluation of the "Longitudinal Stability" and "Metric Factorization" metrics.
    

**Why this is better:** It provides a "Qualitative and Quantitative Study of Embedding Stability," which is a much more robust PhD-level contribution than "We tried it in Swedish and it also worked."