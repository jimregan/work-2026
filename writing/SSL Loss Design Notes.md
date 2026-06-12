
# SSL Loss Design Notes: Multi-Axis Spoken Sentence Embeddings

## Context

Extension to the spoken sentence embeddings package. Goal: multi-axis encoder where each axis (semantic content, speaker identity, dialect, prosody, channel) has a separately trained head off a shared encoder bottleneck.

## Architecture

**Siamese UNet encoder** with shared weights. Two utterances are passed through the same encoder; the bottleneck produces the sentence-level embedding. Skip connections improve gradient flow and bottleneck quality but do not directly feed the loss (at least for paper 2 scope).

Per-axis projection heads (thin MLP or linear) sit on top of the bottleneck, one per axis.

## Loss Decomposition

The key insight: axes have different epistemological status, so a single loss type is wrong.

```
L_total = λ_sem * L_contrastive(h_sem)
        + Σ_k λ_k * L_VICReg(h_k)      # speaker, dialect, prosody, channel, ...
        + λ_decorr * L_crossaxis(H)     # optional Barlow-style inter-axis decorrelation
```

- **Semantic axis**: contrastive loss. Two utterances with the same words but different speakers are a positive pair; different words are a negative pair. Non-contrastive would cause collapse on the axis that matters most.
- **Identity-preserving axes** (speaker, dialect, channel): VICReg. Two crops/augmentations of the same utterance are genuine views of the same thing. Variance term prevents collapse; no negatives needed.
- **Cross-axis decorrelation**: optional Barlow-style term over the concatenated per-axis heads `H`, discouraging any two axes becoming redundant regardless of their individual loss type.

A single crop pair does double duty: negative for the semantic loss, positive for the speaker VICReg. Useful for batch construction efficiency.

## Positive Pair Construction for the Semantic Axis

Primary source: **text-matched pairs** from LibriVox / Common Voice — same sentence read by different speakers. Clean ground truth, no ASR noise.

Secondary source: **word-aligned crops from the same utterance** where the crop's semantic similarity to the source sentence falls below a threshold. These become hard negatives for the semantic axis while remaining positive pairs for the speaker axis (same speaker, same session — channel and speaker axes are controlled).

Similarity threshold operates as a **dead zone**:

- High similarity → positive (or discard as redundant augmentation)
- Mid-band → discard (ambiguous)
- Low similarity → hard negative

Use BERTScore or embedding-based similarity rather than n-gram overlap; function words contribute little to contextual embeddings and are effectively down-weighted automatically.

## Crop Boundary Conditions

**Snap all crop boundaries to word edges** using forced alignment (MFA or equivalent, already available from the word-aligned CV pipeline).

Rationale: random audio crops produce partially articulated boundary words that are phonetically ambiguous — a partial word can be perceptually equivalent to the full word, corrupting both the similarity calculation and the training signal. Word-boundary snapping eliminates this entirely: a word is either in the crop or not.

Secondary constraint: **minimum crop duration** to ensure prosody is a reliable signal. Very short crops make the prosody axis noisy.

Also filter or down-weight crops whose boundaries land on function words / clitics (determiners, prepositions) — technically clean but semantically near-zero weight.

## Scope Notes

- Paper 2: bottleneck-only loss; UNet as architectural choice for representation quality; per-axis heads with mixed contrastive/VICReg loss.
- Paper 3: multi-scale loss using skip connections, with axis-specific positive pairs at phone/word/phrase resolution; natural spectral graph theory framing.