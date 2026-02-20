

Corpus:

LibriVox... [librispeech, libritts, etc.]

LibriVox does not discourage its volunteers from re-recording works that
already exist in its collection; consequently, a number of books have
multiple recordings. Not only does it not discourage re-recordings, there
are specific collections that are dedicated to multiple recordings of the
same works, though these are usually shorter items, such as poems.

> https://librivox.org/pages/about-librivox/
>
> Our readers are free to choose the books they wish to record. LibriVox sees itself as a library of audiobooks.
>
> We accept all volunteers in all languages, with all kinds of accents. You’re welcome to volunteer to read any language you speak, as long as you can make yourself understood in it.

-----

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

-----

This checklist is very solid. The structure is right. What you need now is tightening and pruning so it doesn’t quietly explode into five papers.

I’ll go through it axis by axis and mark:

* ✅ Keep as core
* ⚠️ Optional / exploratory
* ❌ Dangerous scope creep
* 🔧 Suggested refinement

---

# 1️⃣ Metrics

Your metric set is good, but slightly overcomplete.

## Recommended Core Set

✅ **Recall@K (1, 5, 10)**
This should be your primary metric.

Why:

* Intuitive
* Retrieval-aligned
* Works with multi-correct answers
* Easy to explain

✅ **nDCG@K**
Very good for axes with multiple valid targets.

This is especially important for:

* Speaker
* Gender
* Accent
* Fluency (if categorical)

Optional:

⚠️ **MAP**
Fine, but redundant if you already have nDCG.

⚠️ **MRR**
Only include if you want comparability to speech-to-text retrieval literature.

I would not report all five unless necessary. Reviewers prefer clarity over metric buffet.

---

# 2️⃣ Per-Axis Evaluation

Now let’s tighten each axis.

---

## 🟢 Semantic Axis

> Define "correct": same sentence/content, different speaker

✅ Keep exactly as written.

This is your cleanest axis.

🔧 Important:
Make sure:

* Same sentence, same speaker is excluded for this axis.
* Different sentence, same speaker is excluded.

You want pure content similarity.

This axis will anchor the whole paper.

---

## 🟢 Speaker Identity Axis

> Define "correct": same speaker, different utterance

✅ Keep.

This is standard speaker retrieval.

🔧 Important:
Exclude:

* Same utterance, same speaker
* Otherwise trivial.

This axis should clearly diverge from semantic axis.

---

## 🟢 Gender Axis

This is good but be cautious.

⚠️ Gender retrieval can become trivial if the encoder strongly clusters by speaker and speakers are gender-homogeneous.

🔧 Suggestion:
Use gender as:

* A secondary analysis axis
* Or clustering evaluation
* Not a central claim

You don’t want reviewers to think:

> “This is just demographic clustering.”

Keep it, but don’t overemphasize.

---

## 🟢 Accent Axis (TTS corpora only)

✅ Strong axis on VCTK / GB&I.

Important refinement:

* Do not use LibriVox accent unless you have strong labels.
* Keep accent evaluation to controlled corpora.

Accent retrieval should:

* Exclude same speaker
* Exclude same utterance

So it isolates accent from speaker.

This is a strong, clean experimental axis.

---

## 🟡 Fluency Axis (LibriVox)

This is where scope can explode.

### Strong recommendation:

Start with **categorical bins**, not continuous regression.

Define bins using simple acoustic proxies:

* Pause rate tertiles
* Speech rate variance tertiles
* Disfluency count bins

Then:

> "correct" = same bin, different speaker/utterance

This keeps evaluation retrieval-style.

Continuous correlation approaches:
❌ Too messy for this paper.

Keep it simple.

---

## 🟡 Prosody Axis

Be careful here.

Prosody is hard to operationalize without:

* F0 extraction
* Rhythm modeling
* Style embedding

This risks becoming a separate paper.

### Recommendation:

Either:

* Merge prosody into fluency/production axis
  OR
* Make it exploratory only.

Do not make it a full core axis in v1.

---

# 3️⃣ Implementation Notes

Your checklist is good. Add these refinements:

---

## 🔧 Multi-Correct Handling

Important for all non-semantic axes.

For each query:

* Precompute set of valid target IDs.
* Ensure evaluator handles variable cardinality.
* Validate on toy data first.

This is essential.

---

## 🔧 Mixed Retrieval Task (Add This!)

You are missing the key experiment that justifies vector similarity.

Add:

### Mixed Task:

Define "correct" = same content OR same speaker.

Now evaluate:

* Scalar similarity
* Vector similarity (weighted)

Show:

Scalar fails.
Vector succeeds.

This is the paper’s justification.

---

# 4️⃣ Paper Framing Improvements

Add these explicit clarifications:

* "Correct answers per axis may exceed one; therefore we use ranking-based multi-label retrieval metrics."
* "All retrieval is speech-to-speech; text is used only for supervision."

And very important:

Include a small confusion matrix or correlation table showing:

* Content axis similarity vs speaker axis similarity correlation

This visually proves axes are distinct.

---

# 5️⃣ What To Cut (For Scope Safety)

I would recommend:

* Keep: Semantic, Speaker, Accent
* Add: Fluency (simple bin version)
* Defer: Prosody (unless trivial)

Do not try to implement 6 fully polished axes in first submission.

---

# 6️⃣ Big Strategic Advice

If this checklist is implemented exactly as written, you will have:

* A legitimate retrieval benchmark
* Axis-specific evaluation
* Strong technical framing
* Minimal architectural overhead

-----

Good.

That simplifies things dramatically — and, honestly, makes your paper cleaner.

If you commit to:

> ❌ No fine-tuning of backbone encoders
> ❌ No modifying codec models
> ✅ Fully off-the-shelf feature extractors

then your contribution becomes:

> A multi-axis similarity formulation + projection layer + evaluation framework.

That is much more defensible than half-fine-tuned encoders.

Now let’s re-ground the plan around that constraint.

---

# 1️⃣ Architecture Under “Frozen Encoder” Constraint

Your system becomes:

```
Audio
  ↓
Frozen encoder (HuBERT / Whisper encoder / wav2vec2 / codec LM)
  ↓
Pooling
  ↓
Shared embedding z_base
  ↓
Axis-specific projection heads
```

Important:

The encoder is fixed.
Only small projection heads are trainable.

That’s very safe.

---

# 2️⃣ What This Means for “No Starting Point”

You do have a starting point:

The pretrained encoder representation space.

Your job is not to learn speech representations.

Your job is to:

> Reorganize and expose what information is already there.

That’s actually a very elegant thesis.

---

# 3️⃣ How Training Works Now

You train only:

* `W_content`
* `W_speaker`
* `W_accent`
* `W_fluency`

Each can be:

* Linear layer (even better for stability)
* Or small 2-layer MLP

Losses:

* Content → contrastive on projected embedding
* Speaker → cross-entropy
* Accent → cross-entropy
* Fluency → cross-entropy or regression

Encoder stays frozen.

That dramatically reduces risk.

---

# 4️⃣ Why This Is Still Legitimate

A reviewer might ask:

> “If you don’t fine-tune the encoder, what are you contributing?”

Your answer:

We show that:

* Existing encoders already encode multiple axes.
* A simple axis-specific projection can expose them.
* Similarity becomes vector-valued.
* Speech-to-speech retrieval benefits.

That is a representation analysis + modeling contribution.

Perfectly acceptable at Interspeech.

---

# 5️⃣ Even Simpler Alternative (If You Want Maximum Safety)

You could even:

* Skip training projection heads.
* Compute raw embedding once.
* Compute axis similarity using supervised probes only.

But that weakens the method section.

I would recommend training small projection heads.

They’re low-risk.

---

# 6️⃣ Why Frozen Encoders Are Strategically Smart

Advantages:

✔ No compute arms race
✔ No instability
✔ No reviewer complaints about insufficient fine-tuning
✔ Easy reproducibility
✔ Clear comparison across encoder families
✔ Clean ablation: raw vs projected

It keeps focus on:

> Similarity geometry.

---

# 7️⃣ Practical Training Recipe (Low Risk)

* Extract utterance-level embeddings for all data first.
* Save to disk.
* Train projection heads on saved embeddings.
* No backprop through encoder.
* Fast iteration.

This also lets you:

* Swap encoders easily.
* Compare them fairly.
* Avoid GPU memory issues.

---

# 8️⃣ This Makes Word-Level Easier Too

For word-level:

* Extract frame-level embeddings.
* Freeze.
* Pool spans.
* Train projection heads on pooled span embeddings.

No WhisperX gradients.
No encoder tuning.

Clean.

---

# 9️⃣ What Changes in Your Evaluation Framing

You can now claim:

> We do not modify pretrained encoders; instead, we analyze and reorganize their representations under a multi-axis similarity framework.

That sounds careful and principled.

---

# 🔟 Important Conceptual Upgrade

Your work becomes less about:

> “Learning better speech embeddings”

and more about:

> “Revealing multidimensional similarity structure in existing speech encoders.”

That is intellectually sharper.

---

# 11️⃣ You Just Removed 70% of the Risk

No encoder fine-tuning means:

* No catastrophic forgetting
* No complex learning rate schedules
* No delicate loss balancing
* No exploding gradients
* No compute surprises

This is a good decision.

---

# 12️⃣ One Final Clarifying Question

Under this frozen setup, are you comfortable:

* Training projection heads on stored embeddings only?
* Or do you prefer fully zero-training (pure evaluation)?

Because training small heads is very safe — but zero-training is even safer.

---

If you’d like, I can now:

* Redesign your evaluation plan under a strictly frozen-encoder regime, or
* Sketch the exact minimal projection-head architecture and training loop that keeps everything stable and simple.


-----
[BEST-STD]

This is very relevant to your **speech sentence embedding** direction — not because it *is* a sentence embedding model, but because it solves a closely related structural problem:

> How do we turn speech into a sequence of discrete, speaker-invariant, semantically meaningful units that are efficient to index and compare?

The BEST-STD paper gives you several pieces that map directly onto your embedding ambitions.

---

# 1️⃣ What BEST-STD Actually Contributes (Mechanically)

From the paper :

### Core pipeline

1. **Frame-level encoder (Bidirectional Mamba)**
2. **DTW alignment between same-word utterances**
3. **Contrastive loss at frame level**
4. **Vector quantization → discrete tokens**
5. **Bigram indexing + Jaccard similarity for retrieval**

Key design choice:

> They do *not* learn word-level embeddings.
> They learn **consistent frame-level representations**, then discretize.

That distinction matters for you.

---

# 2️⃣ Why This Is Directly Relevant to a Sentence Embedding Model

Your goal:

> Speech equivalent of SentenceTransformers — i.e., a semantic vector for whole utterances.

BEST-STD gives you three important insights:

---

## Insight 1: Frame Consistency > Word Embedding Segmentation

They explicitly avoid word segmentation and instead align utterances with DTW at frame level (Sec II-B) .

For sentence embeddings, this suggests:

* You don’t need explicit segmentation.
* You need **stable local representations that align across paraphrases / repetitions**.
* Sentence embeddings could be built *on top of* a stable frame-token layer.

This is extremely CLIP-like structurally.

---

## Insight 2: Discrete Tokens as Retrieval Substrate

They demonstrate that discrete tokens:

* Are more speaker invariant
* Enable inverted index retrieval
* Work better than continuous DTW features (Table II) 

That is a big deal.

Because:

If discrete tokens are already consistent across speakers for words,
then **a bag-of-token or pooled representation might already be a decent sentence embedding baseline**.

You could test:

* Mean pooling over token embeddings
* IDF-weighted pooling
* Bigram histogram vectors
* Learned projection over token histograms

This is a *very low engineering* path to a first speech embedding baseline.

---

## Insight 3: Bidirectional Mamba > Transformer for Local Temporal Fidelity

They argue (Sec IV) that bidirectional Mamba works better because:

> Fine-grained local temporal modeling matters more than global attention for STD 

That may generalize to:

* Short sentence embeddings
* Keyword-level semantics
* Spoken semantic similarity

For long discourse embeddings, maybe less so — but for your sentence-level work, this is relevant.

---

# 3️⃣ Where This Connects to Your Specific Project

You’ve been thinking about:

* Speech LM tokenization
* Codec tokens
* Sentence embedding analogue
* Possibly multi-distance vector representations

Here’s how this paper intersects:

| Your Direction                 | BEST-STD Relevance       |
| ------------------------------ | ------------------------ |
| Speech as token sequence       | ✔ Yes                    |
| Avoid text ASR                 | ✔ Yes                    |
| Self-supervised alignment      | ✔ Yes                    |
| Word vs sentence granularity   | ✔ Frame-level foundation |
| Retrieval-style objective      | ✔ Jaccard ranking        |
| CLIP-like contrastive training | ✔ Frame contrastive loss |

But:

They do **same-word contrastive training**.
You likely want:

* Same-sentence paraphrase
* Same-meaning across speakers
* Possibly speech–text contrastive

So their training signal is narrower than yours would be.

---

# 4️⃣ The Big Conceptual Difference

BEST-STD optimizes:

> Invariance of token sequences for identical spoken terms.

A sentence embedding model must optimize:

> Invariance of global representation for semantically equivalent utterances.

That is a stronger condition.

However:

Their frame-level contrastive mechanism + quantization might be an excellent base encoder.

You could:

1. Remove VQ.
2. Add pooling head.
3. Train with sentence-level contrastive loss.
4. Optionally add discrete bottleneck later.

---

# 5️⃣ A Concrete Architecture Bridge

Here is a natural evolution from BEST-STD to Speech Sentence Embeddings:

### Step 1 — Use their encoder

Bidirectional Mamba (or similar speech SSM)

### Step 2 — Keep frame embeddings continuous

Skip VQ for now.

### Step 3 — Pool

Try:

* Mean pooling
* Attention pooling
* Learnable query pooling
* Multi-scale pooling (this might match your “vector of distances” intuition)

### Step 4 — Contrastive objective

Instead of DTW frame alignment:

Use:

* Same sentence, different speaker
* Augmented version
* Speech–text pairs (CLIP style)
* Speech–speech paraphrase

Now you have:

> Speech Sentence Transformer analogue

---

# 6️⃣ The Most Interesting Hidden Angle

Table I (Jaccard similarity)  shows:

BEST-STD tokens are much more consistent than HuBERT, WavLM, EnCodec.

That suggests something important:

> General SSL speech models are not optimized for *token consistency across identical lexical items*.

That is exactly the kind of insight that could justify your embedding paper.

You could argue:

> Existing speech SSL models optimize masked prediction or reconstruction, not semantic invariance at utterance level.

This paper gives empirical evidence in a related task.

---

# 7️⃣ What You Should Extract for Your Paper

From BEST-STD you can borrow:

* Framing: discrete speech tokens for efficient retrieval
* Argument: token consistency across speakers matters
* Empirical demonstration that invariance is not guaranteed in current SSL models
* Bidirectional SSM as viable encoder backbone

But you would shift:

STD → semantic similarity
Jaccard ranking → vector similarity
Term-level invariance → utterance-level invariance

---

# 8️⃣ The Strategic Question

Are you thinking of:

A) Building on this tokenization idea and moving upward to sentence embeddings?

or

B) Ignoring tokenization and going directly for pooled continuous embeddings?

Those lead to two different research identities:

* **Discrete speech representation research**
* **Speech semantic embedding research**

They overlap but are not the same community.

---

If you'd like, I can:

* Sketch a minimal “Speech Sentence Transformer” design grounded in this paper
* Compare this against codec-token LLM-style approaches
* Or map this onto your SPEAKABLE / SIGUL positioning

Where do you want to go next?
