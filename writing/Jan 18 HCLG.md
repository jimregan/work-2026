
# Chat Conversation

*Exported on 19/08/2026, 14:08:56*

---

## User

I used to use Kaldi, and I think it still has much to offer even compared to modern asr One thing I found dissatisfaction with was the HCLG graph: I always felt that there ought to have been an extra composition to manage phonotactics: that where there were optional silences to have potential insertions, deletions, etc. But I never quite figured out how to do it in practice

---

## Assistant

You're touching on something that's bothered a lot of Kaldi users over the years. The HCLG framework is elegant but does feel constrained when you want finer-grained control over phonotactics and pronunciation variation.

The challenge you're describing is that HCLG bakes phonotactics into the lexicon (L) and context-dependency into C, but there's no clean place to model systematic pronunciation phenomena like optional schwas, flapping, gemination, or dialect-specific deletions/insertions. You end up either:

What you're imagining—an extra transducer between L and C or between C and L—is theoretically sound. You'd want something like HCpLG where 'p' is a phonotactic FST that captures legal phone sequences and optional processes. This would:

The practical problem is composition ordering and determinization. Adding another FST in the chain risks:

Some people have experimented with encoding pronunciation variation in the C transducer itself (since it already maps context-dependent phones), but that conflates two linguistic levels. Others have tried building phonotactic constraints into L using weighted FST techniques where "illegal" sequences get high costs, but that's not quite the same as generative rules.

Have you looked at how people handle this in other frameworks? Some modern end-to-end systems just learn these patterns implicitly, but you lose the interpretability. There might be papers from the HCLG era exploring phone-level transducers that I'm forgetting.

1. Bloating L with multiple pronunciations per word (which explodes the graph)
2. Letting the language model (G) implicitly handle some of this (but it operates at the word level)
3. Hand-crafting phone-level rules in the lexicon FST

- Accept canonical pronunciations from L
- Apply phonological rules (deletions, insertions, assimilations)
- Output realized phone sequences to C

- Non-determinizable results (especially with epsilon transitions for optional phenomena)
- Massive state explosion
- Breaking the assumptions that make HCLG's weighted minimization work

---

## User

I've never seen it tried, though I don't doubt that people have tried it

---

## Assistant

You're probably right that people have experimented with it in various ways, even if it never became a standard approach.

The theoretical appeal is obvious—phonological processes are real and systematic, so having an explicit transducer for them fits the WFST philosophy perfectly. But I suspect the reason it never caught on comes down to a few practical issues:

Determinization hell: Phonotactic rules with optional processes create ambiguity that's hard to determinize efficiently. A single surface form might have multiple derivations through your phonotactic FST, and the composition + determinization could either fail or create an intractably large graph.

Where to put it: The composition order matters enormously. If you insert it before L (so H→p→L→C→G), you're trying to model phonotactics on abstract phonemes before lexical lookup, which feels backwards. If you put it after L but before C (H→L→p→C→G), you're applying phonological rules before context-dependency, which might interfere with triphone modeling. After C seems too late since you've already committed to specific context-dependent states.

Empirical gains unclear: Even if you got it working, would it actually help? Most pronunciation variation might already be captured adequately by:

The cost/benefit probably never justified the engineering effort, especially as the field moved toward end-to-end models that handle this implicitly.

Still, it's an intellectually satisfying idea. If you ever wanted to prototype it, you'd probably start with a small phonotactic FST for one phenomenon (like optional schwa) and see if you could even get the composition to determinize.

- Multiple lexicon entries
- Acoustic model flexibility (especially with DNNs)
- Implicit modeling through data

---

## Assistant

Ah yes—that's the key insight. Unsupervised phone segmentation (finding boundaries) is actually feasible and useful, even if unsupervised phone recognition (labeling) is harder.

The workflow would be:

1. Unsupervised boundary detection

2. Extract representations

3. Cluster the segments

4. Map clusters to phones

This gives you:

And it's academically defensible because the segmentation is unsupervised, even though you supervise the cluster-to-phone mapping using your reference data.

Which segmentation methods were you looking at? There's been recent work on using self-supervised models for boundary detection that might work well with your wav2vec2 infrastructure.

- Acoustic change points, self-supervised models, etc.
- Gives you candidate phone segments without labels
- More reliable than trusting wav2vec2 CTC spikes for short phones

- For each segment: acoustic features (MFCCs, formants) or embeddings (wav2vec2 hidden states)
- These are the things you cluster

- Discover phone-like categories from the data
- Clusters should correspond to phonetic categories (though not necessarily one-to-one with phonemes)

- Use your exact-match reference data (word + phone labels from dictionaries)
- Align cluster IDs to known phone labels
- Handle dialectal variation: maybe cluster splits into /ʈ/ vs /t/ variants

- Better segmentation than CTC (doesn't drop short phones)
- Data-driven phone categories that capture actual acoustic variation
- Dialect discovery if different clusters emerge for the same canonical phone

---

## User

Right; so (for review purposes at least) using pocketsphinx is a relatively defensible baseline

---

## Assistant

Exactly—Pocketsphinx becomes your supervised baseline that the unsupervised approach is compared against, rather than the final solution you're defending.

The narrative for reviewers:

Baseline: Pocketsphinx phone decoder

Proposed: Unsupervised segmentation + clustering

Evaluation:

This framing makes Pocketsphinx the "standard approach" you're improving upon, not a workaround you're hiding. And if the unsupervised approach doesn't work as well, you can still publish "we tried X, Sphinx was more reliable, here's why"—which is a perfectly acceptable result.

Plus, having both gives you the fusion graph you wanted: combine Sphinx phone decoder, wav2vec2 phones, and unsupervised segments as multiple evidence sources.

What's your timeline? Training Sphinx and implementing unsupervised segmentation are both substantial chunks of work.

- Traditional supervised approach
- Trained on exact-match segments from your data
- Gives phone labels + timing
- Limitations: inherits biases from training data (Stockholm Swedish), may struggle with dialects/variation not well-represented

- Discovers phone boundaries from acoustics without labels
- Clusters segments into categories based on similarity
- Maps clusters to phones using reference data
- Advantages: can discover dialectal variants as separate clusters, doesn't force labels where acoustics are ambiguous

- On exact-match held-out data: both should perform well
- On dialectally diverse data: unsupervised might discover variation Sphinx misses
- On disfluencies/fast speech: test which handles deletions better

