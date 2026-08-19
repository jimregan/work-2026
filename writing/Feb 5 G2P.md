# Chat Conversation

*Exported on 19/08/2026, 14:04:15*

---

## User

My current project involves training a byt5 model for sentence level G2p
My intention is to show that by being selective in terms of complementary tasks for multi task learning, there can be improvements

---

## Assistant

That's an interesting project! Using ByT5 for grapheme-to-phoneme conversion at the sentence level is a solid choice given its byte-level tokenization, which naturally handles the character-level nature of G2P.

Your hypothesis about selective complementary tasks for multi-task learning makes sense. The key challenge will be identifying tasks that share useful representations with G2P while not introducing conflicting objectives. Some potentially complementary tasks that come to mind:

Phonologically-related tasks:

Orthographically-related tasks:

Cross-lingual tasks:

The effectiveness will likely depend on:

Are you planning to compare against single-task baselines? And do you have specific auxiliary tasks in mind already, or are you exploring what might work? Also, what language(s) are you targeting - that could influence which tasks would be most complementary.

- Stress prediction
- Syllabification
- Phoneme duration prediction
- Homograph disambiguation

- Morphological segmentation
- Grapheme type classification
- Word boundary detection in continuous text

- G2P in related languages (if applicable)
- Transliteration

1. Shared linguistic structure - tasks that require similar phonological or orthographic knowledge
2. Task weighting/scheduling - how you balance the auxiliary tasks
3. Data availability - whether you have quality annotations for auxiliary tasks

---

## User

Well, ok, there are two tasks: one is single word G2p but in a multilingual framework. The other is whole sentence. For that, text normalisation is the key task, but also POS tagging, and for swedish, compound splitting

---

## Assistant

Ah, this is a nice setup! You're essentially testing whether auxiliary tasks help the model better understand the linguistic context needed for sentence-level G2P. The distinction between single-word multilingual G2P and sentence-level with contextual tasks is clever.

Your auxiliary task choices make solid linguistic sense:

Text normalization is crucial for sentence-level G2P - handling numbers, abbreviations, acronyms, dates, etc. This directly impacts the graphemic input before phonemization.

POS tagging helps with homograph disambiguation (READ as present vs. past tense, LIVE as verb vs. adjective), which is one of the main advantages of sentence-level over word-level G2P. The syntactic context is often necessary for correct pronunciation.

Compound splitting for Swedish is particularly smart since:

Some questions about your setup:

This could make for a nice empirical study showing which types of linguistic knowledge transfer best to contextual G2P.

- Swedish compounds affect stress patterns significantly
- Knowing boundaries helps with proper phoneme realization at morpheme edges
- It's a clear case where word-internal structure matters for pronunciation

1. Task architecture - Are you using a unified seq2seq approach where all tasks share the encoder, or are you using task-specific heads/decoders?
2. Training strategy - Simultaneous training on all tasks, or staged/curriculum approach?
3. Evaluation - Beyond phoneme error rate, are you planning to measure specifically on cases where context matters (like homographs, normalized text, compounds)?
4. Baseline comparison - Will you compare against the single-word multilingual model to isolate the benefit of the auxiliary tasks versus just having sentence-level context?

---

## Assistant

That's a thoughtful consideration! Training from scratch gives you full control over what the model learns and avoids potential contamination from pre-training data.

However, I'd encourage you to think carefully about this tradeoff:

1. Linguistic knowledge: Pre-trained ByT5 has learned general patterns about:

2. Data efficiency: G2P datasets are often relatively small. Pre-training provides a strong initialization that could help you reach better performance with less task-specific data.

3. Your hypothesis still holds: You're testing whether selective auxiliary tasks improve G2P. This is about task composition, not about pre-training. You can compare:

4. Practical: Training from scratch requires massive compute and data

5. Clean experiment: No confounding from unknown pre-training data
6. Full control: You know exactly what went into the model
7. Specialized representations: Model learns only what's needed for your tasks

What's driving your concern about "accidental inputs"? Are you worried about data leakage in evaluation, or that pre-training might mask the effects of your auxiliary tasks?

- Character sequences and word structure
- Morphology across languages
- Syntactic patterns (useful for your POS task)
- Cross-lingual regularities

- Fine-tuned ByT5 on G2P alone
- Fine-tuned ByT5 on G2P + your selected auxiliary tasks
- Fine-tuned ByT5 on G2P + poorly-chosen auxiliary tasks (negative control)

- Ablation study: Run both and compare
- Controlled pre-training: Pre-train on generic text in your target languages only
- Document your approach: If you fine-tune, clearly state this and discuss potential implications

---

## User

I did consider comparing with a regular t5, but that would need a task of predicting letters from words to counter act the tokenization

---

## Assistant

Ah, I see the issue! Regular T5's subword tokenization would be problematic for G2P because:

The core problem: T5 might tokenize a word like "through" as a single token, but you need character-level alignment to produce phonemes. The model would need to learn to "unpack" tokens back into characters first before mapping to phonemes.

Your proposed solution (predicting letters from words) is interesting - essentially teaching T5 to do implicit character-level processing. But this adds complexity and might not fully solve the alignment problem.

ByT5's byte-level tokenization means:

If you want to show ByT5's architectural advantages for G2P specifically, comparing against standard T5 makes sense. But you'd need to be careful that you're not just showing "byte-level is better for character-based tasks" (which is somewhat known) versus your actual hypothesis about selective auxiliary tasks.

Cleaner experimental design might be:

Does your project need to justify using ByT5 over T5, or is ByT5 already your chosen architecture and you're focusing on the multi-task aspect?

- Direct character access: Each character is a token
- Natural alignment: One-to-one or one-to-many character-to-phoneme mappings work smoothly
- No tokenization artifacts: No risk of morpheme boundaries being hidden inside subword tokens

- Main comparison: ByT5 with G2P alone vs. ByT5 with G2P + your auxiliary tasks
- Optional baseline: Standard T5 to show byte-level benefits, but keep this separate from your multi-task learning claims

