Speech, in contrast with text, is inherently variable.
We never say the same thing the same way twice: different speakers say the same thing differently, and even the same speaker may produce the same utterance differently at different times, due to factors such as mood, context, environment, and communicative intent.
Beyond linguistic content, speech reveals much about the speaker: most fundamentally, it can reveal their identity, sex, and even aspects of their health \cite{nolan1995can}.
These attributes are often intertwined, and disentangling them within a single representation is difficult.

The prompts used when constructing text-to-speech datasets are typically selected to be phonetically balanced \cite{kominek2004cmu}.
In multi-speaker datasets there is therefore often substantial overlap between the sentences read by different speakers.
This overlap provides a natural testbed for evaluating attribute-conditioned similarity: given two utterances of the same sentence, an ideal representation should allow retrieval based either on *what* was said or *who* said it.

We introduce a factor-partitioned embedding framework for speech that learns a single representation space partitioned into subspaces corresponding to different attributes.
Given an utterance, the model produces a concatenated embedding vector whose subregions are specialised for different factors such as semantic content and speaker identity.

Similarity in this framework is treated as a vector rather than a scalar.
Instead of computing a single similarity score between utterances, we compute per-axis similarities and combine them with a weighted sum.
This allows retrieval to be conditioned on different attributes: similarity can be computed jointly over multiple axes, or one attribute can be suppressed to emphasise another.
For example, assigning a negative weight to the speaker axis discourages same-speaker matches and surfaces semantically similar utterances spoken by different speakers.