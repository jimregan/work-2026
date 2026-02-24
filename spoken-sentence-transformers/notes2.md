[we don't expect multi-axis similarity to be perfect]

The use case most closely tied with similarity learning is retrieval: in a multi-field database search a better match on the keywords in the title might be expected to rank higher despite a mismatch in the author field;

---

Another potential use case is in indexing material for dialectological research: for a linguist researching English dialects, the relevant axes might be a set of factors based on, for example, the lexical sets of Wells: here, whether or not ``path'' and ``bath'' have the same vowel is relevant, while matters such as speaker identity and gender might not, or might be less so.

---

Although in this work we concentrate on a partition that is to be learned directly from the acoustic input, we do so primarily as a matter of [I need a reason here] -- we fully expect and intend to use inputs derived in other manners from the input and [fed alongside it] in future explorations: even if we can learn to partition based on gender or speaker identity directly from the audio, if supplying instead the output of a classifier gives better results, then that ought to be used instead. For some potential use cases, it will absolutely be necessary to feed other input [I have pitch tracking in mind here]

---

Our work has a degree of superficial similarity to work on disentangled representation learning: our goals, however, are much more modest. We focus on supervised learning of multiple relational axes and train per-axis similarity functions within a shared representation.
