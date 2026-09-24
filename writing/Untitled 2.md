Yes — the Meta/LeCun paper is **Balestriero & LeCun, “Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods”**. It explicitly frames VICReg, SimCLR, Barlow Twins, etc. as spectral embedding methods on graphs/manifolds.  

  

VICReg is **Bardes, Ponce & LeCun, “Variance-Invariance-Covariance Regularization for Self-Supervised Learning”**; its relevance here is that it directly targets collapse with variance and covariance regularization, rather than relying on negatives or architectural tricks.  

  

A possible follow-up framing:

  

The central observation from the speech experiments is that semantic similarity alone is not sufficient to learn a useful speech representation: when trained only against a semantic teacher, the model collapses to nearly uniform embeddings. This suggests that the failure is not merely an implementation issue, but a geometric one. The acoustic representation space is dominated by nuisance factors such as speaker identity, so semantic gradients from different speakers interfere rather than align.

  

This motivates moving beyond InfoNCE-style objectives toward self-supervised objectives that make collapse avoidance explicit. VICReg is especially relevant because it separates three requirements that are conflated in many contrastive formulations: invariance between matched views, variance preservation across the batch, and covariance reduction across dimensions. In the speech setting, this provides a natural way to ask whether semantic alignment can be learned without allowing the representation to collapse when speaker variation dominates the input space.

  

The broader theoretical connection is provided by Balestriero and LeCun’s spectral interpretation of self-supervised learning. Under this view, SSL objectives are not simply heuristic similarity losses; they recover spectral embeddings of a graph whose edges encode the positive-pair relation. This is useful for the present work because the choice of graph is precisely the problem: semantic pairs, speaker pairs, and joint motion–speech pairs define different neighbourhood structures. Collapse under semantic-only training can therefore be interpreted as a failure of the semantic graph to induce a recoverable geometry over the pooled speech features, while auxiliary speaker structure supplies an additional graph that stabilises the embedding.

  

This fits your current paper well because your uploaded draft already reports semantic-only collapse, speaker-auxiliary rescue, and GRL failure as core findings.

-----

I think that’s actually a stronger story than the original one.

The original hypothesis was essentially:

Learn separate similarity geometries, one per factor, and then combine them at query time.

What your results suggest is slightly different:

Learn a single representation whose geometry is induced by multiple relational graphs, with different factors contributing different spectral structure.

That fits remarkably well with the spectral SSL interpretation.

In the current paper, you already have evidence that the semantic graph alone is insufficient. The semantic-only objective collapses, while adding speaker supervision creates enough structure for semantic retrieval to emerge.  

Viewed spectrally, the speaker objective is not merely “another task”. It supplies additional graph structure that stabilizes the eigenspace being learned. The semantic signal is then recoverable because the representation is no longer trying to organise itself around a single weak relational graph.

That suggests a natural progression:

1. **Paper 1 (current)**:  
    Multi-axis retrieval and the discovery that auxiliary structure prevents collapse.
2. **Paper 2**:  
    Investigate collapse avoidance directly through VICReg-style objectives and spectral graph interpretations.
3. **Paper 3**:  
    Generalise from binary similarity to graded similarity via augmentations.

The augmentation angle is particularly interesting because speech is awkward compared with vision. In images, augmentations define “different views of the same object”. In speech, there are several plausible notions of “same utterance”:

|**Augmentation**|**Preserves**|
|---|---|
|Random frame crop|Speaker, local phonetics|
|VAD crop|Speaker, prosody, partial content|
|Word-aligned crop|Semantic fragment|
|Pitch shift|Content, speaker partially|
|Speed perturbation|Content, speaker mostly|
|Noise/reverb|Content, speaker|

Instead of a binary positive/negative relation, you effectively get a **continuum of similarity strengths**.

For example:

- Full utterance ↔ full utterance = similarity 1.0
- Word-aligned crop ↔ original = 0.8
- VAD segment ↔ original = 0.6
- Random frame crop ↔ original = 0.3

This is where VICReg becomes more attractive than InfoNCE.

InfoNCE ultimately wants:

x_i \leftrightarrow x_j

and everything else apart.

VICReg only requires:

- invariance between views,
- variance preservation,
- decorrelation.

That means you can replace the binary “same/not-same” relation with something more graph-like:

- strong edges for word-based crops,
- medium edges for VAD segments,
- weak edges for random crops,
- stronger speaker edges within an utterance,
- semantic edges across speakers.

Then the objective becomes less about finding positives and negatives and more about recovering a useful embedding of a weighted graph.

The really interesting connection to your current collapse result is that the collapse may be telling you something about the graph itself:

The semantic graph induced by sentence labels is too sparse or too inconsistent relative to the acoustic manifold.

Speaker identity creates dense local connectivity, making the graph easier to embed. Once that structure exists, semantic organisation can emerge on top of it.

That would let you reframe the current finding from:

“Semantic distillation collapses but speaker supervision fixes it”

to

“Certain relational graphs are insufficient to induce a stable spectral embedding of speech representations; auxiliary graphs provide the structural constraints needed for semantic organisation to emerge.”

That’s a much more theoretically interesting claim, and it connects the speech results directly to both VICReg and the spectral SSL literature.

---
Demonstrate it as a **graph diagnostic**, not just a retrieval result.

You want to show that adding the speaker objective changes the learned representation from “collapsed / spectrally degenerate” to “structured in a way that preserves semantic edges.”

A clean experiment:

**1. Build three target graphs on the same utterances**

Let nodes be utterances.

- G_{sem}: edge if same sentence / same transcript / high text-teacher similarity.
- G_{spk}: edge if same speaker.
- G_{mix}: weighted union, e.g.  
    A_{mix} = \alpha A_{sem} + \beta A_{spk}

**2. Train models against different objectives**

- semantic-only
- speaker-only
- semantic + speaker
- semantic + speaker with GRL
- VICReg semantic augmentations
- VICReg semantic + speaker / augmentation graph

Your current paper already has the first four ingredients: semantic-only collapse, speaker-supervised success, and GRL failure.  

**3. Compare learned embedding geometry to graph geometry**

For each trained model, compute:

- embedding similarity matrix K = ZZ^\top
- graph Laplacian eigenvectors of G_{sem}, G_{spk}, G_{mix}
- alignment between K’s principal eigenspace and each graph eigenspace

The punchline metric:

Does the semantic+speaker model have higher eigenspace alignment with G_{sem} than the semantic-only model?

If yes, that supports the claim that speaker structure stabilizes the semantic geometry.

**4. Add collapse diagnostics**

Report:

- embedding variance per dimension
- effective rank of covariance
- mean pairwise cosine
- cosine standard deviation
- spectral entropy

Expected pattern:

|**Model**|**Effective rank**|**Mean cosine**|**Semantic graph alignment**|**Retrieval**|
|---|---|---|---|---|
|semantic-only|low|very high|low|bad|
|speaker-only|medium/high|normal|low/medium|speaker-good|
|semantic+speaker|high|normal|high|semantic-good|
|sem+speaker+GRL|low/unstable|high|low|bad|

**5. Use augmentation graphs for VICReg**

For cropping, define different edge types:

- frame crop ↔ full utterance
- VAD crop ↔ full utterance
- word crop ↔ full utterance
- same transcript across speakers
- same speaker across utterances

Then test whether each edge type induces different local structure in the learned graph.

The most convincing demonstration would be:

word-based crops improve semantic graph alignment; frame/VAD crops improve speaker/local-acoustic alignment; combining them produces a representation with higher effective rank and better semantic retrieval than semantic-only training.

So the claim becomes empirical:

The speaker objective stabilizes learning because it increases the spectral quality of the induced utterance graph: the learned kernel becomes higher-rank, less collapsed, and more aligned with semantic graph eigenvectors.

That is much stronger than saying “speaker supervision helps.”

---
