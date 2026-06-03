Yes — the Meta/LeCun paper is **Balestriero & LeCun, “Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods”**. It explicitly frames VICReg, SimCLR, Barlow Twins, etc. as spectral embedding methods on graphs/manifolds.  

  

VICReg is **Bardes, Ponce & LeCun, “Variance-Invariance-Covariance Regularization for Self-Supervised Learning”**; its relevance here is that it directly targets collapse with variance and covariance regularization, rather than relying on negatives or architectural tricks.  

  

A possible follow-up framing:

  

The central observation from the speech experiments is that semantic similarity alone is not sufficient to learn a useful speech representation: when trained only against a semantic teacher, the model collapses to nearly uniform embeddings. This suggests that the failure is not merely an implementation issue, but a geometric one. The acoustic representation space is dominated by nuisance factors such as speaker identity, so semantic gradients from different speakers interfere rather than align.

  

This motivates moving beyond InfoNCE-style objectives toward self-supervised objectives that make collapse avoidance explicit. VICReg is especially relevant because it separates three requirements that are conflated in many contrastive formulations: invariance between matched views, variance preservation across the batch, and covariance reduction across dimensions. In the speech setting, this provides a natural way to ask whether semantic alignment can be learned without allowing the representation to collapse when speaker variation dominates the input space.

  

The broader theoretical connection is provided by Balestriero and LeCun’s spectral interpretation of self-supervised learning. Under this view, SSL objectives are not simply heuristic similarity losses; they recover spectral embeddings of a graph whose edges encode the positive-pair relation. This is useful for the present work because the choice of graph is precisely the problem: semantic pairs, speaker pairs, and joint motion–speech pairs define different neighbourhood structures. Collapse under semantic-only training can therefore be interpreted as a failure of the semantic graph to induce a recoverable geometry over the pooled speech features, while auxiliary speaker structure supplies an additional graph that stabilises the embedding.

  

This fits your current paper well because your uploaded draft already reports semantic-only collapse, speaker-auxiliary rescue, and GRL failure as core findings.