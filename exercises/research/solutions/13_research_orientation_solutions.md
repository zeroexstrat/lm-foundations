# Research Orientation Solutions

## Paper Anatomy

1. Example paper: `[@attention_is_all_you_need]`.
2. One-sentence claim: replacing recurrent sequence processing with multi-head self-attention and positional information can improve translation quality while training more efficiently.
3. Bottleneck: recurrent models limit parallelism and create long path lengths between distant tokens.
4. Mechanism: scaled dot-product attention builds weighted combinations of token representations, and multi-head structure lets the model mix several interaction patterns at once.
5. Change type: primarily an architecture change, while keeping the sequence-to-sequence training objective familiar.

## Evidence And Limits

6. Strongest evidence: the paper reports strong machine-translation results together with lower training cost than prior recurrent systems.
7. Limitation: the evidence mixes several architectural choices at once, so it does not fully isolate which component matters most for each gain.
8. Toy replication: compare an order-blind mean-pooling classifier against an order-aware attention-pooling classifier on a first-token retrieval task where order is the only useful signal.

## Experiment Design

9. Baseline: embed each token, average across positions, and predict the first token from the pooled vector.
10. Primary metric: accuracy on a held-out set.
11. Paired comparison: per-example accuracy delta or per-example negative-log-likelihood improvement between the baseline and the variant on the same evaluation split.
12. Ablation: remove positional embeddings from the attention-pooling variant; if the mechanism is really order-sensitive, that ablation should collapse toward baseline behavior.

## Critique

13. Claim: self-attention can replace recurrence for sequence transduction. Mechanism: token-token weighting plus positional information shortens communication paths. Evidence: translation quality and training-speed results support the claim, but mostly at benchmark scale rather than in tightly isolated controls. Limitation: the paper does not remove the quadratic attention cost, so sequence length remains a systems bottleneck.

## Small Experiment Design

14. Hypothesis: an order-aware pooling variant will beat a mean-pooling baseline on first-token retrieval because only the variant can focus reliably on position 0.
15. Dataset: synthetic integer sequences of fixed length where the label is the first token.
16. Baseline and variant: mean pooling vs attention pooling with positional embeddings.
17. Metric and stop rule: require at least `+0.25` paired accuracy delta and positive paired log-loss gain on the same validation set; stop trusting the toy claim if the ablation without positions performs almost as well as the full variant.
18. Confounder to watch: if the data generator accidentally leaks the label through token frequency or a fixed sentinel token, the comparison stops measuring order sensitivity.
