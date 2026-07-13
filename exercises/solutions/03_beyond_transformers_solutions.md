# Beyond-Transformers Solutions

## Complexity Reasoning

1. Full attention computes `T^2` scores per head; sliding-window attention computes roughly `T*w` scores.
2. Full attention computes `8192^2 = 67,108,864` scores; sliding-window attention computes roughly `8192 * 256 = 2,097,152` scores.

## Architecture Tradeoffs

3. Sparse attention can reduce memory or compute; it may also remove useful long-range interactions unless the sparsity pattern is well matched to the task.
4. Grouped-query attention shares keys and values across multiple query heads, reducing the amount of KV-cache data stored and read during decoding.
5. Recurrence or state-space models can avoid materializing a full `T x T` attention matrix and can carry compressed state across long sequences.

## Objective Reasoning

6. Predictive representation objectives can train latent states to predict abstract future representations instead of directly predicting the next discrete token.
7. JEPA-style methods are interesting because they encourage predictive world representations without requiring every target to be reconstructed at pixel or token level.

## Paper Reading

8. Ask what objective is optimized, what bottleneck is targeted, and whether the evaluation supports the architectural claim.
9. Classify the claim by asking what changed: the loss, the architecture's assumptions, the asymptotic or hardware cost, the data source, or the benchmark evidence.
