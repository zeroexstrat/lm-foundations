# Sparse Attention Patterns Exercises

## Mask Semantics

1. In one sentence each, describe what information a local mask, a block mask, and a global-token mask assume is important.
2. Why must every causal sparse mask keep the diagonal entries?
3. Give one task where a pure local window is a good inductive bias and one task where it is a bad one.

## Reachability And Complexity

4. Let a local mask have window size `w`. What is the approximate number of allowed score entries as a function of sequence length `T`?
5. Explain why stacked sparse attention layers can increase receptive field even when each single layer is very local.
6. In the notebook's toy setup, why can a pure dilated mask with dilation `2` fail to retrieve position `0` from the final odd-index token?

## Paper Patterns

7. What role do designated global tokens play in Longformer-style attention?
8. Why do BigBird-style random edges help more than adding extra local edges at the same asymptotic order?
9. Sparse Transformers combine block structure with strided or factorized links. What graph property are those extra links trying to improve?

## Experimental Design

10. Propose a delayed-copy task where local attention should eventually succeed but need more layers than a global-token pattern.
11. Name one metric besides exact retrieval accuracy that helps compare sparse masks on a far-token task.
12. State one failure mode of using only graph reachability as evidence that a sparse attention pattern is good.
