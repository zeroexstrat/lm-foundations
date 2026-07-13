# Training And Generation Exercises

## Shape Reasoning

1. Explain why the target tensor is shifted one position relative to the input tensor.
2. If `idx` has shape `(B, T)`, what shape should logits have before cross-entropy?

## Derivation Checks

3. Write the AdamW parameter update in words, including how weight decay differs from L2 penalty inside the gradient.
4. Explain why perplexity is `exp(loss)` when loss is average negative log-likelihood.

## Implementation Gaps

5. Write the one-line tensor operation that appends `next_token` with shape `(B, 1)` to `idx` with shape `(B, T)`.
6. Add a top-k filter call before sampling from logits.

## Debugging

7. Describe one symptom of causal mask leakage.
8. A tiny overfit test does not reduce loss. List three likely causes.

## Extension

9. Increase temperature from `0.7` to `1.3`. Predict how generations should change.
10. Compare top-k and top-p sampling in one paragraph.
11. Explain why a tiny overfit test is a useful debugging tool.
