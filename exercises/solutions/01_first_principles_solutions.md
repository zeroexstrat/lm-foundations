# First-Principles Solutions

## Shape Reasoning

1. Logits become `(B*T, V)` and targets become `(B*T)`.
2. The score tensor has shape `(B, H, T, T)`.
3. One head computes `2048 * 2048 = 4,194,304` attention scores.

## Derivation Checks

4. Softmax is invariant to adding or subtracting the same constant from every logit because the common exponential factor cancels from numerator and denominator.
5. The derivative of `log(sum_j exp(z_j))` is `p_i`, while the derivative of `-z_y` is `-1` only for the target index, giving `p_i - 1[i = y]`.

## Implementation Gaps

6. `torch.tril(torch.ones(4, 4, dtype=torch.bool))`.
7. `torch.logsumexp(logits, dim=-1, keepdim=True)`.

## Debugging

8. If the mask allows a position to read future tokens, the model can copy label information from the input rather than learning the conditional distribution.
9. Subtract the maximum logit before exponentiating.

## Extension

10. Character tokenization keeps the mapping transparent, but it creates long sequences and misses reusable subword structure.
