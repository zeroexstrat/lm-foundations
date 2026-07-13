# First-Principles Exercises

## Shape Reasoning

1. Given logits with shape `(B, T, V)` and targets with shape `(B, T)`, write the flattened shapes used by cross-entropy.
2. If queries have shape `(B, H, T, D)` and keys have shape `(B, H, T, D)`, what shape does `q @ k.transpose(-2, -1)` produce?
3. For a sequence length `T = 2048`, how many attention scores does one head compute?

## Derivation Checks

4. Explain why subtracting `max(logits)` does not change softmax probabilities.
5. Starting from `L = -log softmax(z)_y`, derive `dL/dz_i = p_i - 1[i = y]`.

## Implementation Gaps

6. Write a PyTorch expression that creates a lower-triangular causal mask for `T = 4`.
7. Fill in the missing line: `log_probs = logits - ____`.

## Debugging

8. A model's validation loss is suspiciously low after one step. Explain how future-token leakage could cause this.
9. A softmax call returns `nan` for large logits. Name the numerical stability fix.

## Extension

10. Explain why character tokenization is useful for learning but not ideal for real LLMs.
