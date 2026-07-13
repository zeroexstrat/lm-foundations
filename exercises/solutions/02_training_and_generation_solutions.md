# Training And Generation Solutions

## Shape Reasoning

1. Each input position predicts the next token, so targets are the same sequence shifted left by one.
2. Logits should have shape `(B, T, V)`.

## Derivation Checks

3. AdamW updates parameters using Adam's adaptive moment estimates, then applies decoupled weight decay directly to the parameter value rather than adding the penalty into the gradient.
4. Average negative log-likelihood is the negative log probability per token, so exponentiating returns the inverse geometric mean probability assigned to the correct token.

## Implementation Gaps

5. `idx = torch.cat((idx, next_token), dim=1)`.
6. `logits = top_k_filter(logits, top_k)`.

## Debugging

7. Validation loss may look artificially strong because the model can read future labels during training.
8. Likely causes include no gradient flow, labels not aligned with inputs, optimizer not stepping, learning rate too low or too high, model left in eval mode with unexpected behavior, or all targets being identical by mistake.

## Extension

9. Higher temperature flattens the distribution, increasing randomness and lowering confidence in the top token.
10. Top-k keeps a fixed number of candidates; top-p keeps the smallest high-probability prefix whose cumulative probability exceeds `p`.
11. A tiny overfit test proves the model, loss, gradients, and optimizer can fit a memorization problem before larger data introduces noise.
