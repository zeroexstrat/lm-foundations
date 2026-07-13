# Attention As Kernel Operator Solutions

## Entropy-Regularized Derivation

1. Solve `max_{p in Delta^{T-1}} p^T s + H(p)`, where `H(p) = -sum_j p_j log p_j` and `Delta^{T-1}` is the probability simplex.
2. The Lagrangian derivative is `s_j - (1 + log p_j) - lambda = 0`, so `p_j = exp(s_j - lambda - 1)`. Normalizing over `j` gives `p_j = exp(s_j) / sum_l exp(s_l)`.
3. A causal mask removes future positions from the simplex by forcing their probability mass to zero before normalization.

## Exact Attention Diagnostics

4. A row-stochastic attention matrix has nonnegative entries and every row sums to one.
5. Entropy is minimized by a one-hot row, giving `0`, and maximized by the uniform row over `T` valid positions, giving `log T`.
6. Sharp pattern: one row like `[0.99, 0.01, 0.0, 0.0]`. Diffuse pattern: one row like `[0.25, 0.25, 0.25, 0.25]`.

## Numerical Checks

7. Stabilization subtracts the row maximum before exponentiation, which prevents overflow and keeps the normalized weights numerically meaningful.
8. Masking after softmax would zero some entries without renormalizing the row, so the row would stop summing to one and the operator would no longer be a valid probability kernel.
9. It compares autograd gradients to central-difference numerical gradients, which helps catch silent mistakes in the exact-attention implementation before using it as the baseline for later approximations.

## Small Experiment Design

10. Use one fixed query against four keys, then multiply the query by increasing scale factors; the entropy should drop as the distribution sharpens, but each row should still sum to one.
11. One useful statistic is the row maximum or effective support size `exp(H(p))`, because both shrink toward the one-hot limit as attention concentrates.
12. Any efficient approximation should preserve causal zeros, near-unit row sums, and small output error relative to the exact attention operator on the same inputs.
