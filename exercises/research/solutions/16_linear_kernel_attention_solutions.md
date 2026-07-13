# Linear And Kernelized Attention Solutions

## Associativity

1. If `k(q_i, k_j) = phi(q_i)^T phi(k_j)`, then
   `sum_j k(q_i, k_j) v_j = sum_j phi(q_i)^T phi(k_j) v_j = phi(q_i)^T (sum_j phi(k_j) v_j^T)`.
2. Exact softmax attention normalizes each query row after forming all pairwise logits, so the row-wise normalization depends on the full query-key interaction pattern rather than one shared associative summary.
3. The feature map should be elementwise nonnegative, and ideally strictly positive after smoothing with `eps`, so the denominator `phi(q_i)^T sum_j phi(k_j)` cannot flip sign.

## Causal Prefix Computation

4. A standard choice is `S_i = sum_{j <= i} phi(k_j) v_j^T` and `z_i = sum_{j <= i} phi(k_j)`.
5. Causality only allows each query to use keys from its prefix, so the sufficient statistics must stop at position `i` instead of including future tokens.
6. `S_i` has shape `(m, d_v)` because each term is the outer product of an `m`-dimensional feature vector with a `d_v`-dimensional value vector.

## Approximation Quality

7. The random features approximate the exponential kernel `exp(q_i^T k_j / sqrt(d))`, which is the unnormalized part of softmax attention.
8. Increasing feature dimension reduces Monte Carlo variance in the kernel estimate while keeping the algorithm linear in sequence length for fixed feature dimension.
9. A rare exact-copy or far-token retrieval task can fail at one critical position even if the mean absolute output error over the whole sequence looks small.

## Experimental Design

10. That check separates two questions: whether the linear algebra implementation is correct for a chosen kernel, and whether the chosen kernel approximates exact softmax well.
11. Common reasons include high variance from too few features, unstable scaling for large query/key norms, or evaluating on a task that is especially sensitive to small weight perturbations.
12. Record the named intermediates from the linear path and assert that none of them has trailing shape `(T, T)` while the algorithm still matches an explicit small-reference computation.
