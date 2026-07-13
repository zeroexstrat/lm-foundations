# Linear And Kernelized Attention Exercises

## Associativity

1. Starting from the unnormalized kernel attention sum `sum_j k(q_i, k_j) v_j`, show the algebraic step that lets you regroup it into `phi(q_i)^T (sum_j phi(k_j) v_j^T)`.
2. Why does the regrouping above fail for softmax attention if you insist on computing it exactly with the row-wise normalized `T x T` matrix?
3. What property must the feature map `phi` satisfy for the linear-attention denominator to stay positive?

## Causal Prefix Computation

4. Define the causal prefix state `S_i` for the numerator and `z_i` for the denominator.
5. Why does causal linear attention need a running prefix sum instead of one global sum over all keys?
6. If `phi(k_i)` has dimension `m` and `v_i` has dimension `d_v`, what is the shape of the prefix state `S_i`?

## Approximation Quality

7. The notebook compares exact softmax attention against positive random features. What quantity is actually being approximated by the random features?
8. Why can approximation error decrease with larger feature dimension even when the algorithm stays linear in sequence length?
9. Give one task where a low-error average output approximation could still hide a bad failure on a specific token.

## Experimental Design

10. Why is it useful to compare the deterministic `elu(x) + 1` kernel path against an explicit kernelized attention reference before comparing random features to exact softmax?
11. Name one reason a random-feature benchmark can look worse than the theory even when the implementation is correct.
12. State one concrete check that supports the claim that the linear reference path never materializes a full `T x T` attention matrix.
