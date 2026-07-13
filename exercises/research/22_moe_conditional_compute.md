# Mixture-of-Experts And Conditional Computation Exercises

## Routing Algebra

1. Let `p_t = softmax(z_t)` over `E` experts. Define the top-k selected set `S_t`. Write the sparse mixture weights `w_{t,e}` for experts inside and outside `S_t`.
2. For `T` tokens, `E` experts, top-k routing, and capacity factor `c`, derive the capacity heuristic `C = ceil(c * T * k / E)`. What average load is this trying to approximate?
3. Why does the normalization of the selected top-k probabilities matter after routing?

## Capacity And Overflow

4. Suppose `T = 12`, `E = 4`, `k = 1`, and `c = 1.25`. What capacity does each expert receive?
5. If every token chooses expert `0` under the setting above, how many assignments are accepted and how many are dropped?
6. Why is dropping overflow assignments usually better than silently overfilling an expert when reasoning about throughput and memory?

## Load Balancing

7. In words, what failure mode does load CV^2 detect?
8. The notebook uses a Switch-style post-routing diagnostic `E * sum_e mean(p_e) * f_e`. What does it mean qualitatively when this quantity is much larger than `1`?
9. Give one reason a router might collapse onto one expert even if several experts could do nearly as well on the task.

## Compute Accounting

10. A two-layer expert MLP with width pair `(d_model, d_hidden)` has about `2 d_model d_hidden` weights. Write formulas for:
    - total expert parameters with `E` experts;
    - active expert parameters per token with top-k routing.
11. Why is it incorrect to say that an MoE with more total parameters necessarily does more FLOPs per token?
12. Give one situation where lower active FLOPs still might not produce a proportional wall-clock speedup.

## Modeling Judgment

13. Name one benefit of top-2 routing compared with top-1 routing.
14. Name one cost or risk of top-2 routing compared with top-1 routing.
15. Why does a perfectly balanced router not automatically imply a good MoE model?
