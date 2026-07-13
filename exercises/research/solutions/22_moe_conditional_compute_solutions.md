# Mixture-of-Experts And Conditional Computation Solutions

## Routing Algebra

1. If `e in S_t`, then `w_{t,e} = p_{t,e} / sum_{j in S_t} p_{t,j}`. If `e not in S_t`, then `w_{t,e} = 0`.
2. Top-k routing creates `T * k` attempted assignments in total. Uniform average load would therefore be `(T * k) / E` assignments per expert. The factor `c` scales that average to allow slack or impose tighter limits, and the ceiling makes the capacity integral.
3. After dropping all but the selected experts, the surviving probabilities no longer sum to `1`. Renormalization turns them back into a proper convex mixture over the active experts.

## Capacity And Overflow

4. `C = ceil(1.25 * 12 * 1 / 4) = ceil(3.75) = 4`.
5. Expert `0` accepts `4` assignments and drops the remaining `8`. Every other expert accepts `0`.
6. Hard overflow makes the bottleneck explicit. It preserves the throughput or memory budget you are trying to model instead of hiding the imbalance inside an unrealistic expert execution path.

## Load Balancing

7. Load CV^2 detects dispersion in expert usage. It is `0` when all experts carry equal accepted load and grows as routing becomes more uneven.
8. A value much larger than `1` means probability mass and realized assignments are concentrated on a small subset of experts rather than being spread uniformly.
9. A strong initialization bias, optimization shortcut, or slightly better default expert can attract early traffic, and once that expert dominates, the other experts receive too little signal to recover.

## Compute Accounting

10.
    - Total expert parameters: `E * 2 d_model d_hidden`.
    - Active expert parameters per token: `k * 2 d_model d_hidden`.
11. Total parameters count stored weights, while per-token FLOPs count only the experts that execute for that token. Sparse routing can therefore increase stored parameters while keeping active compute bounded by `k`.
12. Routing overhead, cross-device communication, kernel launch latency, and poor expert balance can all reduce or erase the theoretical speedup.

## Modeling Judgment

13. Top-2 routing can let a token combine two expert views instead of relying on one hard winner, which may improve robustness or specialization.
14. Top-2 routing increases active compute, complicates dispatch, and can still overflow capacity if secondary choices bunch up on the same experts.
15. A balanced router can still send tokens to the wrong experts. Good MoE behavior requires both healthy utilization and useful specialization.
