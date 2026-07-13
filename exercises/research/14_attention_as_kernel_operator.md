# Attention As Kernel Operator Exercises

## Entropy-Regularized Derivation

1. For one query row with scores `s in R^T`, write the optimization problem over the probability simplex whose solution is `softmax(s)`.
2. Differentiate the Lagrangian with respect to one coordinate `p_j` and solve for `p_j` before normalization.
3. Explain in one sentence how causal masking changes the feasible set of that optimization problem.

## Exact Attention Diagnostics

4. State two conditions that make an attention matrix row-stochastic.
5. Why is row-wise entropy always between `0` and `log T` for a row with `T` valid positions?
6. Give one example of a sharp attention pattern and one example of a diffuse attention pattern.

## Numerical Checks

7. Why does log-sum-exp stabilization matter when logits become large in magnitude?
8. What failure would you expect to see if a causal mask were applied after softmax instead of before it?
9. What does a finite-difference gradient check compare, and why is it useful in a research notebook?

## Small Experiment Design

10. Propose a toy sequence setup where sharper logits should reduce entropy while leaving the row sums unchanged.
11. Name one statistic besides entropy that could help diagnose whether exact attention is becoming nearly one-hot.
12. State one approximation property that any later efficient-attention method should preserve when compared against this exact baseline.
