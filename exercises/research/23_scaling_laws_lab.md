# Scaling Laws And Empirical Science Exercises

## Power-Law Fitting

1. Start from `L(N) = L_inf + A N^{-alpha}`. Show how subtracting `L_inf` and taking logs turns the fit into a line in `log N`.
2. Why does the transformation above fail if `L_inf` is greater than or equal to one of the observed losses?
3. In words, what do `A`, `alpha`, and `L_inf` each control in the curve?

## Bootstrap Intervals

4. What question does the bootstrap answer that a single best-fit exponent does not?
5. Why is resampling with replacement reasonable for a small synthetic scaling sweep?
6. Suppose a bootstrap interval for `alpha` is very wide. What does that imply about the strength of the empirical claim?

## Compute Accounting

7. Write the standard dense-training compute proxy `C ~= 6 N D` and define each symbol.
8. If `N = 70M` parameters and `D = 1.4B` tokens, what compute budget does the proxy assign?
9. Under the `20 tokens per parameter` Chinchilla-style heuristic, how many tokens would a `280M` parameter model want?
10. Keeping compute fixed, why must increasing parameters reduce tokens under `C ~= 6 N D`?

## Extrapolation Judgment

11. Give one reason a good fit over six small models can still fail at a frontier scale.
12. Why does a regime change break naive power-law extrapolation even if the local fit residuals are tiny?
13. Name two factors besides parameter count that can shift a measured scaling curve.

## Modeling Practice

14. Why is it useful to compare a full-range fit against a fit that only sees the smallest few scales?
15. State one concrete rule you would use before trusting a scaling-law extrapolation in a real project.
