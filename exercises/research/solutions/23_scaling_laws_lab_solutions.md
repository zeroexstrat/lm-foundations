# Scaling Laws And Empirical Science Solutions

## Power-Law Fitting

1. Rearranging gives `L(N) - L_inf = A N^{-alpha}`. Taking logs yields `log(L(N) - L_inf) = log A - alpha log N`, which is linear in `log N` with intercept `log A` and slope `-alpha`.
2. The logarithm is only defined for positive inputs. If `L_inf >= L(N)` for any point, then `L(N) - L_inf <= 0` and the linearized fit is invalid.
3. `A` sets the vertical scale of the reducible loss, `alpha` sets how quickly loss falls with scale, and `L_inf` sets the asymptotic floor the curve approaches.

## Bootstrap Intervals

4. The bootstrap estimates uncertainty: how much the fitted exponent would move if the small sweep had been perturbed or re-sampled.
5. A small sweep gives only a few observations, so replacement sampling is a cheap way to approximate how unstable the fitted curve is under data variation at the same scale.
6. A wide interval means the exponent is weakly identified by the data. The empirical claim is fragile, so any extrapolation built on it should be treated cautiously.

## Compute Accounting

7. `C ~= 6 N D`, where `C` is training FLOPs, `N` is parameter count, and `D` is the number of training tokens. The constant `6` is the common dense-transformer forward/backward proxy.
8. `C ~= 6 * 70,000,000 * 1,400,000,000 = 5.88e17` FLOPs.
9. `280M * 20 = 5.6B` tokens.
10. If compute is fixed, then `D = C / (6 N)`. Increasing `N` raises the denominator, so `D` must fall unless compute also increases.

## Extrapolation Judgment

11. The frontier may differ in data quality, optimization, architecture, or bottlenecks, so the small-scale regime need not continue unchanged.
12. A regime change means the effective slope or floor shifts outside the fitted window. A local power law can therefore fit the observed points well while being wrong about the next regime.
13. Examples include token count, dataset quality, optimizer schedule, architecture changes, regularization, and evaluation contamination.

## Modeling Practice

14. The comparison reveals how much the conclusion depends on the observed range. If the small-range fit and the fuller fit disagree materially at larger scales, the extrapolation is not robust.
15. One reasonable rule is: do not trust the extrapolation unless the fitted interval is narrow, multiple neighboring scale windows agree, and the projected regime matches the intended training recipe and data distribution.
