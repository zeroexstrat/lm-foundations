# State-Space Models Exercises

## Linear Recurrence To Convolution

1. Starting from `x_k = a x_{k-1} + b u_k` and `y_k = c x_k`, derive the causal impulse response `h_l`.
2. Why does zero initial state matter for the clean recurrence/convolution equivalence used in this notebook?
3. For `a = 0.5`, `b = 2`, and `c = 1.5`, write the first four impulse-response coefficients.

## Diagonal Latent Dynamics

4. Suppose `A = diag(a_1, a_2, a_3)`. Why can the scalar output kernel be written as a sum of three exponentials?
5. What limitation does a purely diagonal transition impose on the latent dynamics?
6. Give one reason diagonal models are still useful pedagogically even though they are restrictive.

## Discretization

7. In one sentence, what assumption justifies the zero-order-hold discretization step?
8. Show that when the continuous-time transition is zero, the discrete input gain reduces to `Delta * B`.
9. Why does the term `(exp(Delta a) - 1) / a` appear for nonzero scalar continuous transition `a`?

## Stability

10. Why is the unit circle the relevant stability boundary for the discrete-time scalar recurrence?
11. Compare `a = 0.98` and `a = 1.02`. Which one preserves memory longer over a finite horizon, and which one is asymptotically stable?
12. What failure mode appears when one diagonal mode lies outside the unit circle while the others remain inside it?
