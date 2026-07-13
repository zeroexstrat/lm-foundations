# State-Space Models Solutions

## Linear Recurrence To Convolution

1. Unrolling the recurrence gives `x_k = sum_{j=0}^k a^{k-j} b u_j` under zero initial state, so `y_k = sum_{j=0}^k c b a^{k-j} u_j`. Therefore the impulse response is `h_l = c b a^l` for `l >= 0`.
2. A nonzero initial state adds an extra homogeneous term `c a^{k+1} x_{-1}`, so the output is no longer only the convolution of the input with the impulse kernel.
3. The coefficients are `3.0`, `1.5`, `0.75`, and `0.375`.

## Diagonal Latent Dynamics

4. Each diagonal mode evolves independently as `a_i^l`, and the scalar readout sums the mode-wise contributions `c_i b_i a_i^l`, so the full kernel is a sum of exponentials.
5. A diagonal transition cannot mix information across latent coordinates; each state channel evolves independently.
6. It isolates the core SSM ideas cleanly: recurrence, impulse response, discretization, and stability can all be studied exactly without hiding the algebra inside a full implementation.

## Discretization

7. It assumes the input stays constant within each timestep interval of width `Delta`.
8. If the continuous transition is zero, then `exp(tau A) = 1` across the interval, so the integral becomes `int_0^Delta 1 dt = Delta`, giving `B_bar = Delta * B`.
9. It is the exact value of the exponential integral `int_0^Delta exp(tau a) dt`, multiplied by `B`.

## Stability

10. The homogeneous solution is proportional to `a^k`, so decay happens exactly when `|a| < 1` and growth happens when `|a| > 1`.
11. `a = 1.02` can retain larger magnitudes over a short horizon, but it is unstable because its magnitude eventually grows without bound. `a = 0.98` decays slowly and is asymptotically stable.
12. The unstable mode eventually dominates the output, so even if other modes decay correctly the total response can grow or oscillate with increasing magnitude.
