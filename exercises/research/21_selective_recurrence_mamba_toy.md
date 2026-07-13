# Selective Recurrence And Mamba Toy Exercises

## Gate Algebra

1. Starting from `x_t = g_t a x_{t-1} + (1 - g_t) b u_t`, explain what happens when `g_t` is close to `1` and when it is close to `0`.
2. Assume the gate is constant: `g_t = g` for every timestep. Rewrite the recurrence as a fixed linear recurrence and identify the effective transition and input gain.
3. Why does constraining `g_t` to `[0, 1]` make the "retain vs overwrite" interpretation easier than allowing arbitrary real-valued gates?

## Conditional Retention Task

4. In the toy task, why can a single fixed gate not both overwrite sharply on write steps and retain nearly perfectly on distractor steps?
5. Suppose the payload sequence is `[2, 0, 0, -1, 0]` and writes happen at timesteps `0` and `3`. What is the ideal target memory trace?
6. If a fixed recurrence uses effective gate `g = 0`, what does it do well and what does it fail to do on the conditional retention task?
7. If a fixed recurrence uses effective gate `g` very close to `1`, what does it preserve well and what does it fail to do?

## Failure Analysis

8. The notebook shows a weak-control failure case where the sigmoid gate stays near `0.5` instead of becoming close to `0` or `1`. Why does that produce blurry memory updates?
9. Give one realistic reason an input-dependent gating policy might fail even if the recurrence equation is expressive enough.
10. Why is this notebook described as a Mamba-style toy rather than a faithful implementation of Mamba?

## Architecture Families

11. In one sentence each, describe how the notebook positions Mamba, Hyena, and RWKV relative to the toy selective recurrence:
    - Mamba
    - Hyena
    - RWKV
12. Why would it be misleading to claim from this notebook alone that selective recurrence universally replaces attention or convolution?
