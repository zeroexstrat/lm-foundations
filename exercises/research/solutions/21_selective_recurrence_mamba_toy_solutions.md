# Selective Recurrence And Mamba Toy Solutions

## Gate Algebra

1. When `g_t` is close to `1`, the recurrence mostly keeps the previous state and only weakly mixes in the new input. When `g_t` is close to `0`, it largely discards the previous state and overwrites with the current input contribution.
2. The recurrence becomes `x_t = (g a) x_{t-1} + ((1 - g) b) u_t`. The effective fixed transition is `g a`, and the effective input gain is `(1 - g) b`.
3. Bounding the gate keeps it interpretable as a convex tradeoff between retaining old state and writing new input, instead of allowing sign flips or unbounded amplification from the gate itself.

## Conditional Retention Task

4. The task needs two opposite behaviors at different times: fast overwrite on write steps and near-lossless carry on distractor steps. A single fixed gate can only choose one compromise value everywhere.
5. The ideal trace is `[2, 2, 2, -1, -1]`.
6. With `g = 0`, the model writes the payload immediately and exactly on write steps, but it forgets everything on the very next distractor because the state is fully overwritten by the current input.
7. With `g` near `1`, the model preserves existing state well between writes, but new payloads enter too slowly, so overwriting is weak and delayed.

## Failure Analysis

8. Gates near `0.5` neither retain decisively nor overwrite decisively, so each update averages the past and present. The state becomes a smeared mixture instead of a crisp stored value.
9. The controller input may be noisy, ambiguous, or missing the feature that indicates when memory should update, so the gate policy never becomes selective enough.
10. Real Mamba models use richer selective state-space parameterization and learned sequence blocks. This notebook isolates one scalar gating idea to study the mechanism, not the full architecture.

## Architecture Families

11. 
    - Mamba: a selective state-space family where sequence dynamics depend on the input, of which this notebook shows only a tiny scalar toy.
    - Hyena: a long-convolution family that also targets long-range sequence modeling, but through implicit convolutional filters rather than this gated scalar recurrence.
    - RWKV: a gated recurrent family that mixes recurrence and transformer-like training behavior, related by theme but not identical to the selective SSM construction here.
12. The notebook studies one tiny synthetic task with hand-designed gates, so it does not test scaling, language modeling quality, optimization behavior, or real-world tradeoffs against other architectures.
