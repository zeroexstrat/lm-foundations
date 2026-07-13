# Optimization At Scale Solutions

## AdamW Mechanics

1. AdamW uses
   `m_t = beta_1 m_{t-1} + (1 - beta_1) g_t`,
   `v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2`,
   `m_hat_t = m_t / (1 - beta_1^t)`,
   `v_hat_t = v_t / (1 - beta_2^t)`,
   and
   `theta_t = (1 - lr * lambda) theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)`.
2. Adding `lambda * theta` to the gradient mixes weight decay into Adam's adaptive moments. Decoupled decay scales the parameter directly, so the regularizer is not preconditioned by `v_t`.
3. Epsilon is added after the bias-corrected square root term: `sqrt(v_hat_t) + eps`.
4. Early in training, both moments are biased toward zero because they start at zero. The correction factors undo that initialization bias.

## Gradient Accumulation

5. You need a weighted average when microbatches contain different numbers of examples or tokens. Otherwise a tiny remainder microbatch would count as much as a full microbatch.
6. The effective batch size is `8 * 16 * 4 = 512` examples per optimizer update.
7. The token count is `512 * 2,048 = 1,048,576` tokens per optimizer update.
8. Accumulation delays optimizer updates. With more microbatches consumed before each step, the same epoch produces fewer optimizer steps.

## Schedules And Precision

9. Step-indexed schedules advance on optimizer steps, not raw microbatches. If accumulation changes, the warmup and decay boundaries move unless total optimizer steps are recomputed.
10. The first and second moments integrate many updates and are numerically sensitive. Keeping them in fp32 reduces rounding error and underflow relative to lower precision.
11. The extra term is one fp32 copy of the parameters: `N * b_master`.

## Memory Accounting

12. One simple tracked training-state formula is
    `N * (b_param + b_grad + b_m1 + b_m2 + 1_master * b_master)`,
    where `1_master` is 1 when master weights are enabled and 0 otherwise.
13. The total tracked training-state memory is
    `1e9 * (2 + 2 + 4 + 4 + 4) = 1.6e10` bytes, about `14.9 GiB`.
14. ZeRO-style partitioning targets replicated optimizer moments, gradients, and eventually parameters because those replicated tensors dominate memory as parameter count grows.

## Scaling Judgment

15. An optimizer-related bottleneck is replicated AdamW state memory. A parallelism-related bottleneck is distributing model layers or matrix multiplies efficiently across devices.
16. Width changes can alter stable learning-rate and initialization transfer behavior. MuP is a warning that "same optimizer knobs at larger width" is not automatically scale-invariant.
