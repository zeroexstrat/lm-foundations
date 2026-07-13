# Optimization At Scale Exercises

## AdamW Mechanics

1. Starting from the Adam moments `m_t` and `v_t`, write the AdamW parameter update with decoupled weight decay.
2. Why is `theta <- theta * (1 - lr * lambda)` not the same as adding `lambda * theta` to the gradient inside Adam?
3. In the AdamW denominator, where does epsilon appear relative to the bias-corrected square root term?
4. What problem do the bias-correction factors `1 - beta_1^t` and `1 - beta_2^t` solve?

## Gradient Accumulation

5. Suppose each microbatch gradient is already averaged over its own examples. When do you need a weighted average rather than a plain mean across microbatches?
6. A run uses `micro_batch_size = 8`, `accumulation_steps = 16`, and `data_parallel_size = 4`. What is the effective batch size per optimizer update?
7. If the sequence length is 2,048 tokens, how many tokens does one optimizer update cover in the setup above?
8. Why does increasing accumulation steps change the number of optimizer steps per epoch even when total tokens processed stay fixed?

## Schedules And Precision

9. Give one reason a step-indexed warmup or cosine schedule must be recomputed when accumulation changes.
10. Why do mixed-precision training recipes often keep optimizer moments in fp32 even if model weights are stored in bf16 or fp16?
11. What extra memory term appears when a mixed-precision recipe keeps fp32 master weights?

## Memory Accounting

12. Write a per-parameter training-state memory formula that includes parameters, gradients, first moments, second moments, and optional master weights.
13. For `N = 1e9` parameters, `b_param = 2`, `b_grad = 2`, `b_state = 4`, and fp32 master weights enabled, how many tracked training-state bytes does this AdamW-style recipe require in total?
14. Which components would ZeRO-style partitioning try to shard, and why does that matter more as `N` grows?

## Scaling Judgment

15. Megatron-LM and ZeRO address different bottlenecks. State one optimizer-related bottleneck and one parallelism-related bottleneck.
16. MuP argues for width-aware hyperparameter transfer. Why is that a warning against blindly copying one optimizer setting across scales?
