# Beyond Transformers Reading Map

When reading a new architecture paper, identify:

1. Objective: what training signal is optimized?
2. Inductive bias: what structure is built into the model?
3. Bottleneck: what cost or failure mode is targeted?
4. Scaling behavior: what changes as data, parameters, or context length grow?
5. Data assumptions: what supervision, modalities, or environment interactions are required?
6. Evaluation: what benchmark actually supports the claim?
7. Deployment constraint: what hardware or memory pattern makes the method practical?

Use the attention notebook's `T x T` score matrix as the reference point for sparse and subquadratic attention claims.
