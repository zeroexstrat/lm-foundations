# Beyond-Transformers Exercises

## Complexity Reasoning

1. For context length `T`, compare full attention score count to sliding-window attention with window `w`.
2. Compute the score counts for `T = 8192` and `w = 256`.

## Architecture Tradeoffs

3. Name one benefit and one cost of sparse attention.
4. Explain why grouped-query attention can reduce KV-cache memory pressure during generation.
5. Name one reason recurrence or state-space models can be attractive for long contexts.

## Objective Reasoning

6. Explain why a predictive representation objective differs from next-token generation.
7. State one reason JEPA-style methods are interesting for world models.

## Paper Reading

8. List three questions to ask when reading a post-transformer architecture paper.
9. Identify whether a claimed improvement targets objective, inductive bias, compute bottleneck, data assumptions, or evaluation.
