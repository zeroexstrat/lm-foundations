# Long-Context Evaluation Exercises

## Synthetic Task Design

1. Why is it useful that the generated answer token appears exactly once in each prompt?
2. Give one way a natural-language long-context benchmark can be contaminated by pretraining.
3. Why do reserved symbolic prefixes such as `PASS_` and `VAL_` help reduce shortcut solutions?

## Distance-Conditioned Metrics

4. Define `distance_tokens` in one sentence.
5. Why can aggregate exact-match accuracy hide a serious long-context failure?
6. Suppose bucket `[48, 96)` has accuracy `0.25` and bucket `[0, 16)` has accuracy `1.00`. What does that gap suggest?

## Task Families

7. What does the passkey retrieval task isolate?
8. How is delayed copy different from passkey retrieval even though both use one answer-bearing span?
9. In repeated-key induction, why should the key repeat while the answer token does not?
10. In conflicting near/far evidence, what failure mode is revealed when a model returns the nearby distractor?

## Baselines And Interpretation

11. For a fixed-window baseline with visible window `W`, when should single-evidence task accuracy drop?
12. Why is the notebook's oracle parser useful even though it is not a realistic language model?
13. Name one limitation of exact-match as the only metric for long-context evaluation.
