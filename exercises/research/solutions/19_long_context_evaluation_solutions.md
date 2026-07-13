# Long-Context Evaluation Solutions

## Synthetic Task Design

1. It guarantees there is only one correct evidence span to retrieve, so accuracy cannot be inflated by multiple answer copies in the same prompt.
2. A model may have seen benchmark passages, entities, or answer patterns during pretraining and answer from memorized world knowledge instead of the current context.
3. They make the tokens look synthetic rather than like reusable natural-language facts, which reduces the chance that pretrained associations solve the task.

## Distance-Conditioned Metrics

4. `distance_tokens` is the number of context tokens between the answer-bearing span and the query.
5. A model can score well on short-range examples and still fail badly once the support moves far away, so one global average can blur the collapse.
6. It suggests the system handles local retrieval well but loses accuracy when evidence moves into a longer-context regime.

## Task Families

7. Passkey retrieval isolates whether a model can recover one distant fact embedded in otherwise irrelevant context.
8. Delayed copy focuses on preserving a token over a delay, while passkey retrieval frames the same memory problem as a later question-answer lookup.
9. Repeating the key creates an induction-style cue, while keeping the answer unique prevents multiple valid answer spans from leaking into the prompt.
10. It reveals source-selection failure: the model sees both candidates but trusts the wrong, more local source.

## Baselines And Interpretation

11. It should drop once the answer-bearing span lies outside the last `W` visible context tokens.
12. It gives a flat upper bound and confirms that any degradation we observe comes from the baseline's access limits rather than from a broken dataset.
13. Exact-match ignores calibration and partial uncertainty, so a cautious abstention looks just as wrong as an overconfident distractor answer.
