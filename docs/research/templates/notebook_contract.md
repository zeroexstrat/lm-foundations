# Research Notebook Contract

Use this template for every research notebook. Keep it short, concrete, and falsifiable.

## Required Structure

1. Motivation
2. Hypothesis
3. Baseline
4. Metric
5. Mathematical derivation
6. PyTorch implementation
7. Numerical checks
8. Ablations
9. Assumptions
10. Risks
11. Failure criteria
12. Exercises
13. References

## Writing Rules

- State the claim in one sentence before the derivation starts.
- Name the baseline explicitly; do not compare against "nothing".
- Choose one primary metric and one secondary diagnostic.
- Include at least one ablation that can fail the claim.
- State assumptions that must hold for the mechanism to work.
- State the smallest failure condition that would invalidate the notebook's main claim.
- Keep the code path deterministic in quick mode.
- Reference only papers that appear in `docs/research/references.yaml`.

