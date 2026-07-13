# Research Notebook Index

This index is the navigation spine for the research layer. It keeps the 13-32 notebook sequence in dependency order and pairs each notebook with the contract it must satisfy before it is considered complete.

## Module Dependency Order

Implement the research modules in this order:

1. R01 Advanced attention mathematics
2. R02 Sparse, local, block, and global attention
3. R03 Linear and kernelized attention
4. R04 Low-rank and randomized attention approximations
5. R05 KV-cache, memory bandwidth, and inference complexity
6. R06 Long-context modeling
7. R07 State-space models and recurrence
8. R08 Mixture-of-experts and conditional computation
9. R09 Scaling laws and empirical science
10. R10 Optimization at scale
11. R11 Representation learning and self-supervised objectives
12. R12 JEPA-style predictive representation learning
13. R13 World models, latent dynamics, planning, and agency
14. R14 Retrieval, memory, and external knowledge systems
15. R15 Mechanistic interpretability
16. R16 Evaluation, benchmarks, contamination, and validity
17. R17 Quantization, compression, distillation, and efficient inference
18. R18 Research methodology and experiment design

The dependency rule from the curriculum applies here: keep the build incremental, and do not skip ahead to notebooks that rely on earlier mathematical or experimental machinery.

## Notebook Sequence

Current implementation status: `13_research_orientation.ipynb` is the first completed research notebook and the handoff from the college-level path into the research layer. Run it directly with `uv run python scripts/smoke_notebooks.py --notebook 13_research_orientation.ipynb`.

| Notebook | Module | Depends on | Per-notebook contract |
|---|---|---|---|
| `13_research_orientation.ipynb` | Research method orientation | College-level curriculum completion | Explain paper anatomy, define claim/bottleneck/mechanism/evidence/limitation, fill one paper-lab example, and run a paired-metric baseline-vs-variant toy replication. |
| `14_attention_as_kernel_operator.ipynb` | R01 | `13_research_orientation.ipynb` | Re-derive exact attention, include entropy and gradient checks, and cite only registry-backed sources. |
| `15_sparse_attention_patterns.ipynb` | R02 | `14_attention_as_kernel_operator.ipynb` | Compare sparse masks, prove or test reachability, and state the complexity tradeoff. |
| `16_linear_kernel_attention.ipynb` | R03 | `15_sparse_attention_patterns.ipynb` | Implement feature-map attention, compare against exact attention, and include approximation-error checks. |
| `17_low_rank_attention.ipynb` | R04 | `16_linear_kernel_attention.ipynb` | Test low-rank assumptions with SVD or landmark diagnostics and document retrieval failure cases. |
| `18_kv_cache_inference.ipynb` | R05 | `17_low_rank_attention.ipynb` | Measure decode-time memory cost, compare MHA/MQA/GQA, and include byte-accounting checks. |
| `19_long_context_evaluation.ipynb` | R06 | `18_kv_cache_inference.ipynb` | Build contamination-resistant long-context tests and report distance-conditioned metrics. |
| `20_state_space_models.ipynb` | R07 | `19_long_context_evaluation.ipynb` | Derive recurrence/convolution equivalence, test stability, and compare recurrent and convolutional forms. |
| `21_selective_recurrence_mamba_toy.ipynb` | R07 | `20_state_space_models.ipynb` | Add input-dependent memory and show where selective recurrence helps or fails. |
| `22_moe_conditional_compute.ipynb` | R08 | `21_selective_recurrence_mamba_toy.ipynb` | Implement routing, load balancing, and collapse diagnostics for sparse experts. |
| `23_scaling_laws_lab.ipynb` | R09 | `22_moe_conditional_compute.ipynb` | Fit a small scaling-law experiment, include uncertainty, and avoid unsupported extrapolation. |
| `24_optimization_at_scale.ipynb` | R10 | `23_scaling_laws_lab.ipynb` | Account for optimizer state, schedules, and precision effects with reproducible comparisons. |
| `25_self_supervised_objectives.ipynb` | R11 | `24_optimization_at_scale.ipynb` | Compare contrastive, noncontrastive, redundancy-reduction, and masked objectives. |
| `26_jepa_predictive_representations.ipynb` | R12 | `25_self_supervised_objectives.ipynb` | Implement latent prediction, compare against reconstruction, and report invariance tradeoffs. |
| `27_world_models_and_planning.ipynb` | R13 | `26_jepa_predictive_representations.ipynb` | Build latent dynamics plus planning and measure rollout error. |
| `28_retrieval_memory_systems.ipynb` | R14 | `27_world_models_and_planning.ipynb` | Add external memory, test recall, and separate retrieval gains from parametric memorization. |
| `29_mechanistic_interpretability.ipynb` | R15 | `28_retrieval_memory_systems.ipynb` | Analyze a small circuit causally and avoid equating attention weights with explanations. |
| `30_evaluation_validity.ipynb` | R16 | `29_mechanistic_interpretability.ipynb` | Audit benchmarks, contamination, and uncertainty with paired comparisons. |
| `31_compression_research.ipynb` | R17 | `30_evaluation_validity.ipynb` | Compare quantization or compression variants and state the calibration setup. |
| `32_research_capstone_workshop.ipynb` | R18 | `31_compression_research.ipynb` | Produce a bounded proposal with hypothesis, baseline, metric, ablation, assumptions, risks, and failure criteria. |

## Contract Standard

Every research notebook must include these sections in order:

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

The notebook may add more sections if needed, but it may not omit any of the sections above.
