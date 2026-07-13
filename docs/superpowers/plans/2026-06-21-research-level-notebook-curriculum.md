# Research-Level Notebook Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the approved research-level LLM curriculum into a notebook-based project extension that is rigorous, smoke-tested, and implemented in dependency order.

**Architecture:** Keep the existing college-level notebooks intact. Add a research layer with one independently reviewable notebook task at a time, backed by small tested utilities under `src/llm_from_scratch/research/`, exercises under `exercises/research/`, verified paper references under `docs/research/references.yaml`, and notebook contract/smoke tests.

**Tech Stack:** Python 3.11+, uv, PyTorch, NumPy, Matplotlib, nbformat, nbclient, pytest, PyYAML if needed for reference metadata.

## Global Constraints

- Do not write all notebooks at once.
- Start with scaffolding and curriculum index, then implement modules in dependency order.
- Each notebook must include markdown explanation, mathematical derivation, executable PyTorch code, numerical checks, exercises, and references.
- Each task must be independently reviewable and must end with focused tests or smoke checks.
- Keep paper references verified against primary sources before citing them in notebook text.
- Long or stochastic experiments must default to quick deterministic settings; stretch runs must be opt-in.
- Use `uv run python -m pytest`, not `uv run pytest`, because this checkout has a stale `.venv/bin/pytest` shebang from an older path.
- Do not change the existing 00-12 college-level notebooks except to link the research layer from indexes or navigation cells.

---

## 1. File Tree

```text
llm-from-scratch/
  README.md
  docs/
    notes/
      curriculum_index.md
      research_level_curriculum.md
      research_notebook_index.md
    research/
      references.yaml
      templates/
        notebook_contract.md
        paper_lab.md
        capstone_proposal.md
        experiment_plan.md
        results.md
      capstones/
        README.md
    superpowers/
      plans/
        2026-06-21-research-level-notebook-curriculum.md
  exercises/
    research/
      13_research_orientation.md
      14_attention_as_kernel_operator.md
      15_sparse_attention_patterns.md
      16_linear_kernel_attention.md
      17_low_rank_attention.md
      18_kv_cache_inference.md
      19_long_context_evaluation.md
      20_state_space_models.md
      21_selective_recurrence_mamba_toy.md
      22_moe_conditional_compute.md
      23_scaling_laws_lab.md
      24_optimization_at_scale.md
      25_self_supervised_objectives.md
      26_jepa_predictive_representations.md
      27_world_models_and_planning.md
      28_retrieval_memory_systems.md
      29_mechanistic_interpretability.md
      30_evaluation_validity.md
      31_compression_research.md
      32_research_capstone_workshop.md
      solutions/
        13_research_orientation_solutions.md
        14_attention_as_kernel_operator_solutions.md
        15_sparse_attention_patterns_solutions.md
        16_linear_kernel_attention_solutions.md
        17_low_rank_attention_solutions.md
        18_kv_cache_inference_solutions.md
        19_long_context_evaluation_solutions.md
        20_state_space_models_solutions.md
        21_selective_recurrence_mamba_toy_solutions.md
        22_moe_conditional_compute_solutions.md
        23_scaling_laws_lab_solutions.md
        24_optimization_at_scale_solutions.md
        25_self_supervised_objectives_solutions.md
        26_jepa_predictive_representations_solutions.md
        27_world_models_and_planning_solutions.md
        28_retrieval_memory_systems_solutions.md
        29_mechanistic_interpretability_solutions.md
        30_evaluation_validity_solutions.md
        31_compression_research_solutions.md
        32_research_capstone_workshop_solutions.md
  notebooks/
    13_research_orientation.ipynb
    14_attention_as_kernel_operator.ipynb
    15_sparse_attention_patterns.ipynb
    16_linear_kernel_attention.ipynb
    17_low_rank_attention.ipynb
    18_kv_cache_inference.ipynb
    19_long_context_evaluation.ipynb
    20_state_space_models.ipynb
    21_selective_recurrence_mamba_toy.ipynb
    22_moe_conditional_compute.ipynb
    23_scaling_laws_lab.ipynb
    24_optimization_at_scale.ipynb
    25_self_supervised_objectives.ipynb
    26_jepa_predictive_representations.ipynb
    27_world_models_and_planning.ipynb
    28_retrieval_memory_systems.ipynb
    29_mechanistic_interpretability.ipynb
    30_evaluation_validity.ipynb
    31_compression_research.ipynb
    32_research_capstone_workshop.ipynb
  scripts/
    check_research_notebooks.py
    smoke_notebooks.py
    strip_notebook_outputs.py
    verify_research_references.py
  src/
    llm_from_scratch/
      research/
        __init__.py
        attention_math.py
        sparse_attention.py
        linear_attention.py
        low_rank_attention.py
        inference.py
        sequence_tasks.py
        state_space.py
        moe.py
        scaling.py
        optimization.py
        self_supervised.py
        world_models.py
        retrieval.py
        interpretability.py
        evaluation_validity.py
        compression.py
  tests/
    test_research_references.py
    test_research_notebook_contract.py
    test_research_attention_math.py
    test_research_sparse_attention.py
    test_research_linear_attention.py
    test_research_low_rank_attention.py
    test_research_inference.py
    test_research_sequence_tasks.py
    test_research_state_space.py
    test_research_moe.py
    test_research_scaling.py
    test_research_optimization.py
    test_research_self_supervised.py
    test_research_world_models.py
    test_research_retrieval.py
    test_research_interpretability.py
    test_research_evaluation_validity.py
    test_research_compression.py
```

## 2. Task-By-Task Implementation Plan

Each notebook task follows the same contract:

- Add one notebook only.
- Add or extend only the utility module needed by that notebook.
- Add one exercise file and one solution file.
- Add unit tests for utility code.
- Add the notebook to the research smoke list only after it executes in quick mode.
- Add every cited paper to `docs/research/references.yaml` before referencing it in notebook text.

### Task 1: Research Layer Scaffolding And Index

**Files:**
- Create: `docs/notes/research_notebook_index.md`
- Create: `docs/research/references.yaml`
- Create: `docs/research/templates/notebook_contract.md`
- Create: `docs/research/templates/paper_lab.md`
- Create: `docs/research/templates/capstone_proposal.md`
- Create: `docs/research/templates/experiment_plan.md`
- Create: `docs/research/templates/results.md`
- Create: `docs/research/capstones/README.md`
- Create: `exercises/research/solutions/.gitkeep`
- Modify: `README.md`
- Modify: `docs/notes/curriculum_index.md`

**Work:**
- Add the research notebook order, module dependency order, and per-notebook contract.
- Seed `references.yaml` with the already verified references from `docs/notes/research_level_curriculum.md`.
- Make the README explicit that research notebooks are implemented incrementally and may not all exist yet.

**Acceptance criteria:**
- `docs/notes/research_notebook_index.md` lists notebooks 13-32 in dependency order.
- `docs/research/references.yaml` includes title, authors, year, primary URL, arXiv ID when available, and modules using the source.
- Templates force hypothesis, baseline, metric, ablation, assumptions, risks, and failure criteria.
- No notebook files 13-32 are created in this task.

### Task 2: Notebook QA And Smoke Harness

**Files:**
- Create: `scripts/check_research_notebooks.py`
- Create: `scripts/verify_research_references.py`
- Create: `tests/test_research_notebook_contract.py`
- Create: `tests/test_research_references.py`
- Modify: `scripts/smoke_notebooks.py`
- Modify: `tests/test_notebook_scripts.py`

**Work:**
- Extend `smoke_notebooks.py` with `RESEARCH_NOTEBOOKS`, `--research`, and `--notebook NAME`.
- Add contract checks for required markdown headings: Motivation, Mathematical derivation, PyTorch implementation, Numerical checks, Exercises, References.
- Add reference checks that every notebook citation key exists in `references.yaml`.
- Keep live web verification manual: `scripts/verify_research_references.py --live` should exist, but normal tests must not require network.

**Acceptance criteria:**
- Existing 00-12 quick smoke behavior is unchanged.
- `uv run python scripts/smoke_notebooks.py --notebook 00_project_orientation.ipynb` executes one notebook.
- Contract tests pass even when no research notebooks exist yet.

### Task 3: Research Orientation Notebook

**Files:**
- Create: `notebooks/13_research_orientation.ipynb`
- Create: `exercises/research/13_research_orientation.md`
- Create: `exercises/research/solutions/13_research_orientation_solutions.md`
- Modify: `docs/notes/research_notebook_index.md`
- Modify: `scripts/smoke_notebooks.py`

**Work:**
- Teach paper anatomy: claim, bottleneck, mathematical mechanism, objective/architecture change, evidence, limitations, toy replication.
- Include an executable PyTorch micro-example showing how to compare a baseline and one variant with a paired metric.
- Include a filled paper-lab example using one already verified source.

**Acceptance criteria:**
- Notebook satisfies the contract check.
- Notebook executes in quick mode.
- Exercise solution includes a critique and a small experiment design.

### Task 4: R01 Advanced Attention Mathematics

**Files:**
- Create: `src/llm_from_scratch/research/__init__.py`
- Create: `src/llm_from_scratch/research/attention_math.py`
- Create: `tests/test_research_attention_math.py`
- Create: `notebooks/14_attention_as_kernel_operator.ipynb`
- Create: `exercises/research/14_attention_as_kernel_operator.md`
- Create: `exercises/research/solutions/14_attention_as_kernel_operator_solutions.md`

**Work:**
- Implement stable exact attention, entropy diagnostics, row-stochastic checks, and finite-difference gradient checks.
- Derive softmax attention as entropy-regularized selection.
- Reference "Attention Is All You Need" from the verified registry.

**Acceptance criteria:**
- Tests verify row sums, causal masking, shape invariants, entropy bounds, and gradient-check tolerance.
- Notebook compares exact attention behavior across sharp and diffuse logits.

### Task 5: R02 Sparse, Local, Block, And Global Attention

**Files:**
- Create: `src/llm_from_scratch/research/sparse_attention.py`
- Create: `tests/test_research_sparse_attention.py`
- Create: `notebooks/15_sparse_attention_patterns.ipynb`
- Create: `exercises/research/15_sparse_attention_patterns.md`
- Create: `exercises/research/solutions/15_sparse_attention_patterns_solutions.md`

**Work:**
- Implement local, block, dilated, global-token, and random causal masks.
- Add graph reachability utilities to show receptive-field growth.
- Use Longformer, BigBird, and Sparse Transformer references.

**Acceptance criteria:**
- Tests verify masks are causal, shapes are correct, and reachability increases with layers.
- Notebook includes a toy delayed-copy or far-token retrieval task with mask comparisons.

### Task 6: R03 Linear And Kernelized Attention

**Files:**
- Create: `src/llm_from_scratch/research/linear_attention.py`
- Create: `tests/test_research_linear_attention.py`
- Create: `notebooks/16_linear_kernel_attention.ipynb`
- Create: `exercises/research/16_linear_kernel_attention.md`
- Create: `exercises/research/solutions/16_linear_kernel_attention_solutions.md`

**Work:**
- Implement linear attention with deterministic positive features and a small random-feature approximation.
- Derive associativity and causal prefix-sum computation.
- Compare output error against exact attention from Task 4.

**Acceptance criteria:**
- Tests verify no `T x T` attention matrix is materialized in the linear reference path.
- Notebook reports approximation error as feature dimension changes.

### Task 7: R04 Low-Rank And Randomized Attention Approximations

**Files:**
- Create: `src/llm_from_scratch/research/low_rank_attention.py`
- Create: `tests/test_research_low_rank_attention.py`
- Create: `notebooks/17_low_rank_attention.ipynb`
- Create: `exercises/research/17_low_rank_attention.md`
- Create: `exercises/research/solutions/17_low_rank_attention_solutions.md`

**Work:**
- Implement SVD diagnostics, projected sequence attention, and a small Nystrom-style approximation.
- Derive Eckart-Young and landmark approximation intuition.
- Compare local smoothing versus exact retrieval tasks.

**Acceptance criteria:**
- Tests verify reconstruction error decreases as rank/landmarks increase on controlled matrices.
- Notebook explains a case where low Frobenius error still fails token retrieval.

### Task 8: R05 KV-Cache, Memory Bandwidth, And Inference Complexity

**Files:**
- Create: `src/llm_from_scratch/research/inference.py`
- Create: `tests/test_research_inference.py`
- Create: `notebooks/18_kv_cache_inference.ipynb`
- Create: `exercises/research/18_kv_cache_inference.md`
- Create: `exercises/research/solutions/18_kv_cache_inference_solutions.md`

**Work:**
- Implement KV-cache byte accounting for MHA, MQA, and GQA.
- Add a tiny decode simulator that separates prefill from decode.
- Reference MQA, GQA, and FlashAttention papers without claiming FlashAttention changes the math.

**Acceptance criteria:**
- Tests verify byte formulas for fp32, fp16, and int8-like element sizes.
- Notebook reports cache memory by layer and generated token.

### Task 9: R06 Long-Context Modeling

**Files:**
- Create: `src/llm_from_scratch/research/sequence_tasks.py`
- Create: `tests/test_research_sequence_tasks.py`
- Create: `notebooks/19_long_context_evaluation.ipynb`
- Create: `exercises/research/19_long_context_evaluation.md`
- Create: `exercises/research/solutions/19_long_context_evaluation_solutions.md`

**Work:**
- Implement passkey retrieval, delayed copy, repeated-key induction, and conflicting near/far evidence generators.
- Show distance-conditioned evaluation metrics.
- Reference Transformer-XL and Memorizing Transformers.

**Acceptance criteria:**
- Tests verify generated examples contain exactly one answer-bearing span and deterministic seeds reproduce samples.
- Notebook includes a contamination-resistant task construction section.

### Task 10: R07 State-Space Models

**Files:**
- Create: `src/llm_from_scratch/research/state_space.py`
- Create: `tests/test_research_state_space.py`
- Create: `notebooks/20_state_space_models.ipynb`
- Create: `exercises/research/20_state_space_models.md`
- Create: `exercises/research/solutions/20_state_space_models_solutions.md`

**Work:**
- Implement scalar and diagonal linear recurrences, impulse responses, and recurrent/convolution equivalence checks.
- Derive discretized linear state-space dynamics.
- Reference S4.

**Acceptance criteria:**
- Tests verify recurrence and convolution outputs match within tolerance.
- Notebook includes stability sweeps for transition values inside and outside the unit circle.

### Task 11: R07 Selective Recurrence And Mamba Toy

**Files:**
- Modify: `src/llm_from_scratch/research/state_space.py`
- Modify: `tests/test_research_state_space.py`
- Create: `notebooks/21_selective_recurrence_mamba_toy.ipynb`
- Create: `exercises/research/21_selective_recurrence_mamba_toy.md`
- Create: `exercises/research/solutions/21_selective_recurrence_mamba_toy_solutions.md`

**Work:**
- Add input-dependent gates for selective retention/forgetting.
- Compare fixed recurrence and selective recurrence on a conditional retention task.
- Reference Mamba, Hyena, and RWKV carefully as architecture families, not as proved universal replacements.

**Acceptance criteria:**
- Tests verify gate values are bounded and selective recurrence behaves like fixed recurrence when gates are constant.
- Notebook reports both success and failure cases.

### Task 12: R08 Mixture-of-Experts And Conditional Computation

**Files:**
- Create: `src/llm_from_scratch/research/moe.py`
- Create: `tests/test_research_moe.py`
- Create: `notebooks/22_moe_conditional_compute.ipynb`
- Create: `exercises/research/22_moe_conditional_compute.md`
- Create: `exercises/research/solutions/22_moe_conditional_compute_solutions.md`

**Work:**
- Implement top-1/top-2 routing, capacity limits, and simple load-balancing diagnostics.
- Demonstrate routing collapse and auxiliary-loss mitigation.
- Reference GShard, Switch, and Mixtral.

**Acceptance criteria:**
- Tests verify top-k routing shape, capacity behavior, and load histogram accounting.
- Notebook distinguishes parameter count from active FLOPs.

### Task 13: R09 Scaling Laws And Empirical Science

**Files:**
- Create: `src/llm_from_scratch/research/scaling.py`
- Create: `tests/test_research_scaling.py`
- Create: `notebooks/23_scaling_laws_lab.ipynb`
- Create: `exercises/research/23_scaling_laws_lab.md`
- Create: `exercises/research/solutions/23_scaling_laws_lab_solutions.md`

**Work:**
- Implement synthetic power-law fitting, bootstrap intervals, and compute accounting helpers.
- Include an optional stretch hook for tiny model sweeps, disabled by default.
- Reference Kaplan et al. and Hoffmann et al.

**Acceptance criteria:**
- Tests verify fitted exponents recover known synthetic parameters within tolerance.
- Notebook states why small-scale fits do not justify frontier extrapolation.

### Task 14: R10 Optimization At Scale

**Files:**
- Create: `src/llm_from_scratch/research/optimization.py`
- Create: `tests/test_research_optimization.py`
- Create: `notebooks/24_optimization_at_scale.ipynb`
- Create: `exercises/research/24_optimization_at_scale.md`
- Create: `exercises/research/solutions/24_optimization_at_scale_solutions.md`

**Work:**
- Implement AdamW reference updates, gradient accumulation accounting, and optimizer-state memory estimates.
- Compare one update against PyTorch AdamW on a tiny tensor.
- Reference AdamW, muP, ZeRO, and Megatron-LM.

**Acceptance criteria:**
- Tests verify AdamW update parity for a controlled case and correct memory accounting.
- Notebook includes mixed-precision caveats without requiring GPU.

### Task 15: R11 Representation Learning And Self-Supervised Objectives

**Files:**
- Create: `src/llm_from_scratch/research/self_supervised.py`
- Create: `tests/test_research_self_supervised.py`
- Create: `notebooks/25_self_supervised_objectives.ipynb`
- Create: `exercises/research/25_self_supervised_objectives.md`
- Create: `exercises/research/solutions/25_self_supervised_objectives_solutions.md`

**Work:**
- Implement InfoNCE, covariance diagnostics, and collapse metrics on synthetic embeddings.
- Demonstrate contrastive and redundancy-reduction objectives.
- Reference SimCLR, BYOL, Barlow Twins, VICReg, and MAE.

**Acceptance criteria:**
- Tests verify loss shapes, finite values, and collapse metric behavior on constant embeddings.
- Notebook includes at least one anti-collapse ablation.

### Task 16: R12 JEPA-Style Predictive Representation Learning

**Files:**
- Modify: `src/llm_from_scratch/research/self_supervised.py`
- Modify: `tests/test_research_self_supervised.py`
- Create: `notebooks/26_jepa_predictive_representations.ipynb`
- Create: `exercises/research/26_jepa_predictive_representations.md`
- Create: `exercises/research/solutions/26_jepa_predictive_representations_solutions.md`

**Work:**
- Implement a toy context encoder, target encoder, predictor, and stop-gradient latent prediction loss.
- Compare latent prediction against raw reconstruction under nuisance noise.
- Reference I-JEPA and V-JEPA.

**Acceptance criteria:**
- Tests verify stop-gradient behavior and deterministic toy batch generation.
- Notebook evaluates downstream probe quality, not just JEPA loss.

### Task 17: R13 World Models, Latent Dynamics, Planning, And Agency

**Files:**
- Create: `src/llm_from_scratch/research/world_models.py`
- Create: `tests/test_research_world_models.py`
- Create: `notebooks/27_world_models_and_planning.ipynb`
- Create: `exercises/research/27_world_models_and_planning.md`
- Create: `exercises/research/solutions/27_world_models_and_planning_solutions.md`

**Work:**
- Implement a tiny gridworld transition dataset, latent transition model, rollout error metric, and model-predictive planning toy.
- Separate one-step prediction from rollout/planning success.
- Reference World Models, Dreamer, MuZero, and Decision Transformer.

**Acceptance criteria:**
- Tests verify transition dynamics, rollout shapes, and deterministic planning examples.
- Notebook explicitly states what is and is not agency evidence.

### Task 18: R14 Retrieval, Memory, And External Knowledge Systems

**Files:**
- Create: `src/llm_from_scratch/research/retrieval.py`
- Create: `tests/test_research_retrieval.py`
- Create: `notebooks/28_retrieval_memory_systems.ipynb`
- Create: `exercises/research/28_retrieval_memory_systems.md`
- Create: `exercises/research/solutions/28_retrieval_memory_systems_solutions.md`

**Work:**
- Implement a tiny vector datastore, kNN token mixture, and RAG-style fact lookup toy.
- Include oracle, noisy, and no-retrieval baselines.
- Reference RAG, RETRO, kNN-LM, and Neural Turing Machines.

**Acceptance criteria:**
- Tests verify nearest-neighbor ranking and mixture distribution normalization.
- Notebook demonstrates a fact update without retraining model weights.

### Task 19: R15 Mechanistic Interpretability

**Files:**
- Create: `src/llm_from_scratch/research/interpretability.py`
- Create: `tests/test_research_interpretability.py`
- Create: `notebooks/29_mechanistic_interpretability.ipynb`
- Create: `exercises/research/29_mechanistic_interpretability.md`
- Create: `exercises/research/solutions/29_mechanistic_interpretability_solutions.md`

**Work:**
- Implement logit lens helpers, activation patching scaffolds, and residual contribution calculations for tiny models.
- Use a minimal induction-style or factual-association toy.
- Reference Transformer Circuits and ROME.

**Acceptance criteria:**
- Tests verify patching changes only the intended activation tensor and metric deltas are computed correctly.
- Notebook distinguishes attention visualization from causal evidence.

### Task 20: R16 Evaluation, Benchmarks, Contamination, And Scientific Validity

**Files:**
- Create: `src/llm_from_scratch/research/evaluation_validity.py`
- Create: `tests/test_research_evaluation_validity.py`
- Create: `notebooks/30_evaluation_validity.ipynb`
- Create: `exercises/research/30_evaluation_validity.md`
- Create: `exercises/research/solutions/30_evaluation_validity_solutions.md`

**Work:**
- Implement binomial confidence intervals, paired comparison utilities, contamination n-gram checks, and benchmark card templates.
- Reference HELM, MMLU, BIG-bench, and extraction/memorization work.

**Acceptance criteria:**
- Tests verify interval calculations and contamination detection on controlled strings.
- Notebook includes a clean-vs-contaminated score split.

### Task 21: R17 Quantization, Compression, Distillation, And Efficient Inference

**Files:**
- Create: `src/llm_from_scratch/research/compression.py`
- Create: `tests/test_research_compression.py`
- Create: `notebooks/31_compression_research.ipynb`
- Create: `exercises/research/31_compression_research.md`
- Create: `exercises/research/solutions/31_compression_research_solutions.md`

**Work:**
- Implement groupwise quantization, output-error measurement, activation-aware channel protection toy, and calibration-shift demo.
- Build on existing `src/llm_from_scratch/quantization.py` without duplicating its basics.
- Reference GPTQ, AWQ, QLoRA, and LLM.int8().

**Acceptance criteria:**
- Tests verify quantize/dequantize shapes, group behavior, and output-error reduction for protected channels on a controlled example.
- Notebook reports memory, output error, and latency caveats separately.

### Task 22: R18 Research Methodology And Capstone Workshop

**Files:**
- Create: `notebooks/32_research_capstone_workshop.ipynb`
- Create: `exercises/research/32_research_capstone_workshop.md`
- Create: `exercises/research/solutions/32_research_capstone_workshop_solutions.md`
- Modify: `docs/research/capstones/README.md`
- Modify: `docs/research/templates/capstone_proposal.md`
- Modify: `docs/research/templates/experiment_plan.md`
- Modify: `docs/research/templates/results.md`

**Work:**
- Teach how to convert one module into a falsifiable mini-project.
- Include proposal, preregistration, result table, ablation table, and negative-result language.
- Provide 3-5 capstone options from the approved curriculum.

**Acceptance criteria:**
- Notebook includes an executable paired-comparison example.
- Capstone templates reject vague claims by requiring hypothesis, baseline, metric, ablation, compute budget, and failure criteria.

### Task 23: Final Integration And Navigation Pass

**Files:**
- Modify: `README.md`
- Modify: `docs/notes/curriculum_index.md`
- Modify: `docs/notes/research_notebook_index.md`
- Modify: `scripts/smoke_notebooks.py`
- Modify: `tests/test_notebook_scripts.py`

**Work:**
- Mark the research notebook sequence as implemented.
- Ensure quick smoke can run either the original college layer or the research layer.
- Ensure all notebooks are output-free before final review.

**Acceptance criteria:**
- Full unit tests pass.
- College quick smoke passes.
- Research quick smoke passes.
- Notebook outputs are stripped.
- Reference registry covers every cited paper in all research notebooks.

## 3. Acceptance Criteria Per Task

| Task | Review boundary | Required acceptance |
|---|---|---|
| 1 | Scaffolding only | Index, templates, reference registry exist; no notebooks created |
| 2 | QA harness | Notebook contract, reference registry tests, and selective smoke options pass |
| 3 | Orientation | `13_research_orientation.ipynb` smokes and teaches paper lab workflow |
| 4 | R01 | Exact attention utilities pass numerical and gradient checks |
| 5 | R02 | Sparse masks pass causality/reachability tests |
| 6 | R03 | Linear attention runs without materializing full attention matrix in reference path |
| 7 | R04 | Low-rank approximation error decreases on controlled cases |
| 8 | R05 | KV-cache accounting matches expected byte formulas |
| 9 | R06 | Long-context generators are deterministic and contamination-resistant |
| 10 | R07a | SSM recurrence and convolution forms match |
| 11 | R07b | Selective recurrence shows conditional retention and failure cases |
| 12 | R08 | MoE router load and collapse diagnostics work |
| 13 | R09 | Scaling-law fits recover known synthetic exponents |
| 14 | R10 | AdamW reference update matches PyTorch on a controlled case |
| 15 | R11 | SSL objectives expose collapse and anti-collapse behavior |
| 16 | R12 | JEPA toy verifies stop-gradient and downstream probe evaluation |
| 17 | R13 | World-model toy separates one-step and rollout/planning metrics |
| 18 | R14 | Retrieval toy includes no-retrieval, noisy, and oracle baselines |
| 19 | R15 | Patching/logit-lens helpers produce causal metric deltas |
| 20 | R16 | Evaluation notebook includes uncertainty and contamination probes |
| 21 | R17 | Compression notebook separates memory, quality, and latency claims |
| 22 | R18 | Capstone workflow forces falsifiable claims and bounded experiments |
| 23 | Integration | All research notebooks are linked, stripped, referenced, and smoke-tested |

Per-notebook acceptance criteria:

- The notebook has markdown sections for Motivation, Mathematical derivation, PyTorch implementation, Numerical checks, Exercises, and References.
- The derivation includes assumptions and at least one failure mode.
- Code cells run from a clean kernel under quick settings.
- Numerical checks include shapes and at least one value-level assertion.
- Exercises have matching solution files.
- References are in `docs/research/references.yaml` and use primary-source URLs.

## 4. Verification Commands

Run these before starting the research layer:

```bash
uv sync --all-groups
uv run python -m pytest -q
uv run python scripts/smoke_notebooks.py --quick
```

Run these for each notebook task:

```bash
uv run python -m pytest tests/test_research_notebook_contract.py tests/test_research_references.py -q
```

Then run the exact module test and notebook commands named by the task. For Task 4, the commands are:

```bash
uv run python -m pytest tests/test_research_attention_math.py -q
uv run python scripts/smoke_notebooks.py --notebook 14_attention_as_kernel_operator.ipynb
uv run python scripts/check_research_notebooks.py notebooks/14_attention_as_kernel_operator.ipynb
uv run python scripts/strip_notebook_outputs.py notebooks/14_attention_as_kernel_operator.ipynb
```

Run this when adding or changing references:

```bash
uv run python scripts/verify_research_references.py
uv run python scripts/verify_research_references.py --live
```

Run these at cluster gates:

```bash
uv run python -m pytest -q
uv run python scripts/smoke_notebooks.py --quick
uv run python scripts/smoke_notebooks.py --research
uv run python scripts/check_research_notebooks.py notebooks/13_research_orientation.ipynb notebooks/14_attention_as_kernel_operator.ipynb
```

Replace the final command's notebook list with the notebooks in the current cluster.

Run these at final integration:

```bash
uv run python -m pytest -q
uv run python scripts/smoke_notebooks.py --quick
uv run python scripts/smoke_notebooks.py --research
uv run python scripts/verify_research_references.py
uv run python scripts/check_research_notebooks.py notebooks/13_research_orientation.ipynb notebooks/14_attention_as_kernel_operator.ipynb notebooks/15_sparse_attention_patterns.ipynb notebooks/16_linear_kernel_attention.ipynb notebooks/17_low_rank_attention.ipynb notebooks/18_kv_cache_inference.ipynb notebooks/19_long_context_evaluation.ipynb notebooks/20_state_space_models.ipynb notebooks/21_selective_recurrence_mamba_toy.ipynb notebooks/22_moe_conditional_compute.ipynb notebooks/23_scaling_laws_lab.ipynb notebooks/24_optimization_at_scale.ipynb notebooks/25_self_supervised_objectives.ipynb notebooks/26_jepa_predictive_representations.ipynb notebooks/27_world_models_and_planning.ipynb notebooks/28_retrieval_memory_systems.ipynb notebooks/29_mechanistic_interpretability.ipynb notebooks/30_evaluation_validity.ipynb notebooks/31_compression_research.ipynb notebooks/32_research_capstone_workshop.ipynb
uv run python scripts/strip_notebook_outputs.py notebooks/*.ipynb
git diff --check
```

## 5. Review Checklist

Use this checklist for every task:

- Scope: the task adds only the planned notebook, utility code, tests, exercises, and references.
- Dependency order: the task does not rely on later notebooks or unimplemented utilities.
- Notebook contract: markdown explanation, derivation, PyTorch code, numerical checks, exercises, and references are all present.
- Mathematical rigor: assumptions, dimensions, and failure modes are explicit.
- Code quality: utility code is small, deterministic, typed where useful, and covered by tests.
- Numerical checks: notebook assertions verify shapes and values, not just that cells run.
- Exercise quality: prompts are answerable and solution files contain worked answers.
- Reference hygiene: each citation uses `docs/research/references.yaml`; new references are verified before use.
- Scientific claims: notebook separates established results, toy findings, and speculative directions.
- Runtime: quick mode completes locally; stretch work is opt-in and disabled by default.
- Outputs: notebooks are stripped before review.
- Regression: existing 00-12 notebook smoke checks still pass.
