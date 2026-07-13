# PhD/Research-Level LLM Curriculum

This document extends the existing college-level `llm-from-scratch` curriculum. It does not repeat the basic transformer build. It assumes the student can already implement tokenization, embeddings, causal attention, decoder-only blocks, a training loop, generation, evaluation, toy supervised fine-tuning, Hugging Face mappings, quantization basics, and a first orientation to sparse attention, JEPA, and world models.

The goal is research readiness: the student should learn to read a paper as a mathematical and empirical artifact, identify the bottleneck it targets, derive the mechanism, implement a minimal version, test the claimed tradeoff, and state what the evidence does not prove.

## 1. Curriculum Philosophy

The first curriculum answers: "Can I build the mechanism?" This layer answers: "Can I reason about why the mechanism should work, when it fails, and how to test a replacement?"

The shift has five parts:

1. From components to bottlenecks. Attention is no longer just `softmax(QK^T / sqrt(d))V`; it is an all-pairs kernel smoother with O(T^2) score materialization, O(T^2) probability storage, IO constraints on accelerators, and different train-time and decode-time bottlenecks.
2. From implementation correctness to model class assumptions. Sparse attention assumes useful dependency locality or special global tokens. Linear attention assumes a positive feature map can approximate the exponential kernel well enough. State-space models assume compressible sequence history through a latent recurrence.
3. From paper claims to falsifiable experiments. Each module includes a toy replication target, a baseline, a metric, and a failure mode. "Works on the toy task" is not treated as evidence of frontier-scale superiority.
4. From isolated papers to mathematical families. The same paper may be read as kernel approximation, randomized linear algebra, Markov operator learning, variational prediction, control, or statistical learning under compute constraints.
5. From trend following to research taste. Established results, strong empirical regularities, and speculative directions are separated. The student is expected to critique evaluation validity, data leakage, scaling extrapolations, and hidden systems assumptions.

Research standard: paper references below were checked against primary arXiv pages where arXiv IDs are provided. Author lists are abbreviated with "et al." for long-team papers; exact metadata is linked.

## 2. Prerequisite Map

| Area | What the student needs | Why it matters |
|---|---|---|
| Probability and stochastic processes | conditional probability, martingales at a light level, Markov chains, hidden Markov models, stochastic processes, concentration | language modeling is conditional density estimation; SSMs and world models use latent stochastic dynamics |
| Statistical learning theory | risk, empirical risk, uniform convergence intuition, bias-variance, sample complexity, PAC-Bayes orientation | needed to distinguish empirical scaling laws from guarantees and to reason about evaluation uncertainty |
| Information theory | entropy, cross-entropy, KL, mutual information, variational bounds, rate-distortion | needed for contrastive learning, masked prediction, bottlenecks, compression, and representation objectives |
| Numerical linear algebra | matrix norms, SVD, low-rank approximation, randomized projections, conditioning, Krylov intuition | needed for low-rank attention, Nystrom methods, optimizer stability, and compression |
| Optimization | SGD noise, momentum, Adam/AdamW, preconditioning, learning-rate schedules, constrained optimization | needed for large-scale training, stability, hyperparameter transfer, and scaling experiments |
| Dynamical systems | linear recurrences, stability, discretization, convolution as impulse response, state transition matrices | needed for S4, Mamba, recurrent alternatives, and world models |
| Kernels and functional analysis | RKHS intuition, feature maps, positive kernels, operators, spectral decompositions | needed for kernelized attention, representation learning, and Markov operator views |
| Algorithms and complexity | asymptotic cost, memory hierarchy, randomized algorithms, approximate nearest neighbor search | needed for subquadratic attention, retrieval systems, and inference bottlenecks |
| Distributed systems | data parallelism, tensor parallelism, pipeline parallelism, activation checkpointing, memory bandwidth | needed for training and serving claims; optional for toy code but mandatory for serious critique |

Dependency rule: do not start modules R03-R07 before R01. Do not start R11-R13 before information theory and latent-variable modeling. Do not start R09-R10 without enough optimization and experimental design to fit and critique power laws.

## 3. Module Overview

| ID | Module title | Goal | Math prerequisites | Implementation artifact | Associated papers | Exercises/projects |
|---|---|---|---|---|---|---|
| R01 | Advanced attention mathematics | Treat attention as kernels, operators, and normalized energy models | linear algebra, probability, kernels | exact attention reference with invariance and gradient checks | Vaswani et al. 2017 | derive gradients; inspect entropy and rank |
| R02 | Sparse, local, block, and global attention | Understand structural sparsity as an inductive bias and complexity reduction | graph theory, complexity | sparse mask library and toy copy tasks | Child et al. 2019; Longformer; Big Bird | prove reachable paths; compare masks |
| R03 | Linear and kernelized attention | Replace softmax attention with associative kernel features | kernels, random features | FAVOR-style positive random feature toy | Performer | derive associativity and normalization |
| R04 | Low-rank and randomized attention approximations | Approximate attention matrices with projections or landmarks | SVD, randomized linear algebra | Linformer and Nystrom toy variants | Linformer; Nystromformer | spectral error experiments |
| R05 | KV-cache, memory bandwidth, and inference complexity | Separate train-time O(T^2) from decode-time KV bandwidth | algorithms, systems | KV-cache profiler for MHA, MQA, GQA | Shazeer 2019; GQA; FlashAttention | memory formulas; throughput experiment |
| R06 | Long-context modeling | Study recurrence, retrieval, position, and context evaluation | probability, sequence modeling | synthetic long-range benchmark suite | Transformer-XL; Memorizing Transformers | design contamination-resistant tasks |
| R07 | State-space models and recurrence | Derive convolutional and recurrent views of linear dynamical systems | dynamical systems, numerical linear algebra | diagonal SSM and selective scan toy | S4; Mamba; RWKV; Hyena | stability and impulse response exercises |
| R08 | Mixture-of-experts and conditional computation | Learn sparse parameter activation and load balancing | optimization, probability | top-k router with auxiliary losses | GShard; Switch; Mixtral | routing collapse diagnostics |
| R09 | Scaling laws and empirical science | Fit, critique, and extrapolate empirical laws | statistics, regression, compute accounting | small scaling-law experiment harness | Kaplan et al.; Hoffmann et al. | fit exponents with uncertainty |
| R10 | Optimization at scale | Understand optimizer dynamics, normalization, schedules, parallelism | optimization, systems | AdamW and gradient accumulation lab | AdamW; muP; ZeRO; Megatron-LM | derive AdamW; memory accounting |
| R11 | Representation learning and self-supervised objectives | Compare generative, contrastive, redundancy-reduction, and masked objectives | information theory | SimCLR/BYOL/VICReg toy embeddings | SimCLR; BYOL; Barlow Twins; VICReg; MAE | collapse detection |
| R12 | JEPA-style predictive representation learning | Study prediction in representation space rather than pixel/token space | variational objectives, representation learning | image/sequence JEPA toy | I-JEPA; V-JEPA | derive invariance/collapse tradeoff |
| R13 | World models, latent dynamics, planning, and agency | Connect sequence models to latent transition models and planning | control, latent variables | tiny latent dynamics model with MPC | World Models; Dreamer; MuZero; Decision Transformer | rollout error analysis |
| R14 | Retrieval, memory, and external knowledge systems | Treat memory as an external nonparametric component | nearest neighbors, Bayesian mixtures | kNN-LM/RAG toy retrieval adapter | RAG; RETRO; kNN-LM; NTM | retrieval ablations |
| R15 | Mechanistic interpretability | Analyze learned circuits and causal mechanisms | linear algebra, causal interventions | activation patching and logit lens toy | Transformer Circuits; ROME | causal tracing lab |
| R16 | Evaluation, benchmarks, contamination, and validity | Make evaluation scientifically defensible | statistics, measurement | benchmark audit checklist and contamination probe | HELM; MMLU; BIG-bench; Carlini et al. | design a valid benchmark |
| R17 | Quantization, compression, distillation, and efficient inference | Move from quantization basics to research-grade compression tradeoffs | numerical analysis, information theory | GPTQ/AWQ-inspired layer toy | GPTQ; AWQ; QLoRA; LLM.int8 | error propagation |
| R18 | Research methodology and experiment design | Convert ideas into small, honest research projects | statistics, experiment design | pre-registration template and report scaffold | cross-module | capstone proposal and replication |

## 4. Detailed Modules

### R01. Advanced Attention Mathematics

Motivation: exact attention is the baseline every alternative must beat or approximate. The student should understand it as a normalized kernel operator, not only a tensor recipe.

Core concepts: queries, keys, values, score matrix, Gibbs distribution, causal masking, entropy, rank, residual stream interaction, attention as data-dependent smoothing.

Mathematical objects: `Q, K, V in R^{B x H x T x d}`, score matrix `S = QK^T / sqrt(d)`, mask `M`, row-stochastic matrix `A = softmax(S + M)`, output `Y = AV`.

Required background: matrix calculus, categorical distributions, log-sum-exp, kernels.

Derivation:

```text
For one query q_i, attention solves:
  a_i = argmax_{p in simplex} p^T s_i + H(p)
where s_ij = q_i^T k_j / sqrt(d), H(p) = -sum_j p_j log p_j.

Lagrangian:
  L(p, lambda) = sum_j p_j s_j - sum_j p_j log p_j + lambda(sum_j p_j - 1)
dL/dp_j = s_j - log p_j - 1 + lambda = 0
  p_j = C exp(s_j)
Normalization gives p_j = exp(s_j) / sum_l exp(s_l).
```

This makes softmax attention the entropy-regularized selection of values. Causal masking sets infeasible positions to `-inf`, changing the feasible simplex.

Bottleneck addressed: exact attention maximizes flexible content addressing but materializes all pair interactions.

Assumptions and failure modes: softmax can be too sharp, too diffuse, or numerically unstable; attention weights are not explanations by default; low-rank or sparse patterns may emerge but are not guaranteed.

Toy implementation plan: implement exact attention with explicit masks, log-sum-exp stabilization, entropy diagnostics, rank diagnostics, and finite-difference gradient checks.

Paper reading lab: "Attention Is All You Need" by Vaswani et al., 2017, arXiv:1706.03762. Claim: self-attention plus feed-forward blocks can replace recurrence for sequence transduction. Bottleneck targeted: sequential recurrence and long dependency paths. Mechanism: all-token pairwise content routing plus positional encodings. Evidence: translation benchmarks. Limitations: not a long-context or inference-efficiency paper by modern standards. Toy replication: copy/reverse tasks comparing attention to a small RNN.

Exercises with solutions:

- Exercise: Prove each unmasked attention row sums to 1. Solution: softmax divides each positive `exp(s_j)` by `Z = sum_j exp(s_j)`, so `sum_j exp(s_j)/Z = 1`.
- Exercise: Show why scaling by `sqrt(d)` stabilizes logits when query/key coordinates have variance 1. Solution: the dot product has variance `d`; dividing by `sqrt(d)` gives variance approximately 1, preventing softmax saturation at initialization.

Expected outcomes: the student can derive attention from entropy-regularized optimization, implement it exactly, and state what approximations must preserve.

### R02. Sparse, Local, Block, and Global Attention

Motivation: full attention is often wasteful for long sequences. Sparse attention replaces complete graph connectivity with a designed graph.

Core concepts: local windows, dilated windows, block sparsity, global tokens, random links, graph diameter, receptive field growth.

Mathematical objects: adjacency matrix `G in {0,1}^{T x T}`, masked score matrix `S + log G`, block layouts, path length between tokens.

Required background: graph connectivity, asymptotic complexity.

Derivation:

```text
Full causal attention cost per layer: O(T^2 d).
Local window with width w: each token attends to at most w previous tokens.
Cost: O(T w d).
Stack L layers. Information from token t-k reaches t only if k <= Lw
unless global, random, or dilated edges reduce graph diameter.
```

Bottleneck addressed: O(T^2) score computation and memory.

Assumptions and failure modes: locality may miss long-range dependencies; global tokens can bottleneck; random edges add variance; sparse kernels may be slower than dense kernels at small T.

Toy implementation plan: build mask generators for sliding, block, dilated, global, and random patterns; run synthetic retrieval tasks where the relevant token is near, far, or global-indexed.

Paper reading lab: "Longformer: The Long-Document Transformer" by Beltagy, Peters, and Cohan, 2020, arXiv:2004.05150. Claim: local plus task-motivated global attention enables long-document tasks. Mechanism: sparse attention pattern. Evidence: document QA/classification. Limitation: global-token choices encode task assumptions. Toy replication: classify sequences where the answer depends on a designated global marker.

Exercises with solutions:

- Exercise: A window size `w=4` and `L=3` causal local layers can transmit information how far? Solution: at most 12 positions backward if each layer shifts information by 4 positions.
- Exercise: Why can BigBird-style random/global/local patterns be more expressive than pure local patterns? Solution: global and random edges reduce graph diameter, allowing long-range paths in fewer layers.

Expected outcomes: the student can treat sparse attention as graph design and evaluate whether the task actually matches the graph prior.

### R03. Linear and Kernelized Attention

Motivation: if the softmax kernel can be replaced by a feature inner product, attention can be computed without materializing `T x T`.

Core concepts: kernel feature maps, positive random features, associativity, causal prefix sums, normalization.

Mathematical objects: kernel `k(q,k)`, feature map `phi(x) in R^m`, approximate attention `Y_i = sum_j phi(q_i)^T phi(k_j) v_j / sum_j phi(q_i)^T phi(k_j)`.

Required background: kernels, random features, numerical stability.

Derivation:

```text
If exp(q_i^T k_j) approximately phi(q_i)^T phi(k_j), then
  y_i = [sum_j phi(q_i)^T phi(k_j) v_j] / [sum_j phi(q_i)^T phi(k_j)]
      = [phi(q_i)^T sum_j phi(k_j) v_j^T] / [phi(q_i)^T sum_j phi(k_j)].

For causal attention, replace the sums with prefix sums up to i.
The sequence cost becomes O(T m d_v) instead of O(T^2 d_v).
```

Bottleneck addressed: all-pairs score matrix.

Assumptions and failure modes: approximation quality depends on feature dimension and kernel choice; negative features can break probability interpretation; causal prefix sums can accumulate numerical error; some tasks require exact content selection.

Toy implementation plan: implement linear attention with `elu(x)+1` features, then a positive random feature approximation; compare to exact attention on copy, induction, and smoothing tasks.

Paper reading lab: "Rethinking Attention with Performers" by Choromanski et al., 2020, arXiv:2009.14794. Claim: softmax attention can be approximated by positive orthogonal random features. Mechanism: FAVOR+ feature maps make attention associative. Evidence: accuracy-speed tradeoffs. Limitations: approximation variance and task sensitivity. Toy replication: measure output error against exact attention as feature dimension increases.

Exercises with solutions:

- Exercise: Why does associativity reduce complexity? Solution: compute `K_phi^T V` once as an `m x d_v` matrix, then multiply each query feature by it; no `T x T` matrix is formed.
- Exercise: Give a failure case for low-dimensional features. Solution: two nearly identical keys that exact softmax separates by small score differences may collapse under a low-dimensional random map, causing retrieval errors.

Expected outcomes: the student can derive linear attention and distinguish kernel approximation from sparsity.

### R04. Low-Rank and Randomized Attention Approximations

Motivation: many attention maps have structure. Low-rank methods exploit approximate redundancy across positions.

Core concepts: low-rank projection, landmarks, Nystrom approximation, randomized SVD, spectral error.

Mathematical objects: attention matrix `A`, projections `E, F in R^{k x T}`, landmarks `L`, pseudoinverse `A_LL^+`.

Required background: SVD, Johnson-Lindenstrauss intuition, pseudoinverses.

Derivation:

```text
Best rank-k approximation by Frobenius norm:
  A_k = U_k Sigma_k V_k^T
  ||A - A_k||_F^2 = sum_{r>k} sigma_r^2.

Linformer-style projection approximates:
  softmax(QK^T)V approximately softmax(Q(EK)^T)(FV)
with k projected sequence positions.

Nystrom approximates a kernel matrix using landmarks:
  K approximately K_{T,L} K_{L,L}^+ K_{L,T}.
```

Bottleneck addressed: sequence dimension in attention.

Assumptions and failure modes: attention may not be low-rank for exact retrieval; pseudoinverse can be unstable; low-rank errors may compound across layers.

Toy implementation plan: generate attention matrices from synthetic tasks, inspect singular spectra, implement projected-key/value attention and landmark attention, measure approximation error and downstream loss.

Paper reading lab: "Linformer: Self-Attention with Linear Complexity" by Wang et al., 2020, arXiv:2006.04768. Claim: self-attention can be low-rank in sequence length. Mechanism: learned sequence projections. Evidence: long-sequence benchmarks. Limitations: rank assumptions may be model/task dependent. Toy replication: compare low-rank approximation on local smoothing versus exact lookup tasks.

Exercises with solutions:

- Exercise: Why can low Frobenius error still fail retrieval? Solution: a rare but decisive attention entry can contribute little to total Frobenius error while controlling the correct output token.
- Exercise: What diagnostic should precede using a low-rank approximation? Solution: inspect singular value decay of attention matrices from the target task or a trained baseline.

Expected outcomes: the student can connect low-rank attention to classical approximation theory and test whether the assumption holds.

### R05. KV-Cache, Memory Bandwidth, and Inference Complexity

Motivation: training and autoregressive decoding have different bottlenecks. During decoding, the model reads growing KV cache tensors for every generated token.

Core concepts: prefill vs decode, KV cache, multi-head attention, multi-query attention, grouped-query attention, IO-aware exact attention.

Mathematical objects: cache tensors `K,V in R^{L x B x H_kv x T x d_head}`, bytes per token, bandwidth, arithmetic intensity.

Required background: complexity analysis, basic accelerator memory hierarchy.

Derivation:

```text
Per generated token, each layer reads cached K and V:
  bytes approximately 2 * B * H_kv * T * d_head * bytes_per_element.

MHA: H_kv = H_q.
MQA: H_kv = 1.
GQA: H_kv = number of key/value groups.
Thus MQA/GQA reduce cache bandwidth without changing query head count.
```

Bottleneck addressed: decode-time memory bandwidth and cache footprint.

Assumptions and failure modes: MQA/GQA can reduce quality if heads need distinct memories; speedups depend on kernel implementation and hardware; exact attention can still be IO-bound.

Toy implementation plan: build a cache simulator for MHA/MQA/GQA; measure tokens/sec and bytes/token for small PyTorch models; separate prefill from decode.

Paper reading lab: "Fast Transformer Decoding: One Write-Head is All You Need" by Shazeer, 2019, arXiv:1911.02150. Claim: sharing keys/values across query heads accelerates decoding. Mechanism: reduce KV cache writes/reads. Evidence: machine translation decoding. Limitation: task and model size dependency. Toy replication: implement MQA in `CausalSelfAttention` and compare output drift and cache memory.

Exercises with solutions:

- Exercise: If `H_q=16`, `H_kv=4`, `d_head=64`, fp16, and `T=4096`, what KV bytes per layer per batch element are read per decode step? Solution: `2 * 4 * 4096 * 64 * 2 = 4,194,304` bytes, about 4 MiB.
- Exercise: Why does FlashAttention not change the mathematical attention result? Solution: it changes the tiling and IO schedule for exact softmax attention, not the attention formula.

Expected outcomes: the student can evaluate inference claims with bandwidth accounting rather than parameter count alone.

### R06. Long-Context Modeling

Motivation: long context is not just larger `T`. It requires evaluating whether models use distant information reliably.

Core concepts: segment recurrence, position extrapolation, retrieval memory, long-range synthetic tasks, context utilization.

Mathematical objects: context window `T`, recurrence memory `m_t`, position encoding function `p(t)`, retrieval distribution over external chunks.

Required background: sequence modeling, evaluation design.

Derivation:

```text
Transformer-XL-style recurrence detaches prior segment states:
  h_t^l = f(h_t^{l-1}, [stopgrad(h_{t-T:t-1}^l), h_{t-current}^{l}])
This extends effective context while preventing gradients through all past segments.
```

Bottleneck addressed: fixed context windows and unreliable use of distant tokens.

Assumptions and failure modes: synthetic tasks can overstate ability; near-context shortcuts can hide failure; position interpolation can degrade exact retrieval; recurrence memory may stale.

Toy implementation plan: build tasks for passkey retrieval, repeated-key induction, delayed copy, and conflicting near/far evidence; evaluate accuracy as a function of distance.

Paper reading lab: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" by Dai et al., 2019, arXiv:1901.02860. Claim: segment recurrence and relative positions improve long-context modeling. Mechanism: reuse hidden states from previous segments. Evidence: language modeling benchmarks. Limitation: cached states are not updated by future context. Toy replication: compare fixed-window and recurrent memory on delayed copy.

Exercises with solutions:

- Exercise: Why can low perplexity fail to prove long-context use? Solution: most tokens may be predictable from local context, so aggregate loss hides rare long-range failures.
- Exercise: Design a contamination-resistant passkey test. Solution: sample random keys and values at evaluation time, place the answer once at a controlled distance, and ensure no template-specific shortcut reveals it.

Expected outcomes: the student can design long-context tests that isolate distance-sensitive behavior.

### R07. State-Space Models and Recurrence

Motivation: state-space models compress history into a recurrent latent state and can be evaluated with linear or near-linear sequence cost.

Core concepts: continuous and discrete linear systems, convolution kernels, stability, selective recurrence, scan algorithms.

Mathematical objects: state `x_t`, transition `A`, input matrix `B`, output matrix `C`, step size `Delta`, convolution kernel `K_t = C A^t B`.

Required background: linear dynamical systems, matrix exponentials, convolution.

Derivation:

```text
Continuous system:
  dx/dt = A x(t) + B u(t), y(t) = C x(t).

Discretize with step Delta:
  x_t = A_bar x_{t-1} + B_bar u_t
where A_bar = exp(Delta A).

Unroll:
  y_t = C sum_{i=0}^t A_bar^{t-i} B_bar u_i
So a linear recurrence is also a convolution with impulse response C A_bar^n B_bar.
```

Bottleneck addressed: quadratic attention and finite context.

Assumptions and failure modes: fixed linear dynamics may be weak for content-based reasoning; unstable eigenvalues explode; too much compression loses exact token identity; selective/input-dependent parameters improve expressivity but complicate parallelism.

Toy implementation plan: implement a diagonal SSM, compare recurrent and convolution forms, then add input-dependent gates for selective memory on a synthetic colored-copy task.

Paper reading lab: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" by Gu and Dao, 2023, arXiv:2312.00752. Claim: input-dependent SSM parameters improve content-based sequence modeling while retaining efficient scan. Mechanism: selective propagation/forgetting and hardware-aware recurrent computation. Evidence: language, audio, genomics. Limitation: frontier claims remain workload and implementation dependent. Toy replication: compare fixed SSM and selective SSM on tasks requiring conditional retention.

Exercises with solutions:

- Exercise: What condition on scalar `a` keeps `x_t = a x_{t-1}` stable? Solution: `|a| <= 1` prevents unbounded growth for bounded initial state.
- Exercise: Why is convolution form useful for training? Solution: all outputs can be computed in parallel with convolution/FFT-style methods for linear time-invariant dynamics.

Expected outcomes: the student can derive SSMs from dynamical systems and critique "linear time" claims.

### R08. Mixture-of-Experts and Conditional Computation

Motivation: MoE increases parameter count without activating every parameter for every token.

Core concepts: routers, top-k expert selection, capacity factor, load balancing, expert parallelism.

Mathematical objects: router logits `r_t`, gate probabilities `g_t = softmax(r_t)`, expert functions `E_i`, auxiliary load loss.

Required background: categorical distributions, constrained optimization, distributed systems orientation.

Derivation:

```text
Top-k MoE output:
  y_t = sum_{i in topk(g_t)} normalize(g_{t,i}) E_i(x_t).

A simple load objective penalizes imbalance:
  L_load = N * sum_i f_i p_i
where f_i is fraction of tokens routed to expert i and p_i is mean router probability.
```

Bottleneck addressed: dense compute scaling.

Assumptions and failure modes: routing can collapse; experts can specialize spuriously; capacity overflow drops or reroutes tokens; distributed communication can dominate.

Toy implementation plan: implement top-1 and top-2 feed-forward experts on a synthetic mixture task; plot expert load, overflow, and specialization.

Paper reading lab: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" by Fedus, Zoph, and Shazeer, 2021, arXiv:2101.03961. Claim: top-1 routing simplifies sparse expert scaling. Mechanism: activate one expert per token plus load balancing. Evidence: pretraining efficiency. Limitation: quality and stability depend on routing and systems details. Toy replication: induce routing collapse, then add auxiliary loss.

Exercises with solutions:

- Exercise: Why can MoE have many parameters without proportional FLOPs? Solution: each token uses only selected experts, so inactive expert weights are not multiplied for that token.
- Exercise: What does expert collapse look like? Solution: most tokens route to a few experts, causing high load imbalance, dropped tokens, and undertrained experts.

Expected outcomes: the student can separate parameter scaling from compute scaling and diagnose router pathologies.

### R09. Scaling Laws and Empirical Science of LLMs

Motivation: scaling laws are empirical models, not laws of nature. The student must fit them and critique extrapolation.

Core concepts: power laws, irreducible loss, compute-optimal training, data/model tradeoff, confidence intervals.

Mathematical objects: loss `L(N,D,C)`, parameter count `N`, tokens `D`, compute `C`, fitted exponents.

Required background: regression, uncertainty, compute accounting.

Derivation:

```text
A common fit:
  L(N) = L_inf + a N^{-alpha}
If L_inf is known, log(L - L_inf) = log a - alpha log N.
If L_inf is unknown, nonlinear regression is required and uncertainty grows.

Compute-optimal question:
  minimize L(N,D) subject to C approximately k N D.
The optimum depends on the fitted exponents for N and D.
```

Bottleneck addressed: deciding whether to spend compute on larger models, more data, or better training.

Assumptions and failure modes: small-scale fits may not extrapolate; datasets differ; loss may not predict downstream utility; contaminated eval can distort apparent scaling.

Toy implementation plan: train tiny transformers across parameter/data budgets; fit power laws; bootstrap confidence intervals; compare extrapolated and observed losses.

Paper reading lab: "Training Compute-Optimal Large Language Models" by Hoffmann et al., 2022, arXiv:2203.15556. Claim: prior large LMs were often undertrained relative to data. Mechanism: empirical compute-optimal frontier. Evidence: Chinchilla experiments. Limitation: result is empirical and depends on data quality, architecture, and objective. Toy replication: fixed compute budget sweep over model size and token count.

Exercises with solutions:

- Exercise: Why is fitting `L_inf` hard? Solution: it is correlated with the exponent and intercept; small errors in assumed irreducible loss change slope estimates.
- Exercise: What is the smallest honest scaling experiment? Solution: at least a grid over model size and token budget with repeated seeds for selected points and held-out evaluation.

Expected outcomes: the student can produce scaling plots with uncertainty and avoid overclaiming extrapolations.

### R10. Optimization at Scale

Motivation: large-model training is often limited by optimizer stability, memory, and parallelism rather than model equations alone.

Core concepts: AdamW, gradient clipping, learning-rate schedules, warmup, normalization, accumulation, mixed precision, ZeRO-style partitioning.

Mathematical objects: gradients `g_t`, moments `m_t, v_t`, parameters `theta`, optimizer states, memory terms.

Required background: stochastic optimization, numerical precision, systems accounting.

Derivation:

```text
Adam moments:
  m_t = beta1 m_{t-1} + (1-beta1) g_t
  v_t = beta2 v_{t-1} + (1-beta2) g_t^2
  theta_{t+1} = theta_t - eta m_hat_t / (sqrt(v_hat_t) + eps)

AdamW decouples weight decay:
  theta_{t+1} = theta_t - eta update - eta lambda theta_t
instead of mixing L2 penalty into the adaptive gradient.
```

Bottleneck addressed: stable and memory-efficient optimization at large scale.

Assumptions and failure modes: adaptive methods can hide poor initialization; mixed precision can overflow; gradient accumulation changes effective batch behavior; optimizer state can exceed parameter memory.

Toy implementation plan: implement AdamW from scratch, compare to PyTorch, add gradient accumulation and mixed-precision guards, estimate memory per parameter.

Paper reading lab: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" by Yang et al., 2022, arXiv:2203.03466. Claim: maximal update parametrization helps transfer hyperparameters across widths. Mechanism: parameterization chosen so update scales remain stable with width. Evidence: width transfer experiments. Limitations: not a replacement for all tuning and not architecture-universal. Toy replication: compare learning-rate transfer across MLP widths.

Exercises with solutions:

- Exercise: Why does Adam require more memory than SGD? Solution: Adam stores parameters, gradients, first moments, and second moments, roughly several tensors per parameter.
- Exercise: Why is decoupled weight decay different from L2 under Adam? Solution: adaptive preconditioning rescales gradient components, so adding `lambda theta` to the gradient does not equal a uniform shrinkage of parameters.

Expected outcomes: the student can derive optimizer updates and account for training memory.

### R11. Representation Learning and Self-Supervised Objectives

Motivation: not all useful objectives predict the next token. Representation learning studies invariances, predictive features, collapse, and information retention.

Core concepts: contrastive learning, negative samples, redundancy reduction, variance preservation, masked reconstruction.

Mathematical objects: encoders `f`, projections `z`, augmentations `t(x)`, InfoNCE loss, covariance matrix.

Required background: information theory, covariance, optimization.

Derivation:

```text
InfoNCE for positive pair (i,j):
  L_i = -log exp(sim(z_i,z_j)/tau) / sum_k exp(sim(z_i,z_k)/tau).

It classifies the positive among negatives. More negatives can tighten mutual-information-related bounds, but the objective also learns augmentation invariances, not pure information preservation.
```

Bottleneck addressed: learning useful abstractions without labels or exact generative reconstruction.

Assumptions and failure modes: augmentations define invariance and can discard needed information; noncontrastive methods can collapse without variance/covariance constraints; reconstruction may waste capacity on nuisance details.

Toy implementation plan: build a 2D shapes dataset with known factors; compare contrastive, BYOL-style, VICReg-style, and masked reconstruction embeddings using linear probes.

Paper reading lab: "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" by Bardes, Ponce, and LeCun, 2021, arXiv:2105.04906. Claim: invariance, variance, and covariance terms can avoid collapse without negatives. Mechanism: match views while preserving per-dimension variance and decorrelating features. Evidence: vision benchmarks. Limitation: augmentation and architecture dependence. Toy replication: remove variance term and observe collapse.

Exercises with solutions:

- Exercise: What is representation collapse? Solution: the encoder maps many inputs to nearly identical vectors, making downstream distinctions impossible.
- Exercise: Why can masked autoencoding learn useful features? Solution: predicting missing content from visible context pressures the model to encode structure, though it may also model low-level details.

Expected outcomes: the student can compare self-supervised objectives by their invariances and anti-collapse mechanisms.

### R12. JEPA-Style Predictive Representation Learning

Motivation: JEPA predicts target representations from context representations rather than reconstructing raw observations.

Core concepts: context encoder, target encoder, predictor, latent prediction, masking, stop-gradient or EMA targets.

Mathematical objects: context `x_c`, target `x_t`, encoders `f_theta, f_bar`, predictor `p`, representation loss `||p(f_theta(x_c)) - f_bar(x_t)||`.

Required background: self-supervised learning, variational thinking, representation collapse.

Derivation:

```text
Raw reconstruction objective:
  minimize E ||decoder(z_c) - x_t||^2.

JEPA-style objective:
  minimize E ||p(f_context(x_c)) - stopgrad(f_target(x_t))||^2.

The prediction target is an embedding, so the model can ignore unpredictable pixel/token-level detail if the target encoder does not preserve it.
```

Bottleneck addressed: generative objectives spend capacity modeling irrelevant details; contrastive objectives can depend heavily on negatives.

Assumptions and failure modes: target representations must be meaningful; collapse prevention is essential; evaluation must show downstream utility rather than just low latent loss.

Toy implementation plan: masked sequence or image-patch prediction where nuisance noise is high; compare raw reconstruction and latent prediction on downstream factor classification.

Paper reading lab: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" by Assran et al., 2023, arXiv:2301.08243. Claim: predicting image patch representations can learn semantic features without pixel reconstruction. Mechanism: context-to-target latent prediction with masking. Evidence: vision transfer. Limitation: not direct evidence for language/world-model superiority. Toy replication: latent prediction on synthetic images with nuisance texture.

Exercises with solutions:

- Exercise: Why is stop-gradient useful in noncontrastive latent prediction? Solution: it prevents both sides from chasing each other toward a trivial constant solution through symmetric updates.
- Exercise: What must be evaluated beyond JEPA training loss? Solution: downstream representation quality, invariance retention, collapse metrics, and robustness to nuisance variation.

Expected outcomes: the student can distinguish predictive representation learning from autoregressive likelihood.

### R13. World Models, Latent Dynamics, Planning, and Agency

Motivation: language models predict tokens; world models learn latent transitions that can support planning under actions.

Core concepts: latent state, transition model, observation model, reward model, model-predictive control, imagination rollouts.

Mathematical objects: latent `z_t`, action `a_t`, transition `p(z_{t+1}|z_t,a_t)`, observation `p(o_t|z_t)`, reward `r_t`, policy `pi(a|z)`.

Required background: Markov decision processes, variational inference, control.

Derivation:

```text
Latent model factorization:
  p(o_{1:T}, z_{1:T} | a_{1:T})
  = product_t p(o_t | z_t) p(z_t | z_{t-1}, a_{t-1}).

Planning uses imagined rollouts:
  choose a_{t:t+H} to maximize E[sum_{h=0}^H gamma^h r(z_{t+h}, a_{t+h})].
```

Bottleneck addressed: next-token models do not by themselves define grounded state, action, transition, or objective.

Assumptions and failure modes: learned dynamics compound errors; reward misspecification leads to bad planning; latent state may omit task-relevant variables; agency claims are often underdefined.

Toy implementation plan: train a latent dynamics model on a gridworld or simple physics task; compare one-step prediction, rollout prediction, and planning success.

Paper reading lab: "Dream to Control: Learning Behaviors by Latent Imagination" by Hafner et al., 2019, arXiv:1912.01603. Claim: policies can be learned from imagined trajectories in a latent world model. Mechanism: recurrent state-space model plus actor-critic in latent space. Evidence: continuous-control benchmarks. Limitation: model errors and benchmark scope. Toy replication: latent MPC in a small gridworld.

Exercises with solutions:

- Exercise: Why can one-step prediction be good while planning fails? Solution: small one-step errors compound over imagined horizons and can bias action selection.
- Exercise: What extra variable distinguishes a world model from a passive sequence model? Solution: actions; the model must predict how interventions change future states.

Expected outcomes: the student can state what is required before calling a model a world model or agent.

### R14. Retrieval, Memory, and External Knowledge Systems

Motivation: parametric memory is expensive to update and hard to inspect. Retrieval augments a model with external nonparametric memory.

Core concepts: dense retrieval, nearest neighbors, retrieval-augmented generation, datastore LMs, memory reads/writes.

Mathematical objects: query encoder `q(x)`, key-value datastore `(k_i, v_i)`, retrieval distribution `p(i|x)`, mixture distribution over tokens.

Required background: nearest-neighbor search, Bayesian mixtures, information retrieval metrics.

Derivation:

```text
kNN-LM mixture:
  p(y|x) = lambda p_LM(y|x) + (1-lambda) p_kNN(y|x)
where p_kNN is induced by distances from q(x) to datastore keys.

RAG marginalizes over retrieved documents:
  p(y|x) = sum_z p_eta(z|x) p_theta(y|x,z).
```

Bottleneck addressed: static parametric knowledge, hallucination, and context limits.

Assumptions and failure modes: retriever recall bounds generator quality; stale or poisoned corpora mislead; retrieved text can create spurious authority; evaluation may leak answers through corpus overlap.

Toy implementation plan: add a kNN datastore to the tiny LM; build a RAG toy QA system over local notes; ablate retriever quality and context length.

Paper reading lab: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al., 2020, arXiv:2005.11401. Claim: retrieval plus generation improves knowledge-intensive tasks. Mechanism: latent document retrieval marginalization. Evidence: open-domain QA. Limitation: retrieval quality and corpus contamination. Toy replication: synthetic QA where facts are updated after model training.

Exercises with solutions:

- Exercise: Why can retrieval reduce hallucination but not eliminate it? Solution: the generator can ignore, misread, or overgeneralize retrieved evidence; retrieval may also be wrong.
- Exercise: What is the right baseline for a RAG toy? Solution: closed-book LM, oracle retrieval, noisy retrieval, and retrieval-only answer matching.

Expected outcomes: the student can treat retrieval as a probabilistic/system component with measurable failure modes.

### R15. Mechanistic Interpretability

Motivation: interpretability asks for causal mechanisms inside trained models, not only behavior descriptions.

Core concepts: residual stream, logit lens, activation patching, causal tracing, circuits, feature superposition, model editing.

Mathematical objects: activations `h_l`, residual updates, unembedding matrix `W_U`, intervention `do(h_l = h_l')`, causal effect on logit differences.

Required background: linear algebra, causal intervention logic.

Derivation:

```text
For logits y = W_U h_L, a residual component delta contributes:
  Delta logits = W_U delta.

Activation patching estimates causal effect:
  effect = metric(model with h_l patched from clean run)
           - metric(corrupted run).
```

Bottleneck addressed: black-box behavior and inability to localize learned computations.

Assumptions and failure modes: interpretability tools can be misleading under distributed representations; patching can create out-of-distribution activations; found circuits may not be complete.

Toy implementation plan: train a tiny transformer on an induction-head task; inspect attention heads, patch activations, and measure causal recovery.

Paper reading lab: "Locating and Editing Factual Associations in GPT" by Meng et al., 2022, arXiv:2202.05262. Claim: factual associations can be localized and edited in MLP representations. Mechanism: causal tracing and rank-one model editing. Evidence: factual recall edits. Limitation: edit locality and side effects remain hard. Toy replication: train subject-relation-object facts and edit one association.

Exercises with solutions:

- Exercise: Why is attention visualization insufficient for causal explanation? Solution: attention weights show routing probabilities, but changing those activations may or may not change the output.
- Exercise: What makes a patching experiment causal? Solution: it intervenes on an internal variable and measures output metric change relative to controlled clean/corrupt runs.

Expected outcomes: the student can design small causal interpretability experiments and avoid over-reading visualizations.

### R16. Evaluation, Benchmarks, Contamination, and Scientific Validity

Motivation: evaluation is part of the scientific claim. Bad benchmarks produce false progress.

Core concepts: benchmark validity, contamination, memorization, uncertainty, multiple comparisons, task construction, capability extrapolation.

Mathematical objects: estimator `hat m`, confidence interval, contamination event `C`, conditional performance `P(success | C)`.

Required background: statistics, experimental design.

Derivation:

```text
For accuracy over n independent items:
  SE approximately sqrt(p_hat(1-p_hat)/n).
Comparing many models/tasks inflates false discovery risk unless comparisons are planned or corrected.
```

Bottleneck addressed: unreliable claims about model capability.

Assumptions and failure modes: benchmarks saturate; prompts leak answers; training data overlap contaminates results; aggregate scores hide subgroup failures.

Toy implementation plan: build a mini benchmark with generated train/test splits, contamination probes, confidence intervals, and adversarial variants.

Paper reading lab: "Holistic Evaluation of Language Models" by Liang et al., 2022, arXiv:2211.09110. Claim: LMs require broad, transparent, multi-metric evaluation. Mechanism: scenario/metric taxonomy. Evidence: HELM benchmark suite. Limitation: benchmark coverage is still incomplete and can age. Toy replication: evaluate tiny models across accuracy, calibration, robustness, and efficiency.

Exercises with solutions:

- Exercise: If two models differ by 1 percent on 100 examples, what is the likely issue? Solution: the standard error is several percentage points near p=0.5, so the difference is not reliable without more data or paired testing.
- Exercise: Name one contamination test. Solution: search benchmark examples or high-order n-grams against the training corpus or retrieval corpus, then report contaminated and clean scores separately.

Expected outcomes: the student can audit a benchmark before trusting it.

### R17. Quantization, Compression, Distillation, and Efficient Inference

Motivation: efficient inference is not just smaller weights. It involves quantization error, activation outliers, cache size, kernels, and quality tradeoffs.

Core concepts: post-training quantization, groupwise quantization, Hessian-aware reconstruction, activation-aware scaling, distillation, pruning.

Mathematical objects: weight matrix `W`, quantizer `Q(W)`, scale `s`, zero-point `z`, reconstruction error `||WX - Q(W)X||`.

Required background: numerical linear algebra, optimization, information theory.

Derivation:

```text
Uniform affine quantization:
  q = clip(round(x/s) + z, q_min, q_max)
  x_hat = s(q - z).

Layer reconstruction view:
  minimize ||WX - W_q X||_2^2
where X are representative activations. GPTQ-style methods use curvature
information to choose quantization order and compensate errors.
```

Bottleneck addressed: memory footprint, bandwidth, and deployment cost.

Assumptions and failure modes: calibration data may not represent deployment; activation outliers dominate error; lower memory does not guarantee lower latency; kernels decide real speed.

Toy implementation plan: quantize a linear layer per-tensor, per-channel, and groupwise; compare weight error, output error, and generated-token degradation.

Paper reading lab: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" by Lin et al., 2023, arXiv:2306.00978. Claim: protecting activation-salient weights improves low-bit quantization. Mechanism: use activation statistics to scale channels before quantization. Evidence: LLM compression benchmarks. Limitation: calibration and kernel support matter. Toy replication: identify high-activation channels and compare naive 4-bit vs protected-channel quantization.

Exercises with solutions:

- Exercise: Why can per-channel quantization beat per-tensor quantization? Solution: each output channel gets its own scale, reducing error when channel ranges differ.
- Exercise: Why can quantization improve memory but not latency? Solution: without optimized kernels, dequantization overhead and memory layout can erase arithmetic savings.

Expected outcomes: the student can evaluate compression by error, quality, memory, and actual speed.

### R18. Research Methodology and Experiment Design

Motivation: research contributions require disciplined claims, not just novel code.

Core concepts: hypothesis, baseline, ablation, replication, preregistration, error bars, negative results, compute budget.

Mathematical objects: estimator, confidence interval, effect size, experimental factor, confounder.

Required background: statistics, software engineering, scientific writing.

Derivation:

```text
Paired comparison:
  d_i = metric_A(i) - metric_B(i)
  report mean(d), standard error over items/seeds, and the exact test set.
This is often more sensitive than comparing two independent aggregate scores.
```

Bottleneck addressed: weak evidence and irreproducible claims.

Assumptions and failure modes: too many ablations after seeing results; cherry-picked seeds; hidden data changes; compute budget too small for the stated claim.

Toy implementation plan: take one earlier module result, write a one-page experiment plan, run a baseline and one variant, report negative or positive results with uncertainty.

Paper reading lab: choose one failed or ambiguous replication from the course. The lab output must separate "paper claim", "toy claim", and "what our evidence supports."

Exercises with solutions:

- Exercise: What is a minimal ablation? Solution: remove or isolate the proposed mechanism while keeping data, model size, training steps, and evaluation fixed.
- Exercise: What should be logged for reproducibility? Solution: code version, data version, seeds, hyperparameters, hardware, runtime profile, metrics, and generated artifacts.

Expected outcomes: the student can propose a small research direction with a falsifiable experiment.

## 5. Notebook Plan

The research layer should be implemented as a second notebook sequence, leaving existing notebooks `00` to `12` intact.

| Notebook | Purpose | Concepts covered | Derivations included | Code artifacts | Tests/checks | Exercises |
|---|---|---|---|---|---|---|
| `13_research_orientation.ipynb` | Bridge from build curriculum to research method | paper anatomy, bottlenecks, claims | claim-evidence-limitation schema | paper lab template | checklist validation | critique one known paper |
| `14_attention_as_kernel_operator.ipynb` | Re-derive exact attention | entropy view, kernels, rank | softmax as entropy-regularized optimum | exact attention diagnostics | row sums, finite differences | gradient and entropy tasks |
| `15_sparse_attention_patterns.ipynb` | Compare sparse masks | local, block, global, random | complexity and graph reachability | mask generator library | reachability tests | design mask for task |
| `16_linear_kernel_attention.ipynb` | Implement linear attention | feature maps, prefix sums | associativity derivation | linear attention module | compare to exact attention | feature-dimension sweep |
| `17_low_rank_attention.ipynb` | Test low-rank assumptions | SVD, projections, landmarks | Eckart-Young and Nystrom formulas | Linformer/Nystrom toys | spectral diagnostics | retrieval failure case |
| `18_kv_cache_inference.ipynb` | Measure decode bottlenecks | prefill, decode, MQA, GQA | cache memory formulas | cache simulator | byte accounting | MHA vs MQA experiment |
| `19_long_context_evaluation.ipynb` | Build long-context tests | passkey, induction, delayed copy | distance-conditioned metrics | benchmark generator | contamination checks | design robust task |
| `20_state_space_models.ipynb` | Derive SSMs | recurrence, convolution, stability | discretization and impulse response | diagonal SSM | recurrent vs convolution equality | stability sweep |
| `21_selective_recurrence_mamba_toy.ipynb` | Add input-dependent memory | selective gates, scan | gated recurrence | selective SSM toy | conditional copy accuracy | failure under noise |
| `22_moe_conditional_compute.ipynb` | Build top-k experts | routing, load balancing | router loss | MoE MLP | load histograms | collapse and fix |
| `23_scaling_laws_lab.ipynb` | Fit small scaling laws | power laws, compute | nonlinear fit | training sweep harness | bootstrap intervals | bad extrapolation critique |
| `24_optimization_at_scale.ipynb` | Study optimizer and memory | AdamW, schedules, precision | AdamW and memory accounting | optimizer lab | match PyTorch update | accumulation exercise |
| `25_self_supervised_objectives.ipynb` | Compare SSL losses | InfoNCE, BYOL, VICReg, MAE | InfoNCE and covariance terms | synthetic representation lab | collapse metrics | remove anti-collapse term |
| `26_jepa_predictive_representations.ipynb` | Implement JEPA toy | latent prediction, target encoder | latent vs raw reconstruction | sequence/image JEPA toy | downstream probes | nuisance-invariance test |
| `27_world_models_and_planning.ipynb` | Build latent dynamics | state, action, reward, MPC | latent factorization | gridworld world model | rollout error | planning horizon sweep |
| `28_retrieval_memory_systems.ipynb` | Add external memory | kNN-LM, RAG | mixture distributions | datastore adapter | retrieval recall | noisy retrieval ablation |
| `29_mechanistic_interpretability.ipynb` | Analyze tiny circuits | patching, logit lens | residual contribution | induction-head toy | causal patch metric | attention vs causality |
| `30_evaluation_validity.ipynb` | Audit benchmarks | uncertainty, contamination | standard errors | benchmark audit toolkit | clean/dirty split | paired comparison |
| `31_compression_research.ipynb` | Quantization beyond basics | GPTQ/AWQ ideas | quantization error objective | groupwise quant lab | output error | calibration shift |
| `32_research_capstone_workshop.ipynb` | Produce proposal | experiment design | paired test and effect size | report scaffold | prereg checklist | capstone pitch |

Notebook implementation rule: every notebook must have a quick profile that runs in smoke mode, and any expensive sweep must be behind an explicit `RUN_STRETCH = False` guard.

## 6. Exercises And Solutions Plan

Exercise categories:

| Category | Purpose | Solution standard |
|---|---|---|
| Derivation exercises | Fill missing math steps | show equations and state assumptions |
| Proof sketches | Build rigor without full theorem burden | state claim, assumptions, proof idea, counterexample |
| Shape reasoning | Prevent tensor cargo-culting | list every tensor dimension before code |
| Implementation gaps | Make mechanisms real | provide reference code in solution file |
| Debugging tasks | Build research engineering skill | identify symptom, root cause, minimal fix |
| Paper critique | Train claim evaluation | claim, mechanism, evidence, limitation |
| Experiment design | Prevent vague projects | hypothesis, baseline, metric, ablation, stop rule |
| Open-ended prompts | Encourage research taste | require bounded scope and falsifiable outcome |

Solution files should mirror notebooks:

```text
exercises/research/13_research_orientation.md
exercises/research/solutions/13_research_orientation_solutions.md
...
exercises/research/32_research_capstone_workshop.md
exercises/research/solutions/32_research_capstone_workshop_solutions.md
```

Minimum solution quality:

1. Derivations include intermediate algebra, not only final formulas.
2. Code solutions include shape assertions and deterministic seeds.
3. Experiment-design solutions name a baseline and one possible confounder.
4. Paper-critique solutions include one limitation and one toy replication.

## 7. Capstone Projects

1. Sparse attention validity study. Implement local, global, random, and block masks; evaluate on controlled long-range tasks; report which dependency graph succeeds and why. Ambition: low. Risk: toy tasks may be too artificial.
2. Linear attention approximation audit. Implement exact and kernelized attention; measure output error, training loss, and retrieval accuracy as feature dimension changes. Ambition: medium. Risk: implementation instability can be mistaken for theory failure.
3. Selective SSM toy research note. Compare fixed diagonal SSM, gated SSM, and tiny transformer on conditional retention tasks. Ambition: medium. Risk: small tasks may favor one architecture unfairly.
4. Retrieval vs parametric memory update experiment. Train a tiny LM on facts, change facts externally, then compare fine-tuning, kNN-LM, and RAG-style retrieval. Ambition: medium-high. Risk: retrieval corpus design can leak answer format.
5. JEPA vs autoregressive objective under nuisance noise. Create a synthetic environment with semantic state plus high-entropy nuisance details; compare latent prediction, reconstruction, and next-token prediction on downstream state probes. Ambition: high. Risk: representation metrics must be chosen before seeing results.

Final capstone deliverable:

```text
docs/research/capstones/<slug>/
  proposal.md
  experiment_plan.md
  results.md
  figures/
  notebooks/
  reproduction.md
```

The claim must be small: "On this controlled task, mechanism X improves metric Y under condition Z." It must not claim frontier architecture superiority.

## 8. Reading List

Each paper is included because it anchors a mechanism taught above. Links point to primary arXiv pages unless noted.

### Attention, Sparse Attention, and Efficient Exact Attention

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017, arXiv:1706.03762 | baseline transformer mechanism | entropy-normalized kernel smoothing | exact attention vs RNN copy task |
| [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509), Child, Gray, Radford, and Sutskever, 2019, arXiv:1904.10509 | sparse attention patterns | factorized/block sparse graphs | compare graph reachability |
| [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150), Beltagy, Peters, and Cohan, 2020, arXiv:2004.05150 | local plus global attention | task-specific sparse adjacency | global-token classification toy |
| [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062), Zaheer et al., 2020, arXiv:2007.14062 | sparse patterns with theoretical motivation | local, global, random graph connectivity | random-edge long retrieval |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., 2022, arXiv:2205.14135 | exact attention can be IO optimized | tiling, online softmax | compare memory materialization |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), Dao, 2023, arXiv:2307.08691 | systems details alter speed | work partitioning | profile exact kernels conceptually |

### Linear, Low-Rank, and Randomized Attention

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451), Kitaev, Kaiser, and Levskaya, 2020, arXiv:2001.04451 | locality-sensitive hashing attention | randomized nearest-neighbor buckets | hash collisions vs retrieval |
| [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768), Wang et al., 2020, arXiv:2006.04768 | low-rank sequence projection | spectral approximation | singular spectrum lab |
| [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), Choromanski et al., 2020, arXiv:2009.14794 | kernelized linear attention | positive random features | approximation error vs feature count |
| [Nystr\"omformer: A Nystr\"om-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902), Xiong et al., 2021, arXiv:2102.03902 | landmark approximation | Nystrom kernel approximation | landmark count sweep |

### Inference, KV Cache, and Long Context

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), Shazeer, 2019, arXiv:1911.02150 | MQA and decode bandwidth | shared key/value heads | cache memory simulator |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), Ainslie et al., 2023, arXiv:2305.13245 | compromise between MHA and MQA | grouped KV sharing | output drift vs groups |
| [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860), Dai et al., 2019, arXiv:1901.02860 | segment recurrence | detached memory states | delayed copy |
| [Memorizing Transformers](https://arxiv.org/abs/2203.08913), Wu, Rabe, Hutchins, and Szegedy, 2022, arXiv:2203.08913 | nonparametric memory inside transformer | approximate nearest-neighbor memory | datastore recall |

### State-Space, Recurrence, and Post-Transformer Backbones

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396), Gu, Goel, and Re, 2021, arXiv:2111.00396 | S4 foundation | structured linear dynamical systems | recurrence/convolution equivalence |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752), Gu and Dao, 2023, arXiv:2312.00752 | selective SSM | input-dependent state transitions | conditional retention task |
| [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866), Poli et al., 2023, arXiv:2302.10866 | long convolution alternative | implicit filters | long convolution copy task |
| [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048), Peng et al., 2023, arXiv:2305.13048 | recurrent transformer-like design | time-mixing recurrence | recurrent vs attention memory |

### Conditional Computation and Scaling

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668), Lepikhin et al., 2020, arXiv:2006.16668 | expert parallel scaling | conditional computation and sharding | expert load experiment |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961), Fedus, Zoph, and Shazeer, 2021, arXiv:2101.03961 | top-1 MoE | load-balanced routing | routing collapse lab |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088), Jiang et al., 2024, arXiv:2401.04088 | modern sparse MoE LLM | sparse expert activation | top-2 router toy |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), Kaplan et al., 2020, arXiv:2001.08361 | empirical scaling | power-law fits | tiny scaling sweep |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann et al., 2022, arXiv:2203.15556 | data/model compute frontier | constrained optimization | compute allocation sweep |

### Optimization and Systems

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), Loshchilov and Hutter, 2017, arXiv:1711.05101 | AdamW | decoupled shrinkage | compare Adam L2 vs AdamW |
| [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466), Yang et al., 2022, arXiv:2203.03466 | width transfer | parametrization and update scales | learning-rate transfer |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), Rajbhandari, Rasley, Ruwase, and He, 2019, arXiv:1910.02054 | optimizer-state partitioning | memory decomposition | parameter-state accounting |
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), Shoeybi et al., 2019, arXiv:1909.08053 | model parallelism | tensor partitioning | matrix multiply split toy |

### Self-Supervised, JEPA, and World Models

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), Chen, Kornblith, Norouzi, and Hinton, 2020, arXiv:2002.05709 | contrastive baseline | InfoNCE | augmentation invariance |
| [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733), Grill et al., 2020, arXiv:2006.07733 | noncontrastive SSL | target networks | collapse ablation |
| [Barlow Twins](https://arxiv.org/abs/2103.03230), Zbontar et al., 2021, arXiv:2103.03230 | redundancy reduction | cross-correlation objective | covariance diagnostics |
| [VICReg](https://arxiv.org/abs/2105.04906), Bardes, Ponce, and LeCun, 2021, arXiv:2105.04906 | anti-collapse terms | variance/invariance/covariance | remove variance term |
| [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), He et al., 2021, arXiv:2111.06377 | masked prediction | reconstruction under high masking | patch reconstruction |
| [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), Assran et al., 2023, arXiv:2301.08243 | I-JEPA | latent target prediction | nuisance-noise task |
| [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471), Bardes et al., 2024, arXiv:2404.08471 | V-JEPA | video feature prediction | temporal latent prediction |
| [World Models](https://arxiv.org/abs/1803.10122), Ha and Schmidhuber, 2018, arXiv:1803.10122 | compact latent dynamics | VAE plus RNN controller | gridworld latent rollout |
| [Dream to Control](https://arxiv.org/abs/1912.01603), Hafner et al., 2019, arXiv:1912.01603 | latent imagination | recurrent state-space model | MPC with learned dynamics |
| [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265), Schrittwieser et al., 2019, arXiv:1911.08265 | MuZero | planning with learned dynamics | tiny tree search |
| [Decision Transformer](https://arxiv.org/abs/2106.01345), Chen et al., 2021, arXiv:2106.01345 | sequence modeling for RL | return-conditioned trajectories | offline gridworld trajectories |

### Retrieval, Interpretability, Evaluation, and Compression

| Paper | Why it belongs | Mathematical idea to extract | Toy experiment |
|---|---|---|---|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), Lewis et al., 2020, arXiv:2005.11401 | RAG | latent document marginalization | update facts externally |
| [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426), Borgeaud et al., 2021, arXiv:2112.04426 | RETRO | chunk retrieval | retrieval chunk ablation |
| [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172), Khandelwal et al., 2019, arXiv:1911.00172 | kNN-LM | nonparametric token mixture | datastore language model |
| [Neural Turing Machines](https://arxiv.org/abs/1410.5401), Graves, Wayne, and Danihelka, 2014, arXiv:1410.5401 | differentiable memory | content/addressed reads and writes | copy with external memory |
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain, Ganguli, Hatfield-Dodds, Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, and Olah, 2021, primary web source | transformer circuit analysis | residual stream and attention-head algebra | induction-head circuit lab |
| [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), Meng et al., 2022, arXiv:2202.05262 | model editing | causal tracing, rank-one edits | factual association edit |
| [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110), Liang et al., 2022, arXiv:2211.09110 | evaluation taxonomy | multi-metric evaluation | mini HELM-style report |
| [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), Hendrycks et al., 2020, arXiv:2009.03300 | benchmark design | multitask accuracy | small domain benchmark |
| [Beyond the Imitation Game](https://arxiv.org/abs/2206.04615), Srivastava et al., 2022, arXiv:2206.04615 | broad task suite | capability extrapolation | task-family stress test |
| [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805), Carlini et al., 2020, arXiv:2012.07805 | memorization risk | exposure and extraction | canary memorization toy |
| [GPTQ](https://arxiv.org/abs/2210.17323), Frantar, Ashkboos, Hoefler, and Alistarh, 2022, arXiv:2210.17323 | post-training quantization | Hessian-aware layer reconstruction | quantize one linear layer |
| [AWQ](https://arxiv.org/abs/2306.00978), Lin et al., 2023, arXiv:2306.00978 | activation-aware quantization | salient channel protection | calibration shift |
| [QLoRA](https://arxiv.org/abs/2305.14314), Dettmers, Pagnoni, Holtzman, and Zettlemoyer, 2023, arXiv:2305.14314 | quantized finetuning | low-rank adapters over 4-bit base | tiny adapter experiment |
| [LLM.int8()](https://arxiv.org/abs/2208.07339), Dettmers, Lewis, Belkada, and Zettlemoyer, 2022, arXiv:2208.07339 | outlier-aware int8 inference | mixed precision decomposition | outlier channel lab |

## 9. Build Plan

This is an implementation plan for adding the research layer inside the existing project. It is intentionally staged so the first useful artifact is documentation, then runnable notebooks, then tested reusable utilities.

### Phase 1: Documentation spine

Files:

```text
docs/notes/research_level_curriculum.md
docs/notes/curriculum_index.md
README.md
```

Actions:

1. Add this curriculum document.
2. Link it from `docs/notes/curriculum_index.md`.
3. Add a short README note that the research layer is a planned extension, not part of the current smoke-tested college curriculum.

Acceptance checks:

```bash
rg -n "Research-Level" docs/notes README.md
```

### Phase 2: Research exercise structure

Files:

```text
exercises/research/
exercises/research/solutions/
```

Actions:

1. Add one prompt and solution file per research notebook.
2. Start with R01-R05 before broadening into JEPA/world-model modules.
3. Keep every solution deterministic and small enough to self-check.

Acceptance checks:

```bash
rg -n "Solution" exercises/research/solutions
```

### Phase 3: Utility modules

Files:

```text
src/llm_from_scratch/research_attention.py
src/llm_from_scratch/research_sequence.py
src/llm_from_scratch/research_eval.py
tests/test_research_attention.py
tests/test_research_sequence.py
tests/test_research_eval.py
```

Actions:

1. Add sparse mask generators and tests for causality/reachability.
2. Add linear attention and low-rank attention reference implementations.
3. Add KV-cache accounting helpers.
4. Add synthetic long-context task generators.

Acceptance checks:

```bash
uv run python -m pytest tests/test_research_attention.py tests/test_research_sequence.py tests/test_research_eval.py -q
```

### Phase 4: Notebook sequence

Files:

```text
notebooks/13_research_orientation.ipynb
...
notebooks/32_research_capstone_workshop.ipynb
scripts/smoke_notebooks.py
```

Actions:

1. Add notebooks in clusters: R01-R05, R06-R10, R11-R14, R15-R18.
2. Extend smoke checks only after each notebook has a quick profile.
3. Keep long sweeps disabled by default.

Acceptance checks:

```bash
uv run python scripts/smoke_notebooks.py --quick
```

### Phase 5: Capstone scaffolding

Files:

```text
docs/research/capstones/README.md
docs/research/templates/proposal.md
docs/research/templates/experiment_plan.md
docs/research/templates/results.md
```

Actions:

1. Add templates that force hypothesis, baseline, ablation, metric, compute budget, and failure criteria.
2. Add one completed miniature example using a sparse attention or KV-cache experiment.

Acceptance checks:

```bash
rg -n "Hypothesis|Baseline|Ablation|Failure criteria" docs/research
```

### Recommended first implementation slice

Build R01-R05 first:

1. `14_attention_as_kernel_operator.ipynb`
2. `15_sparse_attention_patterns.ipynb`
3. `16_linear_kernel_attention.ipynb`
4. `17_low_rank_attention.ipynb`
5. `18_kv_cache_inference.ipynb`

Reason: these modules directly extend the existing attention, quantization, and beyond-transformers material, and they produce reusable utilities needed by later modules.
