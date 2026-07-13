# LLM From Scratch Curriculum Design

Date: 2026-06-20
Workspace: `/Users/tovarishchrafa/llm-from-scratch`

## Goal

Build a hybrid notebook curriculum that develops a deep, math-forward understanding of language models by constructing a toy but fully working decoder-only LLM from scratch, then mapping the handmade components to modern PyTorch and Hugging Face abstractions.

The deeper purpose is not only to understand GPT-style transformers. The curriculum should give Rafael enough architectural fluency to orient toward questions about what comes after or around LLMs: sparse and subquadratic attention, post-transformer techniques, JEPA-style representation learning, world models, memory, grounding, multimodality, and future architecture papers.

## Project Shape

Use a progressive from-scratch curriculum.

The project will have one main narrative build path plus focused supporting notebooks. Early notebooks use mostly self-contained cells. As concepts stabilize, reusable pieces are extracted into `src/llm_from_scratch/` and tested.

Planned structure:

```text
llm-from-scratch/
  pyproject.toml
  README.md
  notebooks/
    00_project_orientation.ipynb
    01_tensors_autograd_and_probability.ipynb
    02_text_tokenization_char_to_subword.ipynb
    03_embeddings_and_language_modeling.ipynb
    04_attention_from_raw_tensors.ipynb
    05_transformer_block_from_scratch.ipynb
    06_training_loop_loss_and_optimization.ipynb
    07_generation_sampling_and_evaluation.ipynb
    08_finetuning_toy_instruction_task.ipynb
    09_hugging_face_translation_layer.ipynb
    10_quantization_deep_dive.ipynb
    11_modern_llm_orientation.ipynb
    12_beyond_transformers_and_world_models.ipynb
  src/
    llm_from_scratch/
      data.py
      tokenization.py
      model.py
      attention.py
      train.py
      generate.py
      evaluate.py
      configs.py
  tests/
  data/
    tiny_corpus.txt
  exercises/
    solutions/
  docs/
    math/
    notes/
```

This design is intentionally notebook-centered, but it should still mature into a testable Python project.

## Learning Flow

The curriculum moves in three main passes.

### Pass 1: First Principles

Build the smallest useful pieces directly with tensors:

- token IDs
- character-level tokenization
- subword tokenization
- learned embeddings
- logits
- softmax and log-softmax
- cross-entropy / negative log-likelihood
- causal masking
- scaled dot-product attention
- multi-head attention
- sampling
- a minimal training loop

The first-principles pass must emphasize tensor shapes, derivations, numerical stability, and clear failure modes.

### Pass 2: Assembled Architecture

Refactor raw tensor pieces into reusable `torch.nn.Module` components:

- token embeddings
- positional embeddings
- multi-head causal self-attention
- feed-forward network
- residual paths
- layer normalization
- transformer block
- decoder-only language model
- optimizer setup
- checkpointing
- validation and perplexity
- generation utilities

The assembled model should train as a small GPT-style decoder-only language model.

### Pass 3: Modern Abstraction Mapping

Map each handmade component to modern library abstractions:

- PyTorch `Dataset`, `DataLoader`, `nn.Module`, optimizers, and device handling
- Hugging Face `datasets.load_dataset`
- Hugging Face `tokenizers` for BPE-style tokenizer training
- Hugging Face `transformers` causal language modeling APIs
- model configs
- generation APIs
- checkpoint loading and saving
- tokenizer/model packaging
- training abstractions, with `Trainer` compared after the handmade training loop

This pass should explicitly answer: what did we build manually, what do libraries now hide, and what must still be understood to use those libraries well?

## Core Scope

The core implemented curriculum includes:

- character-level language model
- subword-token language model
- decoder-only GPT-style architecture
- pretraining loop
- validation loss and perplexity
- generation with temperature, top-k, and top-p sampling
- small supervised fine-tuning task
- exercises and solutions
- quick, study, and stretch runtime profiles
- CPU, MPS, and CUDA device detection
- package-backed tests for extracted code
- notebook smoke checks for key notebooks

The default runtime profile is study-sized: core runs should finish in roughly 10-30 minutes on a suitable local machine. Quick mode should finish in roughly 1-5 minutes. Stretch mode can run longer and is optional.

## Quantization Scope

Quantization deserves a dedicated deep-dive notebook, not a short orientation paragraph.

The quantization notebook should cover:

- why quantization matters for LLM inference and deployment
- floating point formats versus integer and lower-precision formats
- symmetric and asymmetric quantization
- scale and zero-point
- per-tensor, per-channel, and groupwise quantization
- weight-only quantization
- activation quantization
- dynamic versus static quantization
- calibration and representative data
- dequantization and fake quantization
- memory, bandwidth, latency, and quality tradeoffs
- why quantization can help memory more reliably than end-to-end latency
- common LLM quantization methods and formats, including bitsandbytes-style 8-bit and 4-bit loading, GPTQ, AWQ, and torchao workflows
- practical constraints on Apple Silicon, CUDA, and CPU paths
- how quantization interacts with attention, MLP weights, KV cache memory, and generation throughput

The notebook should include at least one from-scratch toy quantization implementation for a linear layer or embedding matrix, then compare that implementation to current PyTorch and Hugging Face quantization workflows.

## Beyond LLMs Orientation

The final orientation track exists because the long-term goal is to reason about what comes after or around current LLMs.

This track should not try to implement every frontier architecture. It should build a conceptual map that makes future papers easier to read.

Topics:

- why vanilla attention has quadratic time and memory in sequence length
- sparse, local, block, sliding-window, global-token, grouped-query, and multi-query attention
- subquadratic attention ambitions and their tradeoffs
- recurrence and state-space approaches as alternatives to full attention
- post-transformer architecture families and what bottleneck each targets
- predictive representation learning
- JEPA-style objectives
- world-model motivation
- predictive versus generative objectives
- multimodality and grounded representations
- memory, planning, environment interaction, and agency
- how to read future architecture papers: objective, inductive bias, compute bottleneck, data assumptions, scaling behavior, evaluation validity, and deployment constraints

Earlier notebooks should include short "why this matters later" notes. The attention notebook must make the `O(T^2)` compute and memory issue visible, since that is the bridge into sparse and subquadratic attention.

## Math Depth

The notebooks should be math-forward. Rafael has enough mathematical maturity for full derivations, but advanced probability assumptions should still be made explicit.

Each core concept should follow this pattern:

1. Problem statement: what the model needs to compute and why.
2. Derivation: notation, assumptions, tensor shapes, and probability model.
3. Raw tensor implementation: explicit PyTorch tensor code before abstractions.
4. Numerical checks: shape checks, probability normalization, masking checks, and selected gradient checks.
5. Module version: refactor into `nn.Module` once the mechanism is clear.
6. Exercise checkpoint: short problems with separate solutions.

Required math topics:

- categorical language modeling
- factorization `p(x_1, ..., x_T) = product_t p(x_t | x_<t)`
- logits, softmax, log-softmax, and numerical stability
- cross-entropy as negative log-likelihood
- gradient of softmax plus cross-entropy
- embeddings as learned lookup matrices
- sinusoidal versus learned positional embeddings
- scaled dot-product attention
- causal masking and label leakage
- multi-head attention as parallel learned projections
- residual streams
- layer normalization
- GELU and activation functions
- Adam and AdamW update rules
- temperature, top-k, and top-p sampling
- perplexity and validation loss
- quantization ranges, scales, zero-points, and error

## Exercises

Exercises should be integrated into the notebooks and mirrored by solution files.

Exercise types:

- shape reasoning: predict tensor shapes before running cells
- derivation checks: fill in missing steps in loss, attention, or optimizer derivations
- implementation gaps: write a mask, projection, sampling filter, or loss calculation
- debugging tasks: diagnose broken training from label shift, mask leakage, device mismatch, exploding logits, or unstable softmax
- extension tasks: add dropout, switch activation functions, compare optimizers, change tokenizer, tune generation settings, or compare quantized versus unquantized weights

Solutions should be explicit enough that Rafael can self-check without turning the exercise into another opaque abstraction.

## Tooling

Use `uv` and `pyproject.toml` for project and dependency management.

Core dependencies:

- Python
- PyTorch
- Jupyter
- NumPy
- Matplotlib
- pytest
- notebook execution tooling

Hugging Face and related dependencies:

- `datasets`
- `tokenizers`
- `transformers`
- quantization libraries only where needed for the quantization and modern abstraction notebooks

Device detection should prefer:

```text
CUDA -> MPS -> CPU
```

Code must gracefully fall back to CPU if CUDA or MPS is unavailable.

## Verification

Verification is part of the curriculum.

Quality gates:

- unit tests for extracted `src/llm_from_scratch/` modules
- notebook smoke execution for selected notebooks
- deterministic seeds
- tiny-batch overfit test
- loss sanity checks in quick and study runs
- shape assertions
- mask assertions
- CPU tests
- optional MPS/CUDA checks when available
- expected outputs for key exercises

The implementation plan should include tests before code wherever practical.

## Documentation Tone

The notebooks should teach like a careful mathematical engineering text:

- explicit notation
- concrete tensor shapes
- no unexplained magic constants
- diagrams only where they clarify the computation
- clear statements about what is old, what is still foundational, and what modern libraries abstract away
- short "why this matters later" notes that connect current transformer mechanics to post-transformer and world-model questions

## Sources Consulted For Current Library Direction

- PyTorch MPS backend documentation: https://docs.pytorch.org/docs/stable/notes/mps.html
- PyTorch torchao documentation: https://pytorch.org/ao/index.html
- Hugging Face Transformers quantization documentation: https://huggingface.co/docs/transformers/en/main_classes/quantization
- Hugging Face Datasets loading documentation: https://huggingface.co/docs/datasets/en/loading
- Hugging Face Tokenizers quicktour: https://huggingface.co/docs/tokenizers/en/quicktour
- Hugging Face Transformers generation documentation: https://huggingface.co/docs/transformers/en/main_classes/text_generation
- Hugging Face Transformers causal language modeling documentation: https://huggingface.co/docs/transformers/en/tasks/language_modeling
- uv project documentation: https://docs.astral.sh/uv/guides/projects/

## Out Of Scope For The First Implementation Pass

The first implementation pass should not attempt to:

- train a large model
- implement distributed training
- build production serving infrastructure
- implement RLHF, DPO, RAG, or JEPA from scratch
- benchmark frontier sparse attention implementations
- package a model for public release

Those topics can be revisited after the first curriculum is usable and Rafael has studied through it.

## Acceptance Criteria

The design is successful when the implementation plan can produce:

- a runnable `uv`-managed project
- a coherent sequence of notebooks
- a toy decoder-only LLM trained from scratch
- character-level and subword-token paths
- validated training and generation loops
- exercises with solutions
- a dedicated quantization deep dive
- a Hugging Face translation chapter
- a final beyond-transformers orientation chapter
- tests for the extracted package code
- smoke checks for important notebooks

The project should leave Rafael with enough transformer fluency to understand the current architecture stack and enough conceptual scaffolding to begin studying what may come after it.
