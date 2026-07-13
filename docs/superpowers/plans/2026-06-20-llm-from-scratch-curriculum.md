# LLM From Scratch Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `uv`-managed notebook curriculum and tested Python package that teaches LLM architecture by constructing a toy decoder-only GPT from scratch, then mapping it to modern Hugging Face/PyTorch workflows and beyond-transformer orientation.

**Architecture:** The project starts notebook-first, then progressively extracts stable code into `src/llm_from_scratch/`. Package modules hold reusable data, tokenizer, attention, model, training, evaluation, generation, and quantization primitives; notebooks teach the derivations, raw tensor versions, and library mappings.

**Tech Stack:** Python 3.11+, uv, PyTorch, Jupyter, NumPy, Matplotlib, pytest, nbclient, Hugging Face datasets/tokenizers/transformers, optional torchao for quantization orientation.

## Global Constraints

- Workspace root: `/Users/tovarishchrafa/llm-from-scratch`.
- Use `uv` and `pyproject.toml` for project and dependency management.
- Device preference order must be `CUDA -> MPS -> CPU`.
- Code must gracefully fall back to CPU if CUDA or MPS is unavailable.
- Default runtime profile is study-sized: roughly 10-30 minutes on a suitable local machine.
- Quick mode must target roughly 1-5 minute runs.
- Stretch mode is optional and can run longer.
- The curriculum must include character-level and subword-token language-model paths.
- The model architecture must be decoder-only GPT-style.
- The project must include validation loss, perplexity, generation, and a small supervised fine-tuning task.
- Quantization must be a dedicated deep-dive notebook, not a short orientation paragraph.
- Beyond-transformers and world-model material is orientation-only in the first implementation pass.
- Do not implement distributed training, production serving, RLHF, DPO, RAG, or JEPA from scratch in this first implementation pass.
- Include exercises with solution files.
- Include unit tests for extracted package code.
- Include notebook smoke checks for selected notebooks.
- Committed notebooks must have cleared outputs; smoke execution must not write outputs back into source notebooks.
- `scripts/make_notebooks.py` is a one-time scaffold generator. After Task 7 creates notebooks, the `.ipynb` files are the source of truth and the generator must not be re-run during content authoring.
- Notebook edits must be made with `nbformat`, Jupyter, or another notebook-aware tool; do not hand-edit notebook JSON as raw text.
- Functional primitives in `functional.py` are the pedagogical reference implementations. Module code may use optimized PyTorch equivalents after tests and notebooks explicitly connect the two.

---

## File Structure And Responsibilities

Create and maintain these files:

- `.gitignore`: excludes `.venv`, caches, notebook checkpoints, datasets/models/checkpoints generated during study, and `.DS_Store`.
- `pyproject.toml`: project metadata, runtime dependencies, dev dependencies, pytest config, and ruff config.
- `README.md`: project overview, setup, notebook order, runtime profiles, and verification commands.
- `data/tiny_corpus.txt`: tiny public-domain/local training text used for frictionless early runs.
- `src/llm_from_scratch/__init__.py`: package exports and version.
- `src/llm_from_scratch/configs.py`: dataclasses for model/training configs, runtime profiles, seeding, and device selection.
- `src/llm_from_scratch/tokenization.py`: `CharTokenizer`, BPE training/loading helpers, and tokenizer protocols.
- `src/llm_from_scratch/data.py`: text loading, token splitting, train/validation split, batching, and small supervised fine-tuning examples.
- `src/llm_from_scratch/functional.py`: raw tensor math helpers for softmax, cross-entropy, causal masks, and attention.
- `src/llm_from_scratch/attention.py`: attention functions and `CausalSelfAttention`.
- `src/llm_from_scratch/model.py`: `GPTBlock`, `TinyGPT`, parameter counting, and forward loss wiring.
- `src/llm_from_scratch/train.py`: training loop, checkpoint helpers, tiny overfit helper, and metrics collection.
- `src/llm_from_scratch/generate.py`: temperature, top-k, top-p, and autoregressive generation.
- `src/llm_from_scratch/evaluate.py`: validation loss, perplexity, and simple generation checks.
- `src/llm_from_scratch/quantization.py`: toy quantization helpers for scale/zero-point, quantize/dequantize, and quantized linear demonstration.
- `scripts/make_notebooks.py`: one-time notebook scaffold helper using `nbformat`; not the source of truth after notebooks are generated.
- `scripts/smoke_notebooks.py`: selected notebook execution via `nbclient`.
- `scripts/strip_notebook_outputs.py`: clears notebook outputs and execution counts before commit.
- `notebooks/*.ipynb`: tutorial notebooks listed in the design spec.
- `exercises/*.md`: exercise prompts.
- `exercises/solutions/*.md`: worked solutions.
- `docs/math/*.md`: longer derivations that are useful outside notebooks.
- `docs/notes/*.md`: reading maps and beyond-transformer notes.
- `tests/*.py`: unit and smoke tests.

---

### Task 1: Project Scaffold And Environment

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/llm_from_scratch/__init__.py`
- Create: `tests/test_imports.py`

**Interfaces:**
- Produces: importable package `llm_from_scratch`
- Produces: `uv sync --all-groups` development environment
- Produces: `pytest` command target for later tasks

- [ ] **Step 1: Write scaffold files**

Create `.gitignore`:

```gitignore
.DS_Store
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
notebooks/.executed/
*.nbconvert.ipynb
htmlcov/
.coverage
dist/
build/
*.egg-info/
data/cache/
data/downloads/
checkpoints/
models/
outputs/
wandb/
```

Create `pyproject.toml`:

```toml
[project]
name = "llm-from-scratch"
version = "0.1.0"
description = "A math-forward notebook curriculum for building a toy LLM from scratch."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.6",
  "numpy>=2.0",
  "matplotlib>=3.9",
  "jupyterlab>=4.3",
  "ipykernel>=6.29",
  "datasets>=3.0",
  "tokenizers>=0.20",
  "transformers>=4.48",
  "accelerate>=1.0",
  "safetensors>=0.5",
  "tqdm>=4.67",
  "rich>=13.9",
]

[dependency-groups]
dev = [
  "pytest>=8.3",
  "nbclient>=0.10",
  "nbformat>=5.10",
  "ruff>=0.9",
]
quantization = [
  "torchao>=0.9",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

Create `README.md`:

````markdown
# LLM From Scratch

This project is a notebook-first curriculum for building a toy decoder-only language model from scratch, then mapping the handmade pieces to modern PyTorch and Hugging Face workflows.

## Setup

```bash
uv sync --all-groups
uv run python -m ipykernel install --user --name llm-from-scratch --display-name "LLM From Scratch"
```

## Notebook Order

1. `00_project_orientation.ipynb`
2. `01_tensors_autograd_and_probability.ipynb`
3. `02_text_tokenization_char_to_subword.ipynb`
4. `03_embeddings_and_language_modeling.ipynb`
5. `04_attention_from_raw_tensors.ipynb`
6. `05_transformer_block_from_scratch.ipynb`
7. `06_training_loop_loss_and_optimization.ipynb`
8. `07_generation_sampling_and_evaluation.ipynb`
9. `08_finetuning_toy_instruction_task.ipynb`
10. `09_hugging_face_translation_layer.ipynb`
11. `10_quantization_deep_dive.ipynb`
12. `11_modern_llm_orientation.ipynb`
13. `12_beyond_transformers_and_world_models.ipynb`

## Runtime Profiles

- `quick`: shortest runs for smoke tests and fast iteration.
- `study`: default runs for meaningful loss curves.
- `stretch`: optional longer runs for scaling intuition.

## Verification

```bash
uv run pytest
uv run python scripts/smoke_notebooks.py --quick
```
````

Create `src/llm_from_scratch/__init__.py`:

```python
"""LLM-from-scratch curriculum package."""

__version__ = "0.1.0"
```

Create `tests/test_imports.py`:

```python
def test_package_imports() -> None:
    import llm_from_scratch

    assert llm_from_scratch.__version__ == "0.1.0"
```

- [ ] **Step 2: Sync dependencies**

Run:

```bash
uv sync --all-groups
```

Expected: exit code `0`.

- [ ] **Step 3: Run the initial test**

Run:

```bash
uv run pytest tests/test_imports.py -q
```

Expected: one passing test.

- [ ] **Step 4: Commit**

```bash
git add .gitignore pyproject.toml README.md src/llm_from_scratch/__init__.py tests/test_imports.py
git commit -m "chore: scaffold llm curriculum project"
```

---

### Task 2: Runtime Configs, Device Selection, And Reproducibility

**Files:**
- Create: `src/llm_from_scratch/configs.py`
- Create: `tests/test_configs.py`

**Interfaces:**
- Produces: `ModelConfig`
- Produces: `TrainConfig`
- Produces: `RuntimeProfile`
- Produces: `get_device() -> torch.device`
- Produces: `set_seed(seed: int) -> None`
- Produces: `profile_config(name: str) -> tuple[ModelConfig, TrainConfig]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_configs.py`:

```python
import random

import numpy as np
import torch

from llm_from_scratch.configs import (
    ModelConfig,
    RuntimeProfile,
    TrainConfig,
    get_device,
    profile_config,
    set_seed,
)


def test_default_model_config_is_tiny_enough_for_tests() -> None:
    cfg = ModelConfig(vocab_size=32)
    assert cfg.vocab_size == 32
    assert cfg.block_size <= 128
    assert cfg.n_embd % cfg.n_head == 0


def test_runtime_profiles_have_ordered_training_sizes() -> None:
    quick_model, quick_train = profile_config("quick")
    study_model, study_train = profile_config("study")
    stretch_model, stretch_train = profile_config("stretch")

    assert quick_model.n_layer <= study_model.n_layer <= stretch_model.n_layer
    assert quick_train.max_steps < study_train.max_steps < stretch_train.max_steps


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_set_seed_reproducible_for_python_numpy_and_torch() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_runtime_profile_literal_values() -> None:
    assert RuntimeProfile.QUICK.value == "quick"
    assert RuntimeProfile.STUDY.value == "study"
    assert RuntimeProfile.STRETCH.value == "stretch"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_configs.py -q
```

Expected: fail because `llm_from_scratch.configs` does not exist.

- [ ] **Step 3: Implement configs**

Create `src/llm_from_scratch/configs.py`:

```python
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class RuntimeProfile(str, Enum):
    QUICK = "quick"
    STUDY = "study"
    STRETCH = "stretch"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 500
    eval_interval: int = 50
    eval_batches: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def profile_config(name: str) -> tuple[ModelConfig, TrainConfig]:
    profile = RuntimeProfile(name)
    if profile is RuntimeProfile.QUICK:
        return (
            ModelConfig(vocab_size=256, block_size=32, n_embd=64, n_head=4, n_layer=1),
            TrainConfig(batch_size=8, max_steps=100, eval_interval=25, eval_batches=2),
        )
    if profile is RuntimeProfile.STUDY:
        return (
            ModelConfig(vocab_size=256, block_size=64, n_embd=128, n_head=4, n_layer=2),
            TrainConfig(batch_size=16, max_steps=800, eval_interval=100, eval_batches=8),
        )
    return (
        ModelConfig(vocab_size=256, block_size=128, n_embd=256, n_head=8, n_layer=4),
        TrainConfig(batch_size=32, max_steps=3000, eval_interval=250, eval_batches=16),
    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_configs.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/llm_from_scratch/configs.py tests/test_configs.py
git commit -m "feat: add runtime configs and device selection"
```

---

### Task 3: Tokenization And Data Batching

**Files:**
- Create: `data/tiny_corpus.txt`
- Create: `src/llm_from_scratch/tokenization.py`
- Create: `src/llm_from_scratch/data.py`
- Create: `tests/test_tokenization.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: `set_seed(seed: int) -> None`
- Produces: `CharTokenizer.from_text(text: str) -> CharTokenizer`
- Produces: `CharTokenizer.encode(text: str) -> list[int]`
- Produces: `CharTokenizer.decode(ids: list[int]) -> str`
- Produces: `train_bpe_tokenizer(texts: Iterable[str], vocab_size: int = 256) -> tokenizers.Tokenizer`
- Produces: `split_tokens(tokens: torch.Tensor, train_fraction: float) -> tuple[torch.Tensor, torch.Tensor]`
- Produces: `get_batch(tokens: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]`
- Produces: `toy_instruction_examples() -> list[tuple[str, str]]`

- [ ] **Step 1: Write tests**

Create `tests/test_tokenization.py`:

```python
from llm_from_scratch.tokenization import CharTokenizer


def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("banana")
    ids = tokenizer.encode("banana")
    assert tokenizer.decode(ids) == "banana"
    assert tokenizer.vocab_size == 3


def test_char_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    try:
        tokenizer.encode("abd")
    except KeyError as exc:
        assert "Unknown character" in str(exc)
    else:
        raise AssertionError("encoding an unknown character should fail")


def test_train_bpe_tokenizer_encodes_and_decodes_text() -> None:
    from llm_from_scratch.tokenization import train_bpe_tokenizer

    tokenizer = train_bpe_tokenizer(["hello world", "hello token"], vocab_size=64)
    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded.ids)
    assert encoded.ids
    assert "hello" in decoded
```

Create `tests/test_data.py`:

```python
import torch

from llm_from_scratch.data import get_batch, split_tokens, toy_instruction_examples


def test_split_tokens_preserves_total_length() -> None:
    tokens = torch.arange(100)
    train, val = split_tokens(tokens, train_fraction=0.8)
    assert len(train) == 80
    assert len(val) == 20


def test_get_batch_returns_shifted_inputs_and_targets() -> None:
    tokens = torch.tensor([3, 10, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15])
    x, y = get_batch(tokens, block_size=4, batch_size=4, device=torch.device("cpu"))
    assert x.shape == (4, 4)
    assert y.shape == (4, 4)
    for row_x, row_y in zip(x, y, strict=True):
        matches = [
            start
            for start in range(len(tokens) - 4)
            if torch.equal(row_x.cpu(), tokens[start : start + 4])
        ]
        assert len(matches) == 1
        start = matches[0]
        assert torch.equal(row_y.cpu(), tokens[start + 1 : start + 5])


def test_toy_instruction_examples_have_prompt_and_response() -> None:
    examples = toy_instruction_examples()
    assert examples
    prompt, response = examples[0]
    assert isinstance(prompt, str)
    assert isinstance(response, str)
    assert prompt
    assert response
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_tokenization.py tests/test_data.py -q
```

Expected: fail because modules do not exist.

- [ ] **Step 3: Add tiny corpus**

Create `data/tiny_corpus.txt`:

```text
The model learns by predicting the next token.
Attention lets each token read earlier tokens.
A tiny corpus is enough to test shapes, loss, and generation.
We study the small system so the large system becomes less mysterious.

Language modeling turns a sequence into many supervised examples.
At one position, the context is short.
At a later position, the context is longer.
The same parameters must serve both cases.

The token embedding table gives each symbol a learned vector.
The position embedding table tells the model where the symbol appears.
Self-attention compares positions with queries and keys.
Values carry the information that gets mixed into the residual stream.

A causal mask blocks the future.
Without the mask, training loss can look better for the wrong reason.
The model would read the answer instead of learning to predict it.

Optimization changes the weights a little at a time.
The loss is a negative log probability.
Lower loss means the model assigned more probability to the observed next token.
Generation reverses the perspective and samples from the model's predicted distribution.
```

- [ ] **Step 4: Implement tokenization**

Create `src/llm_from_scratch/tokenization.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


@dataclass(frozen=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        itos = {idx: ch for ch, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            if ch not in self.stoi:
                raise KeyError(f"Unknown character: {ch!r}")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int = 256) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer
```

- [ ] **Step 5: Implement data helpers**

Create `src/llm_from_scratch/data.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_tokens(
    tokens: torch.Tensor,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_idx = int(len(tokens) * train_fraction)
    return tokens[:split_idx], tokens[split_idx:]


def get_batch(
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= block_size:
        raise ValueError("tokens length must exceed block_size")
    ix = torch.randint(0, len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def toy_instruction_examples() -> list[tuple[str, str]]:
    return [
        ("Define attention in one sentence.", "Attention mixes information across token positions."),
        ("What does a language model predict?", "It predicts a distribution over the next token."),
        ("Why use a causal mask?", "The mask prevents tokens from reading future tokens."),
    ]
```

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_tokenization.py tests/test_data.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add data/tiny_corpus.txt src/llm_from_scratch/tokenization.py src/llm_from_scratch/data.py tests/test_tokenization.py tests/test_data.py
git commit -m "feat: add tokenization and batching helpers"
```

---

### Task 4: Functional Math Primitives

**Files:**
- Create: `src/llm_from_scratch/functional.py`
- Create: `tests/test_functional.py`
- Create: `docs/math/softmax_cross_entropy.md`
- Create: `docs/math/attention.md`

**Interfaces:**
- Produces: `stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor`
- Produces: `cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`
- Produces: `causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor`
- Produces: `scaled_dot_product_attention(q, k, v, mask=None) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write tests**

Create `tests/test_functional.py`:

```python
import math

import torch
import torch.nn.functional as F

from llm_from_scratch.functional import (
    causal_mask,
    cross_entropy_from_logits,
    scaled_dot_product_attention,
    stable_softmax,
)


def test_stable_softmax_rows_sum_to_one() -> None:
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
    probs = stable_softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(probs).all()


def test_cross_entropy_matches_pytorch() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    targets = torch.tensor([2, 0])
    expected = F.cross_entropy(logits, targets)
    actual = cross_entropy_from_logits(logits, targets)
    assert torch.allclose(actual, expected)


def test_causal_mask_is_lower_triangular() -> None:
    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_scaled_dot_product_attention_shapes_and_masking() -> None:
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 5)
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(3))
    assert out.shape == (2, 3, 5)
    assert weights.shape == (2, 3, 3)
    assert torch.allclose(weights[0, 0, 1:], torch.zeros(2), atol=1e-7)
    assert math.isclose(weights[0, 0].sum().item(), 1.0, rel_tol=1e-6)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_functional.py -q
```

Expected: fail because `functional.py` does not exist.

- [ ] **Step 3: Implement functional primitives**

Create `src/llm_from_scratch/functional.py`:

```python
from __future__ import annotations

import math

import torch


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")
    if logits.shape[:-1] != targets.shape:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    losses = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return losses.mean()


def causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = stable_softmax(scores, dim=-1)
    return weights @ v, weights
```

- [ ] **Step 4: Add derivation docs**

Create `docs/math/softmax_cross_entropy.md`:

````markdown
# Softmax And Cross-Entropy

For logits `z in R^V`, softmax defines

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Subtracting `max(z)` before exponentiation leaves the probabilities unchanged and prevents overflow.

For a target class `y`, the negative log-likelihood is

```text
L = -log p_y = -z_y + log(sum_j exp(z_j)).
```

The gradient with respect to each logit is

```text
dL/dz_i = p_i - 1[i = y].
```

This is why the output layer receives a dense probability-error signal even though the target is a single class index.
````

Create `docs/math/attention.md`:

````markdown
# Scaled Dot-Product Attention

Given queries `Q`, keys `K`, and values `V`, attention computes

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
```

For causal language modeling, token position `t` may read positions `<= t` and must not read positions `> t`. The causal mask replaces forbidden logits with `-inf` before softmax, forcing their probabilities to zero.

The attention score matrix has shape `T x T`, so vanilla self-attention has quadratic sequence-length cost in both score memory and score computation. This fact is the bridge to sparse and subquadratic attention methods later in the curriculum.
````

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_functional.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/llm_from_scratch/functional.py tests/test_functional.py docs/math/softmax_cross_entropy.md docs/math/attention.md
git commit -m "feat: add functional math primitives"
```

---

### Task 5: Attention And GPT Model Modules

**Files:**
- Create: `src/llm_from_scratch/attention.py`
- Create: `src/llm_from_scratch/model.py`
- Create: `tests/test_attention.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `ModelConfig`
- Consumes: `causal_mask(size, device)`
- Produces: `CausalSelfAttention(config: ModelConfig)`
- Produces: `GPTBlock(config: ModelConfig)`
- Produces: `TinyGPT(config: ModelConfig)`
- Produces: `TinyGPT.forward(idx, targets=None) -> tuple[torch.Tensor, torch.Tensor | None]`
- Produces: `count_parameters(model: torch.nn.Module) -> int`

- [ ] **Step 1: Write tests**

Create `tests/test_attention.py`:

```python
import torch

from llm_from_scratch.attention import CausalSelfAttention
from llm_from_scratch.configs import ModelConfig


def test_causal_self_attention_preserves_shape() -> None:
    cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=12, n_head=3, n_layer=1)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 5, 12)
    out = attn(x)
    assert out.shape == x.shape
```

Create `tests/test_model.py`:

```python
import torch

from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.model import TinyGPT, count_parameters


def test_tiny_gpt_forward_without_targets_returns_logits() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss = model(idx)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert loss is None


def test_tiny_gpt_forward_with_targets_returns_loss() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_tiny_gpt_rejects_too_long_context() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=4, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    try:
        model(idx)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("model should reject sequences longer than block_size")


def test_count_parameters_positive() -> None:
    cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(cfg)
    assert count_parameters(model) > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_attention.py tests/test_model.py -q
```

Expected: fail because modules do not exist.

- [ ] **Step 3: Implement attention module**

Create `src/llm_from_scratch/attention.py`:

```python
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from llm_from_scratch.configs import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        return self.proj(out)
```

- [ ] **Step 4: Implement model module**

Create `src/llm_from_scratch/model.py`:

```python
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from llm_from_scratch.attention import CausalSelfAttention
from llm_from_scratch.configs import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("sequence length exceeds model block_size")
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_attention.py tests/test_model.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/llm_from_scratch/attention.py src/llm_from_scratch/model.py tests/test_attention.py tests/test_model.py
git commit -m "feat: add tiny gpt model modules"
```

---

### Task 6: Generation, Evaluation, And Training Loop

**Files:**
- Create: `src/llm_from_scratch/generate.py`
- Create: `src/llm_from_scratch/evaluate.py`
- Create: `src/llm_from_scratch/train.py`
- Create: `tests/test_generate.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `TinyGPT`
- Consumes: `get_batch`
- Produces: `top_k_filter(logits, k)`
- Produces: `top_p_filter(logits, p)`
- Produces: `sample_next_token(logits, temperature=1.0, top_k=None, top_p=None) -> torch.Tensor`
- Produces: `generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None) -> torch.Tensor`
- Produces: `estimate_loss(model, train_tokens, val_tokens, batch_size, eval_batches, device) -> dict[str, float]`
- Produces: `perplexity(loss: float) -> float`
- Produces: `train_steps(model, train_tokens, val_tokens, train_config, device) -> list[dict[str, float]]`
- Produces: `overfit_tiny_batch(model, x, y, steps=80, lr=1e-2) -> tuple[float, float]`
- Produces: `save_checkpoint(path, model, metadata=None) -> None`
- Produces: `load_checkpoint(path, map_location="cpu") -> dict[str, object]`

- [ ] **Step 1: Write tests**

Create `tests/test_generate.py`:

```python
import torch

from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.generate import generate, top_k_filter, top_p_filter
from llm_from_scratch.model import TinyGPT


def test_top_k_filter_keeps_only_k_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    filtered = top_k_filter(logits, k=2)
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert filtered[0, 2] == 3.0
    assert filtered[0, 3] == 4.0


def test_top_p_filter_keeps_high_probability_prefix() -> None:
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    filtered = top_p_filter(logits, p=0.8)
    assert torch.isfinite(filtered).any()
    assert filtered.shape == logits.shape


def test_generate_appends_tokens() -> None:
    cfg = ModelConfig(vocab_size=10, block_size=8, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    idx = torch.zeros((1, 3), dtype=torch.long)
    out = generate(model, idx, max_new_tokens=2)
    assert out.shape == (1, 5)
```

Create `tests/test_train.py`:

```python
import torch

from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.evaluate import perplexity
from llm_from_scratch.model import TinyGPT
from llm_from_scratch.train import load_checkpoint, overfit_tiny_batch, save_checkpoint


def test_perplexity_is_exp_loss() -> None:
    assert abs(perplexity(0.0) - 1.0) < 1e-8


def test_overfit_tiny_batch_reduces_loss() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    first, last = overfit_tiny_batch(model, x, y, steps=40, lr=1e-2)
    assert last < first


def test_checkpoint_round_trip(tmp_path) -> None:
    cfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    model = TinyGPT(cfg)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, metadata={"step": 7})
    checkpoint = load_checkpoint(path)
    assert checkpoint["metadata"]["step"] == 7
    assert "model_state_dict" in checkpoint
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_generate.py tests/test_train.py -q
```

Expected: fail because modules do not exist.

- [ ] **Step 3: Implement generation**

Create `src/llm_from_scratch/generate.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    if p is None or p >= 1.0:
        return logits
    if p <= 0.0:
        raise ValueError("top_p must be greater than 0")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    model.eval()
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
```

- [ ] **Step 4: Implement evaluation and training**

Create `src/llm_from_scratch/evaluate.py`:

```python
from __future__ import annotations

import math

import torch

from llm_from_scratch.data import get_batch


def perplexity(loss: float) -> float:
    return math.exp(loss)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, tokens in {"train": train_tokens, "val": val_tokens}.items():
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(tokens, model.config.block_size, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
        out[split] = float(sum(losses) / len(losses))
    model.train()
    return out
```

Create `src/llm_from_scratch/train.py`:

```python
from __future__ import annotations

from pathlib import Path

import torch

from llm_from_scratch.configs import TrainConfig
from llm_from_scratch.data import get_batch
from llm_from_scratch.evaluate import estimate_loss


def train_steps(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    train_config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    history: list[dict[str, float]] = []
    for step in range(train_config.max_steps):
        if step % train_config.eval_interval == 0:
            metrics = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                train_config.batch_size,
                train_config.eval_batches,
                device,
            )
            metrics["step"] = float(step)
            history.append(metrics)
        x, y = get_batch(train_tokens, model.config.block_size, train_config.batch_size, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
    return history


def overfit_tiny_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 80,
    lr: float = 1e-2,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    first_loss = None
    last_loss = None
    model.train()
    for _ in range(steps):
        _, loss = model(x, y)
        assert loss is not None
        if first_loss is None:
            first_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    assert first_loss is not None
    assert last_loss is not None
    return first_loss, last_loss


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    return torch.load(path, map_location=map_location, weights_only=False)
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_generate.py tests/test_train.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/llm_from_scratch/generate.py src/llm_from_scratch/evaluate.py src/llm_from_scratch/train.py tests/test_generate.py tests/test_train.py
git commit -m "feat: add training evaluation and generation"
```

---

### Task 7: Notebook Creation And Smoke Test Infrastructure

**Files:**
- Create: `scripts/make_notebooks.py`
- Create: `scripts/smoke_notebooks.py`
- Create: `scripts/strip_notebook_outputs.py`
- Create: `tests/test_notebook_scripts.py`

**Interfaces:**
- Produces: `scripts/make_notebooks.py` command that writes all curriculum notebooks.
- Produces: `scripts/smoke_notebooks.py --quick` command that executes selected notebooks with `nbclient`.
- Produces: `scripts/strip_notebook_outputs.py` command that clears outputs before notebooks are committed.

- [ ] **Step 1: Write tests**

Create `tests/test_notebook_scripts.py`:

```python
from pathlib import Path


def test_notebook_scripts_exist() -> None:
    assert Path("scripts/make_notebooks.py").exists()
    assert Path("scripts/smoke_notebooks.py").exists()
    assert Path("scripts/strip_notebook_outputs.py").exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_notebook_scripts.py -q
```

Expected: fail because scripts do not exist.

- [ ] **Step 3: Implement notebook writer**

Create `scripts/make_notebooks.py` with:

```python
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


SCAFFOLD_ONLY_WARNING = (
    "This script is a one-time scaffold generator. After curriculum content is added, "
    "notebooks/*.ipynb are the source of truth."
)


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "00_project_orientation.ipynb": [
        ("markdown", "# Project Orientation\n\nThis notebook explains the curriculum path, runtime profiles, and verification commands."),
        ("code", "from llm_from_scratch.configs import get_device, profile_config\nget_device(), profile_config('quick')"),
    ],
    "01_tensors_autograd_and_probability.ipynb": [
        ("markdown", "# Tensors, Autograd, And Probability\n\nWe connect tensor operations to categorical next-token prediction."),
        ("code", "import torch\nx = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\ny = (x**2).sum()\ny.backward()\nx.grad"),
    ],
    "02_text_tokenization_char_to_subword.ipynb": [
        ("markdown", "# Text Tokenization: Character To Subword\n\nStart with character IDs, then compare the idea to BPE tokenization."),
        ("code", "from pathlib import Path\nfrom llm_from_scratch.tokenization import CharTokenizer\ntext = Path('../data/tiny_corpus.txt').read_text()\ntok = CharTokenizer.from_text(text)\nids = tok.encode(text[:40])\ntok.decode(ids), tok.vocab_size"),
    ],
    "03_embeddings_and_language_modeling.ipynb": [
        ("markdown", "# Embeddings And Language Modeling\n\nDerive next-token modeling and inspect embedding lookup tensors."),
        ("code", "import torch\nembedding = torch.nn.Embedding(10, 4)\nids = torch.tensor([[1, 2, 3]])\nembedding(ids).shape"),
    ],
    "04_attention_from_raw_tensors.ipynb": [
        ("markdown", "# Attention From Raw Tensors\n\nBuild scaled dot-product attention and expose the O(T^2) score matrix."),
        ("code", "import torch\nfrom llm_from_scratch.functional import causal_mask, scaled_dot_product_attention\nq = torch.randn(1, 4, 8)\nk = torch.randn(1, 4, 8)\nv = torch.randn(1, 4, 8)\nout, weights = scaled_dot_product_attention(q, k, v, causal_mask(4))\nout.shape, weights.shape"),
    ],
    "05_transformer_block_from_scratch.ipynb": [
        ("markdown", "# Transformer Block From Scratch\n\nAssemble attention, residual paths, layer norm, and MLP layers."),
        ("code", "import torch\nfrom llm_from_scratch.configs import ModelConfig\nfrom llm_from_scratch.model import TinyGPT\ncfg = ModelConfig(vocab_size=32, block_size=8, n_embd=16, n_head=4, n_layer=1)\nmodel = TinyGPT(cfg)\nmodel(torch.randint(0, 32, (2, 8)))[0].shape"),
    ],
    "06_training_loop_loss_and_optimization.ipynb": [
        ("markdown", "# Training Loop, Loss, And Optimization\n\nTrain a tiny model and inspect loss movement."),
        ("code", "import torch\nfrom llm_from_scratch.configs import ModelConfig\nfrom llm_from_scratch.model import TinyGPT\nfrom llm_from_scratch.train import overfit_tiny_batch\ncfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)\nmodel = TinyGPT(cfg)\nx = torch.randint(0, 8, (4, 4))\ny = torch.randint(0, 8, (4, 4))\noverfit_tiny_batch(model, x, y, steps=5, lr=1e-2)"),
    ],
    "07_generation_sampling_and_evaluation.ipynb": [
        ("markdown", "# Generation, Sampling, And Evaluation\n\nCompare temperature, top-k, top-p, validation loss, and perplexity."),
        ("code", "from llm_from_scratch.evaluate import perplexity\nperplexity(1.0)"),
    ],
    "08_finetuning_toy_instruction_task.ipynb": [
        ("markdown", "# Fine-Tuning A Toy Instruction Task\n\nFormat prompt-response pairs and discuss supervised fine-tuning."),
        ("code", "from llm_from_scratch.data import toy_instruction_examples\ntoy_instruction_examples()[:2]"),
    ],
    "09_hugging_face_translation_layer.ipynb": [
        ("markdown", "# Hugging Face Translation Layer\n\nMap handmade data, tokenizer, model, training, and generation pieces to library abstractions."),
        ("code", "import transformers, datasets, tokenizers\ntransformers.__version__, datasets.__version__, tokenizers.__version__"),
    ],
    "10_quantization_deep_dive.ipynb": [
        ("markdown", "# Quantization Deep Dive\n\nDerive scale and zero-point, implement toy quantization, then compare to modern APIs."),
        ("code", "import torch\nfrom llm_from_scratch.quantization import quantize_tensor, dequantize_tensor\nx = torch.tensor([-1.0, 0.0, 1.0])\nq, params = quantize_tensor(x, num_bits=8)\ndequantize_tensor(q, params)"),
    ],
    "11_modern_llm_orientation.ipynb": [
        ("markdown", "# Modern LLM Orientation\n\nOrient from the toy GPT to instruction tuning, RAG, deployment, and systems concerns."),
        ("code", "print('orientation notebook')"),
    ],
    "12_beyond_transformers_and_world_models.ipynb": [
        ("markdown", "# Beyond Transformers And World Models\n\nUse the transformer build to reason about sparse attention, post-transformer techniques, JEPA, and world models."),
        ("code", "sequence_lengths = [128, 512, 2048]\n[(n, n*n) for n in sequence_lengths]"),
    ],
}


def make_notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "LLM From Scratch",
        "language": "python",
        "name": "llm-from-scratch",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


def main() -> None:
    print(SCAFFOLD_ONLY_WARNING)
    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nbf.write(make_notebook(cells), out_dir / name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement smoke runner**

Create `scripts/smoke_notebooks.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


QUICK_NOTEBOOKS = [
    "00_project_orientation.ipynb",
    "01_tensors_autograd_and_probability.ipynb",
    "02_text_tokenization_char_to_subword.ipynb",
    "03_embeddings_and_language_modeling.ipynb",
    "04_attention_from_raw_tensors.ipynb",
    "05_transformer_block_from_scratch.ipynb",
    "06_training_loop_loss_and_optimization.ipynb",
    "07_generation_sampling_and_evaluation.ipynb",
    "08_finetuning_toy_instruction_task.ipynb",
    "09_hugging_face_translation_layer.ipynb",
    "10_quantization_deep_dive.ipynb",
    "11_modern_llm_orientation.ipynb",
    "12_beyond_transformers_and_world_models.ipynb",
]


def execute_notebook(path: Path, timeout: int = 120) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3")
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run the quick notebook subset.")
    args = parser.parse_args()
    names = QUICK_NOTEBOOKS if args.quick else sorted(p.name for p in Path("notebooks").glob("*.ipynb"))
    for name in names:
        execute_notebook(Path("notebooks") / name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement output stripper**

Create `scripts/strip_notebook_outputs.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def strip_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop("widgets", None)
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["notebooks"])
    args = parser.parse_args()
    for raw_path in args.paths:
        path = Path(raw_path)
        notebooks = [path] if path.is_file() else sorted(path.glob("*.ipynb"))
        for notebook_path in notebooks:
            strip_notebook(notebook_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_notebook_scripts.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Generate notebooks**

Run:

```bash
uv run python scripts/make_notebooks.py
```

Expected: all 13 notebook files exist under `notebooks/`.

- [ ] **Step 8: Strip notebook outputs**

Run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks
```

Expected: command exits `0` and notebooks have no committed outputs.

- [ ] **Step 9: Commit**

```bash
git add scripts/make_notebooks.py scripts/smoke_notebooks.py scripts/strip_notebook_outputs.py tests/test_notebook_scripts.py notebooks
git commit -m "feat: add notebook scaffolding and smoke runner"
```

---

### Task 8: Quantization Helpers And Deep-Dive Notebook Upgrade

**Files:**
- Create: `src/llm_from_scratch/quantization.py`
- Create: `tests/test_quantization.py`
- Modify: `notebooks/10_quantization_deep_dive.ipynb`
- Create: `docs/math/quantization.md`
- Create: `exercises/quantization.md`
- Create: `exercises/solutions/quantization_solutions.md`

**Interfaces:**
- Produces: `QuantizationParams(scale: float, zero_point: int, qmin: int, qmax: int)`
- Produces: `quantize_tensor(x: torch.Tensor, num_bits: int = 8, symmetric: bool = False) -> tuple[torch.Tensor, QuantizationParams]`
- Produces: `dequantize_tensor(q: torch.Tensor, params: QuantizationParams) -> torch.Tensor`
- Produces: `fake_quantize_tensor(x: torch.Tensor, num_bits: int = 8, symmetric: bool = False) -> torch.Tensor`
- Produces: `quantize_per_channel(x: torch.Tensor, axis: int, num_bits: int = 8) -> tuple[torch.Tensor, list[QuantizationParams]]`
- Produces: `quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]`
- Produces: `estimate_kv_cache_bytes(n_layer: int, n_head: int, head_dim: int, seq_len: int, batch_size: int, bytes_per_value: int) -> int`
- Produces: quantization notebook with at least 8 markdown cells, 4 executable code cells, 2 numerical error comparisons, and explicit API notes for torchao, bitsandbytes-style loading, GPTQ, and AWQ

- [ ] **Step 1: Write tests**

Create `tests/test_quantization.py`:

```python
import torch

from llm_from_scratch.quantization import (
    dequantize_tensor,
    estimate_kv_cache_bytes,
    fake_quantize_tensor,
    quantization_error,
    quantize_per_channel,
    quantize_tensor,
)


def test_quantize_dequantize_preserves_shape() -> None:
    x = torch.linspace(-1, 1, steps=17)
    q, params = quantize_tensor(x, num_bits=8)
    x_hat = dequantize_tensor(q, params)
    assert q.shape == x.shape
    assert x_hat.shape == x.shape
    assert q.min() >= params.qmin
    assert q.max() <= params.qmax


def test_symmetric_quantization_has_zero_zero_point() -> None:
    x = torch.tensor([-2.0, 0.0, 1.0])
    _, params = quantize_tensor(x, num_bits=8, symmetric=True)
    assert params.zero_point == 0


def test_quantization_error_reports_mae_and_max_abs() -> None:
    x = torch.tensor([0.0, 1.0, 2.0])
    x_hat = torch.tensor([0.0, 1.1, 1.8])
    err = quantization_error(x, x_hat)
    assert set(err) == {"mae", "max_abs"}
    assert err["max_abs"] > 0


def test_fake_quantize_preserves_float_dtype_and_shape() -> None:
    x = torch.randn(3, 4)
    x_hat = fake_quantize_tensor(x, num_bits=4, symmetric=True)
    assert x_hat.shape == x.shape
    assert x_hat.dtype == torch.float32


def test_quantize_per_channel_returns_one_param_per_channel() -> None:
    x = torch.randn(3, 4)
    q, params = quantize_per_channel(x, axis=0, num_bits=8)
    assert q.shape == x.shape
    assert len(params) == 3


def test_estimate_kv_cache_bytes_counts_keys_and_values() -> None:
    bytes_used = estimate_kv_cache_bytes(
        n_layer=2,
        n_head=4,
        head_dim=8,
        seq_len=16,
        batch_size=1,
        bytes_per_value=2,
    )
    assert bytes_used == 2 * 2 * 4 * 8 * 16 * 1 * 2
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_quantization.py -q
```

Expected: fail because `quantization.py` does not exist.

- [ ] **Step 3: Implement quantization helpers**

Create `src/llm_from_scratch/quantization.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int


def quantize_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    symmetric: bool = False,
) -> tuple[torch.Tensor, QuantizationParams]:
    if not 2 <= num_bits <= 8:
        raise ValueError("num_bits must be between 2 and 8")
    if symmetric:
        qmax = 2 ** (num_bits - 1) - 1
        qmin = -qmax
        max_abs = x.abs().max().item()
        scale = max(max_abs / qmax, 1e-12)
        zero_point = 0
    else:
        qmin = 0
        qmax = 2**num_bits - 1
        x_min = x.min().item()
        x_max = x.max().item()
        scale = max((x_max - x_min) / (qmax - qmin), 1e-12)
        zero_point = round(qmin - x_min / scale)
        zero_point = int(max(qmin, min(qmax, zero_point)))
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return q.to(torch.int32), QuantizationParams(scale, zero_point, qmin, qmax)


def dequantize_tensor(q: torch.Tensor, params: QuantizationParams) -> torch.Tensor:
    return (q.to(torch.float32) - params.zero_point) * params.scale


def fake_quantize_tensor(
    x: torch.Tensor,
    num_bits: int = 8,
    symmetric: bool = False,
) -> torch.Tensor:
    q, params = quantize_tensor(x, num_bits=num_bits, symmetric=symmetric)
    return dequantize_tensor(q, params)


def quantize_per_channel(
    x: torch.Tensor,
    axis: int,
    num_bits: int = 8,
) -> tuple[torch.Tensor, list[QuantizationParams]]:
    axis = axis % x.ndim
    q = torch.empty_like(x, dtype=torch.int32)
    params: list[QuantizationParams] = []
    for channel in range(x.shape[axis]):
        index = [slice(None)] * x.ndim
        index[axis] = channel
        q_channel, channel_params = quantize_tensor(x[tuple(index)], num_bits=num_bits)
        q[tuple(index)] = q_channel
        params.append(channel_params)
    return q, params


def quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, float]:
    diff = (x - x_hat).abs()
    return {"mae": diff.mean().item(), "max_abs": diff.max().item()}


def estimate_kv_cache_bytes(
    n_layer: int,
    n_head: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    bytes_per_value: int,
) -> int:
    keys_and_values = 2
    return keys_and_values * n_layer * n_head * head_dim * seq_len * batch_size * bytes_per_value
```

- [ ] **Step 4: Add quantization derivation doc**

Create `docs/math/quantization.md`:

````markdown
# Quantization

Uniform affine quantization maps a real value `x` to an integer `q` using

```text
q = round(x / scale + zero_point).
```

Dequantization maps back to an approximate real value:

```text
x_hat = scale * (q - zero_point).
```

The scale controls bin width. The zero-point represents real zero exactly when possible. Symmetric quantization fixes `zero_point = 0`; asymmetric quantization uses the observed range to place zero inside an unsigned integer interval.

Per-channel quantization gives each output channel its own range. Groupwise quantization is a compromise: it stores several scales instead of one scale per tensor, but fewer scales than one per channel.

Fake quantization quantizes and immediately dequantizes during analysis or training. It lets a float model experience quantization error without requiring integer kernels.

For LLMs, weight-only quantization often saves memory with less calibration complexity than full activation quantization. Activation and KV-cache quantization can further reduce bandwidth or memory pressure, but the accuracy and hardware behavior depend strongly on kernels, calibration data, and device support.
````

- [ ] **Step 5: Upgrade notebook content**

Modify `notebooks/10_quantization_deep_dive.ipynb` so it contains these sections in order:

```text
1. Why quantization matters for LLM memory and inference.
2. Floating point versus integer ranges.
3. Derivation of scale and zero-point.
4. Symmetric versus asymmetric quantization.
5. Per-tensor, per-channel, and groupwise intuition.
6. From-scratch quantize/dequantize demo using `quantize_tensor`.
7. Quantization error measurement on a toy weight matrix.
8. Weight-only versus activation quantization.
9. KV-cache memory discussion.
10. KV-cache memory estimate with `estimate_kv_cache_bytes`.
11. Current PyTorch/Hugging Face API map: torchao, bitsandbytes-style 8-bit/4-bit loading, GPTQ, and AWQ.
12. Apple Silicon, CUDA, and CPU caveats.
13. Exercises and solution references.
```

Include at least this code cell:

```python
import torch
from llm_from_scratch.quantization import (
    dequantize_tensor,
    estimate_kv_cache_bytes,
    fake_quantize_tensor,
    quantization_error,
    quantize_per_channel,
    quantize_tensor,
)

weights = torch.randn(4, 8)
q, params = quantize_tensor(weights, num_bits=4, symmetric=True)
weights_hat = dequantize_tensor(q, params)
q_channel, channel_params = quantize_per_channel(weights, axis=0, num_bits=4)
kv_bytes_fp16 = estimate_kv_cache_bytes(2, 4, 16, 128, 1, bytes_per_value=2)
params, quantization_error(weights, weights_hat), q_channel.shape, len(channel_params), kv_bytes_fp16
```

- [ ] **Step 6: Add quantization exercises and solutions**

Create `exercises/quantization.md`:

```markdown
# Quantization Exercises

## Range And Scale

1. For unsigned 8-bit asymmetric quantization, what are `qmin` and `qmax`?
2. Given `x_min = -1`, `x_max = 3`, `qmin = 0`, and `qmax = 255`, compute the scale.

## Implementation

3. Fill in the missing dequantization formula: `x_hat = ____`.
4. Write one sentence explaining fake quantization.

## Error Analysis

5. Quantize `[-1.0, 0.0, 1.0]` to 2 bits. Why should you expect visible reconstruction error?
6. Compare per-tensor and per-channel quantization for a weight matrix whose rows have very different ranges.

## LLM Systems

7. Explain why weight-only quantization primarily targets model memory.
8. Explain why KV-cache quantization affects long-context generation.
9. Name one reason a quantization method may work on CUDA but not Apple Silicon.
```

Create `exercises/solutions/quantization_solutions.md`:

```markdown
# Quantization Solutions

## Range And Scale

1. `qmin = 0` and `qmax = 255`.
2. `scale = (3 - (-1)) / (255 - 0) = 4 / 255`.

## Implementation

3. `scale * (q - zero_point)`.
4. Fake quantization quantizes and immediately dequantizes so the tensor remains floating point while experiencing quantization error.

## Error Analysis

5. Two bits provide only four representable integer levels, so several real values must share coarse bins.
6. Per-tensor quantization uses one range for the whole matrix; per-channel quantization lets each row or column use a tighter range, often reducing error when channel ranges differ.

## LLM Systems

7. Weight-only quantization stores large parameter matrices in fewer bits while often leaving activations in floating point.
8. During autoregressive decoding, the KV cache grows with layers, heads, head dimension, batch size, and sequence length, so lower precision can reduce long-context memory pressure.
9. Quantization speedups depend on kernels and hardware instructions; CUDA may have optimized kernels that are unavailable or immature on Apple Silicon.
```

- [ ] **Step 7: Run tests**

Run:

```bash
uv run pytest tests/test_quantization.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run quick notebook smoke after quantization content is added**

Run:

```bash
uv run python scripts/smoke_notebooks.py --quick
```

Expected: all quick notebooks execute without exceptions.

- [ ] **Step 9: Strip notebook outputs**

Run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks/10_quantization_deep_dive.ipynb
```

Expected: command exits `0` and the quantization notebook has no committed outputs.

- [ ] **Step 10: Commit**

```bash
git add src/llm_from_scratch/quantization.py tests/test_quantization.py docs/math/quantization.md notebooks/10_quantization_deep_dive.ipynb exercises/quantization.md exercises/solutions/quantization_solutions.md
git commit -m "feat: add quantization deep dive"
```

---

## Notebook Content Authoring Contract For Tasks 9-12

Each notebook expanded in Tasks 9-12 must contain enough teaching substance to stand alone. For every core technical concept in a notebook, add cells in this order:

1. **Problem statement markdown:** state the computational problem in plain language.
2. **Notation and derivation markdown:** define symbols, dimensions, and assumptions.
3. **Raw tensor implementation code:** use explicit PyTorch tensor operations before higher-level modules.
4. **Numerical check code:** verify shape, normalization, masking, loss, or error behavior.
5. **Abstraction bridge markdown/code:** show how the same idea appears in `nn.Module`, PyTorch, or Hugging Face.
6. **Exercise checkpoint markdown:** include at least one shape, derivation, implementation, debugging, or extension prompt.

Use this exact representative markdown pattern for the first technical section in every core notebook, then adapt it to the local topic:

````markdown
## Problem: What Must This Component Compute?

At this point in the model, the input has shape `(B, T, C)`: batch size, sequence length, and channel dimension. The component must transform that tensor without leaking information from future token positions.

We will first write the operation directly with tensors so every dimension is visible. Only after the numerical checks pass will we wrap the operation in a reusable module.

Why this matters later: when an architecture paper claims to improve attention, memory, quantization, or world modeling, it is usually changing one of these constraints: what information can move across positions, how much memory is required, what objective is optimized, or which representations are preserved.
````

All committed notebooks must have cleared outputs. After editing notebooks in Tasks 9-12, run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks
```

Do not run `scripts/make_notebooks.py` after Task 7 unless the user explicitly asks to regenerate scaffolds and accepts that notebook content will be overwritten.

---

### Task 9: First-Principles Curriculum Notebooks

**Files:**
- Modify: `notebooks/00_project_orientation.ipynb`
- Modify: `notebooks/01_tensors_autograd_and_probability.ipynb`
- Modify: `notebooks/02_text_tokenization_char_to_subword.ipynb`
- Modify: `notebooks/03_embeddings_and_language_modeling.ipynb`
- Modify: `notebooks/04_attention_from_raw_tensors.ipynb`
- Create: `exercises/01_first_principles.md`
- Create: `exercises/solutions/01_first_principles_solutions.md`

**Interfaces:**
- Consumes: package modules from Tasks 2-4
- Produces: first-principles notebooks that teach tensors, probability, tokenization, embeddings, loss, and attention
- Produces: each edited notebook with at least 6 markdown cells, 4 executable code cells, 2 numerical checks, and 2 exercise checkpoints

- [ ] **Step 1: Expand notebook 00**

Add these sections to `notebooks/00_project_orientation.ipynb`:

```text
1. What the project builds.
2. Why build from scratch before using libraries.
3. Runtime profiles: quick, study, stretch.
4. Device order: CUDA -> MPS -> CPU.
5. Notebook sequence.
6. Verification commands.
7. Long-term bridge: sparse attention, quantization, JEPA, world models.
```

Include this code cell:

```python
from llm_from_scratch.configs import get_device, profile_config

device = get_device()
quick_model, quick_train = profile_config("quick")
device, quick_model, quick_train
```

- [ ] **Step 2: Expand notebook 01**

Add derivations and code cells for:

```text
1. Tensor shapes and broadcasting.
2. Autograd on scalar objectives.
3. Categorical next-token modeling.
4. Factorization p(x_1, ..., x_T) = product_t p(x_t | x_<t).
5. Softmax and numerical stability.
6. Cross-entropy as negative log-likelihood.
7. Gradient of softmax plus cross-entropy.
```

Include this code cell:

```python
import torch
from llm_from_scratch.functional import cross_entropy_from_logits, stable_softmax

logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
target = torch.tensor([2])
loss = cross_entropy_from_logits(logits, target)
loss.backward()
stable_softmax(logits.detach()), loss.item(), logits.grad
```

- [ ] **Step 3: Expand notebook 02**

Add sections:

```text
1. Text as a sequence of discrete symbols.
2. Character vocabulary construction.
3. Encoding and decoding.
4. Why character tokenization is educational but unrealistic.
5. BPE intuition: merges, subwords, vocabulary size.
6. Hugging Face tokenizers comparison.
```

Include this code cell:

```python
from pathlib import Path
from llm_from_scratch.tokenization import CharTokenizer, train_bpe_tokenizer

text = Path("../data/tiny_corpus.txt").read_text(encoding="utf-8")
char_tokenizer = CharTokenizer.from_text(text)
char_ids = char_tokenizer.encode(text[:80])
bpe_tokenizer = train_bpe_tokenizer([text], vocab_size=64)
bpe_ids = bpe_tokenizer.encode(text[:80]).ids
char_tokenizer.vocab_size, len(bpe_tokenizer.get_vocab()), char_ids[:20], bpe_ids[:20]
```

- [ ] **Step 4: Expand notebook 03**

Add sections:

```text
1. One-hot vectors versus learned lookup matrices.
2. Embedding matrix shape.
3. Logits as unnormalized class scores.
4. Label shifting for next-token prediction.
5. Loss sanity checks.
```

Include this code cell:

```python
import torch

vocab_size = 10
n_embd = 4
embedding = torch.nn.Embedding(vocab_size, n_embd)
ids = torch.tensor([[1, 2, 3, 4]])
x = embedding(ids)
x.shape
```

- [ ] **Step 5: Expand notebook 04**

Add sections:

```text
1. Queries, keys, values.
2. Dot products as content-based addressing.
3. Scaling by sqrt(d_k).
4. Causal masking.
5. The T x T score matrix and O(T^2) bottleneck.
6. Why this bottleneck motivates sparse and subquadratic attention.
```

Include this code cell:

```python
import torch
from llm_from_scratch.functional import causal_mask, scaled_dot_product_attention

T = 6
D = 8
q = torch.randn(1, T, D)
k = torch.randn(1, T, D)
v = torch.randn(1, T, D)
out, weights = scaled_dot_product_attention(q, k, v, causal_mask(T))
out.shape, weights.shape, weights[0, 0]
```

- [ ] **Step 6: Add exercises and solutions**

Create `exercises/01_first_principles.md`:

```markdown
# First-Principles Exercises

## Shape Reasoning

1. Given logits with shape `(B, T, V)` and targets with shape `(B, T)`, write the flattened shapes used by cross-entropy.
2. If queries have shape `(B, H, T, D)` and keys have shape `(B, H, T, D)`, what shape does `q @ k.transpose(-2, -1)` produce?
3. For a sequence length `T = 2048`, how many attention scores does one head compute?

## Derivation Checks

4. Explain why subtracting `max(logits)` does not change softmax probabilities.
5. Starting from `L = -log softmax(z)_y`, derive `dL/dz_i = p_i - 1[i = y]`.

## Implementation Gaps

6. Write a PyTorch expression that creates a lower-triangular causal mask for `T = 4`.
7. Fill in the missing line: `log_probs = logits - ____`.

## Debugging

8. A model's validation loss is suspiciously low after one step. Explain how future-token leakage could cause this.
9. A softmax call returns `nan` for large logits. Name the numerical stability fix.

## Extension

10. Explain why character tokenization is useful for learning but not ideal for real LLMs.
```

Create `exercises/solutions/01_first_principles_solutions.md`:

```markdown
# First-Principles Solutions

## Shape Reasoning

1. Logits become `(B*T, V)` and targets become `(B*T)`.
2. The score tensor has shape `(B, H, T, T)`.
3. One head computes `2048 * 2048 = 4,194,304` attention scores.

## Derivation Checks

4. Softmax is invariant to adding or subtracting the same constant from every logit because the common exponential factor cancels from numerator and denominator.
5. The derivative of `log(sum_j exp(z_j))` is `p_i`, while the derivative of `-z_y` is `-1` only for the target index, giving `p_i - 1[i = y]`.

## Implementation Gaps

6. `torch.tril(torch.ones(4, 4, dtype=torch.bool))`.
7. `torch.logsumexp(logits, dim=-1, keepdim=True)`.

## Debugging

8. If the mask allows a position to read future tokens, the model can copy label information from the input rather than learning the conditional distribution.
9. Subtract the maximum logit before exponentiating.

## Extension

10. Character tokenization keeps the mapping transparent, but it creates long sequences and misses reusable subword structure.
```

- [ ] **Step 7: Smoke first-principles notebooks**

Run:

```bash
uv run python scripts/smoke_notebooks.py --quick
```

Expected: all quick notebooks execute without exceptions.

- [ ] **Step 8: Strip notebook outputs**

Run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks
```

Expected: command exits `0` and edited notebooks have no committed outputs.

- [ ] **Step 9: Commit**

```bash
git add notebooks/00_project_orientation.ipynb notebooks/01_tensors_autograd_and_probability.ipynb notebooks/02_text_tokenization_char_to_subword.ipynb notebooks/03_embeddings_and_language_modeling.ipynb notebooks/04_attention_from_raw_tensors.ipynb exercises/01_first_principles.md exercises/solutions/01_first_principles_solutions.md
git commit -m "docs: expand first principles curriculum notebooks"
```

---

### Task 10: Architecture, Training, Generation, And Fine-Tuning Notebooks

**Files:**
- Modify: `notebooks/05_transformer_block_from_scratch.ipynb`
- Modify: `notebooks/06_training_loop_loss_and_optimization.ipynb`
- Modify: `notebooks/07_generation_sampling_and_evaluation.ipynb`
- Modify: `notebooks/08_finetuning_toy_instruction_task.ipynb`
- Create: `exercises/02_training_and_generation.md`
- Create: `exercises/solutions/02_training_and_generation_solutions.md`

**Interfaces:**
- Consumes: `TinyGPT`
- Consumes: `train_steps`
- Consumes: `generate`
- Produces: notebooks that train, evaluate, generate, and fine-tune the toy model
- Produces: each edited notebook with at least 6 markdown cells, 4 executable code cells, 1 deterministic seeded demo, and 2 exercise checkpoints

- [ ] **Step 1: Expand notebook 05**

Add sections:

```text
1. Residual stream.
2. Layer normalization.
3. Multi-head causal self-attention module.
4. Feed-forward network.
5. Transformer block.
6. Full decoder-only GPT.
7. Parameter count and scaling intuition.
```

Include this code cell:

```python
import torch
from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.model import TinyGPT, count_parameters

cfg = ModelConfig(vocab_size=64, block_size=16, n_embd=32, n_head=4, n_layer=2)
model = TinyGPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
logits, loss = model(idx, idx)
logits.shape, loss.item(), count_parameters(model)
```

- [ ] **Step 2: Expand notebook 06**

Add sections:

```text
1. Train/validation split.
2. Label shifting.
3. AdamW update rule.
4. Gradient clipping.
5. Tiny overfit test.
6. Quick versus study run configuration.
7. Loss curve interpretation.
```

Include this code cell:

```python
import torch
from llm_from_scratch.configs import ModelConfig, set_seed
from llm_from_scratch.model import TinyGPT
from llm_from_scratch.train import overfit_tiny_batch

set_seed(123)
cfg = ModelConfig(vocab_size=16, block_size=8, n_embd=32, n_head=4, n_layer=1, dropout=0.0)
model = TinyGPT(cfg)
x = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
y = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
first_loss, last_loss = overfit_tiny_batch(model, x, y, steps=30, lr=1e-2)
first_loss, last_loss
```

- [ ] **Step 3: Expand notebook 07**

Add sections:

```text
1. Autoregressive generation loop.
2. Temperature.
3. Top-k filtering.
4. Top-p filtering.
5. Perplexity.
6. Qualitative generation checks.
```

Include this code cell:

```python
import torch
from llm_from_scratch.configs import ModelConfig
from llm_from_scratch.generate import generate
from llm_from_scratch.model import TinyGPT

cfg = ModelConfig(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1)
model = TinyGPT(cfg)
idx = torch.zeros((1, 3), dtype=torch.long)
generate(model, idx, max_new_tokens=5, temperature=1.0, top_k=5)
```

- [ ] **Step 4: Expand notebook 08**

Add sections:

```text
1. What supervised fine-tuning changes.
2. Prompt-response formatting.
3. Why this toy task is not instruction tuning at production scale.
4. Loss masking discussion.
5. Small fine-tuning loop using toy examples.
```

Include this code cell:

```python
from llm_from_scratch.data import toy_instruction_examples

for prompt, response in toy_instruction_examples():
    print(f"### Instruction:\\n{prompt}\\n\\n### Response:\\n{response}\\n")
```

- [ ] **Step 5: Add exercises and solutions**

Create `exercises/02_training_and_generation.md`:

```markdown
# Training And Generation Exercises

## Shape Reasoning

1. Explain why the target tensor is shifted one position relative to the input tensor.
2. If `idx` has shape `(B, T)`, what shape should logits have before cross-entropy?

## Derivation Checks

3. Write the AdamW parameter update in words, including how weight decay differs from L2 penalty inside the gradient.
4. Explain why perplexity is `exp(loss)` when loss is average negative log-likelihood.

## Implementation Gaps

5. Write the one-line tensor operation that appends `next_token` with shape `(B, 1)` to `idx` with shape `(B, T)`.
6. Add a top-k filter call before sampling from logits.

## Debugging

7. Describe one symptom of causal mask leakage.
8. A tiny overfit test does not reduce loss. List three likely causes.

## Extension

9. Increase temperature from `0.7` to `1.3`. Predict how generations should change.
10. Compare top-k and top-p sampling in one paragraph.
11. Explain why a tiny overfit test is a useful debugging tool.
```

Create `exercises/solutions/02_training_and_generation_solutions.md`:

```markdown
# Training And Generation Solutions

## Shape Reasoning

1. Each input position predicts the next token, so targets are the same sequence shifted left by one.
2. Logits should have shape `(B, T, V)`.

## Derivation Checks

3. AdamW updates parameters using Adam's adaptive moment estimates, then applies decoupled weight decay directly to the parameter value rather than adding the penalty into the gradient.
4. Average negative log-likelihood is the negative log probability per token, so exponentiating returns the inverse geometric mean probability assigned to the correct token.

## Implementation Gaps

5. `idx = torch.cat((idx, next_token), dim=1)`.
6. `logits = top_k_filter(logits, top_k)`.

## Debugging

7. Validation loss may look artificially strong because the model can read future labels during training.
8. Likely causes include no gradient flow, labels not aligned with inputs, optimizer not stepping, learning rate too low or too high, model left in eval mode with unexpected behavior, or all targets being identical by mistake.

## Extension

9. Higher temperature flattens the distribution, increasing randomness and lowering confidence in the top token.
10. Top-k keeps a fixed number of candidates; top-p keeps the smallest high-probability prefix whose cumulative probability exceeds `p`.
11. A tiny overfit test proves the model, loss, gradients, and optimizer can fit a memorization problem before larger data introduces noise.
```

- [ ] **Step 6: Run package tests and notebook smoke**

Run:

```bash
uv run pytest -q
uv run python scripts/smoke_notebooks.py --quick
```

Expected: all tests pass and quick notebooks execute without exceptions.

- [ ] **Step 7: Strip notebook outputs**

Run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks
```

Expected: command exits `0` and edited notebooks have no committed outputs.

- [ ] **Step 8: Commit**

```bash
git add notebooks/05_transformer_block_from_scratch.ipynb notebooks/06_training_loop_loss_and_optimization.ipynb notebooks/07_generation_sampling_and_evaluation.ipynb notebooks/08_finetuning_toy_instruction_task.ipynb exercises/02_training_and_generation.md exercises/solutions/02_training_and_generation_solutions.md
git commit -m "docs: add model training generation and fine tuning notebooks"
```

---

### Task 11: Hugging Face Translation And Modern Orientation

**Files:**
- Modify: `notebooks/09_hugging_face_translation_layer.ipynb`
- Modify: `notebooks/11_modern_llm_orientation.ipynb`
- Create: `docs/notes/library_translation_map.md`
- Create: `docs/notes/modern_llm_orientation.md`

**Interfaces:**
- Consumes: package model/data/tokenizer concepts
- Produces: explicit mapping from from-scratch pieces to modern libraries
- Produces: orientation notes for instruction tuning, RAG, deployment, and systems concerns
- Produces: each edited notebook with at least 5 markdown cells, 2 executable code cells, and a table mapping handmade components to library abstractions

- [ ] **Step 1: Expand Hugging Face translation notebook**

Add sections:

```text
1. From text file to Hugging Face datasets.
2. From CharTokenizer to tokenizers BPE.
3. From ModelConfig to transformer config objects.
4. From TinyGPT.forward to AutoModelForCausalLM.
5. From handmade training loop to Trainer.
6. From handmade generate loop to generate().
7. What abstractions hide and what they cannot remove.
```

Include this code cell:

```python
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import GPT2Config, GPT2LMHeadModel

dataset = Dataset.from_dict({"text": ["attention reads earlier tokens", "loss trains logits"]})
config = GPT2Config(vocab_size=128, n_positions=32, n_embd=64, n_layer=2, n_head=4)
model = GPT2LMHeadModel(config)
dataset, Tokenizer(BPE()), model.num_parameters()
```

- [ ] **Step 2: Add library translation note**

Create `docs/notes/library_translation_map.md`:

```markdown
# Library Translation Map

| From-scratch concept | PyTorch or Hugging Face abstraction |
| --- | --- |
| Token list | `Dataset` column or tensor dataset |
| Character tokenizer | `tokenizers.Tokenizer` or `AutoTokenizer` |
| `ModelConfig` | Transformers config object |
| `TinyGPT` | `AutoModelForCausalLM`-style model |
| Handmade training loop | `Trainer` or custom Accelerate loop |
| Handmade generation loop | `GenerationMixin.generate()` |
| Checkpoint dictionary | `save_pretrained()` and `from_pretrained()` |

The libraries reduce boilerplate. They do not remove the need to understand shapes, masks, loss shifting, optimizer behavior, device placement, or evaluation validity.
```

- [ ] **Step 3: Expand modern orientation notebook**

Add sections:

```text
1. Pretraining versus supervised fine-tuning.
2. Instruction tuning.
3. Preference optimization overview: RLHF and DPO.
4. Retrieval-augmented generation.
5. Quantization and serving constraints.
6. Evaluation hazards.
7. Why architecture knowledge still matters.
```

Include this code cell:

```python
topics = [
    "instruction tuning",
    "preference optimization",
    "retrieval augmented generation",
    "quantization",
    "evaluation",
]
topics
```

- [ ] **Step 4: Add modern orientation note**

Create `docs/notes/modern_llm_orientation.md`:

```markdown
# Modern LLM Orientation

The from-scratch GPT build gives names and shapes to the core moving parts. Modern LLM practice layers additional training objectives, data pipelines, retrieval systems, compression methods, and deployment constraints on top of those parts.

This first curriculum treats RLHF, DPO, RAG, and deployment as orientation topics. They are important, but implementing them before the core language-modeling path is solid would blur the main learning goal.
```

- [ ] **Step 5: Run smoke checks**

Run:

```bash
uv run pytest -q
uv run python scripts/smoke_notebooks.py --quick
```

Expected: package tests pass and quick notebooks execute without exceptions.

- [ ] **Step 6: Strip notebook outputs**

Run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks
```

Expected: command exits `0` and edited notebooks have no committed outputs.

- [ ] **Step 7: Commit**

```bash
git add notebooks/09_hugging_face_translation_layer.ipynb notebooks/11_modern_llm_orientation.ipynb docs/notes/library_translation_map.md docs/notes/modern_llm_orientation.md
git commit -m "docs: add hugging face translation and modern orientation"
```

---

### Task 12: Beyond-Transformers And World-Models Orientation

**Files:**
- Modify: `notebooks/12_beyond_transformers_and_world_models.ipynb`
- Create: `docs/notes/beyond_transformers_reading_map.md`
- Create: `exercises/03_beyond_transformers.md`
- Create: `exercises/solutions/03_beyond_transformers_solutions.md`

**Interfaces:**
- Consumes: attention derivation and `O(T^2)` bridge from earlier notebooks
- Produces: a conceptual map for sparse/subquadratic attention, post-transformer approaches, JEPA, and world models
- Produces: notebook content with at least 8 markdown cells, 2 executable code cells, and one complexity comparison table

- [ ] **Step 1: Expand beyond-transformers notebook**

Add sections:

```text
1. The vanilla attention bottleneck.
2. Sparse and local attention.
3. Sliding-window, block, and global-token patterns.
4. Grouped-query and multi-query attention.
5. Subquadratic attention ambitions.
6. Recurrence and state-space alternatives.
7. Predictive representation learning.
8. JEPA-style objective intuition.
9. World models, memory, planning, grounding, and agency.
10. How to read future architecture papers.
```

Include this code cell:

```python
sequence_lengths = [128, 512, 2048, 8192]
quadratic_scores = {n: n * n for n in sequence_lengths}
linear_window_scores = {n: n * 256 for n in sequence_lengths}
quadratic_scores, linear_window_scores
```

- [ ] **Step 2: Add reading map**

Create `docs/notes/beyond_transformers_reading_map.md`:

```markdown
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
```

- [ ] **Step 3: Add exercises and solutions**

Create `exercises/03_beyond_transformers.md`:

```markdown
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
```

Create `exercises/solutions/03_beyond_transformers_solutions.md`:

```markdown
# Beyond-Transformers Solutions

## Complexity Reasoning

1. Full attention computes `T^2` scores per head; sliding-window attention computes roughly `T*w` scores.
2. Full attention computes `8192^2 = 67,108,864` scores; sliding-window attention computes roughly `8192 * 256 = 2,097,152` scores.

## Architecture Tradeoffs

3. Sparse attention can reduce memory or compute; it may also remove useful long-range interactions unless the sparsity pattern is well matched to the task.
4. Grouped-query attention shares keys and values across multiple query heads, reducing the amount of KV-cache data stored and read during decoding.
5. Recurrence or state-space models can avoid materializing a full `T x T` attention matrix and can carry compressed state across long sequences.

## Objective Reasoning

6. Predictive representation objectives can train latent states to predict abstract future representations instead of directly predicting the next discrete token.
7. JEPA-style methods are interesting because they encourage predictive world representations without requiring every target to be reconstructed at pixel or token level.

## Paper Reading

8. Ask what objective is optimized, what bottleneck is targeted, and whether the evaluation supports the architectural claim.
9. Classify the claim by asking what changed: the loss, the architecture's assumptions, the asymptotic or hardware cost, the data source, or the benchmark evidence.
```

- [ ] **Step 4: Run smoke checks**

Run:

```bash
uv run pytest -q
uv run python scripts/smoke_notebooks.py --quick
```

Expected: package tests pass and quick notebooks execute without exceptions.

- [ ] **Step 5: Strip notebook outputs**

Run:

```bash
uv run python scripts/strip_notebook_outputs.py notebooks
```

Expected: command exits `0` and edited notebooks have no committed outputs.

- [ ] **Step 6: Commit**

```bash
git add notebooks/12_beyond_transformers_and_world_models.ipynb docs/notes/beyond_transformers_reading_map.md exercises/03_beyond_transformers.md exercises/solutions/03_beyond_transformers_solutions.md
git commit -m "docs: add beyond transformers orientation"
```

---

### Task 13: Final Verification, README Polish, And Curriculum Index

**Files:**
- Modify: `README.md`
- Create: `docs/notes/curriculum_index.md`
- Modify: `scripts/smoke_notebooks.py`

**Interfaces:**
- Consumes: all notebooks and package modules
- Produces: final handoff commands and a navigable curriculum index

- [ ] **Step 1: Add curriculum index**

Create `docs/notes/curriculum_index.md`:

````markdown
# Curriculum Index

## Core Build

1. Project orientation
2. Tensors, autograd, and probability
3. Tokenization
4. Embeddings and language modeling
5. Attention from raw tensors
6. Transformer block
7. Training loop
8. Generation and evaluation
9. Toy supervised fine-tuning

## Abstraction And Orientation

10. Hugging Face translation layer
11. Quantization deep dive
12. Modern LLM orientation
13. Beyond transformers and world models

## Verification

Run:

```bash
uv run pytest
uv run python scripts/smoke_notebooks.py --quick
```
````

- [ ] **Step 2: Update README**

Revise `README.md` to include:

```text
1. Setup commands.
2. Notebook order.
3. Runtime profile definitions.
4. Test commands.
5. Note that generated checkpoints, model files, and downloaded datasets are ignored by Git.
6. Explanation that quantization and beyond-transformer material are dedicated orientation tracks.
```

- [ ] **Step 3: Ensure full test suite passes**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Ensure quick notebook smoke passes**

Run:

```bash
uv run python scripts/smoke_notebooks.py --quick
```

Expected: selected notebooks execute without exceptions.

- [ ] **Step 5: Check repository status**

Run:

```bash
git status --short
```

Expected: only intended final files are modified or untracked before commit.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/notes/curriculum_index.md scripts/smoke_notebooks.py
git commit -m "docs: finalize curriculum handoff"
```

---

## Final Acceptance Checklist

Before reporting implementation complete, run:

```bash
uv run pytest -q
uv run python scripts/smoke_notebooks.py --quick
git status --short
```

Required results:

- `pytest` exits `0`.
- quick notebook smoke exits `0`.
- `git status --short` shows no unintended untracked files.
- `.DS_Store` is ignored and not committed.
- The project contains all 13 notebooks.
- The project contains exercises and solutions.
- The project contains a dedicated quantization deep dive.
- The project contains a beyond-transformers/world-model orientation notebook.
- The package can train a tiny model enough for the overfit test to reduce loss.

## Recommended Execution Order

Use subagent-driven development for Tasks 1-13. Each task is independently reviewable and has its own test and commit gate. If implementing inline, stop after each task for review before proceeding.
