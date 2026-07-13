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
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
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
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Propagates to child processes started after this call; it cannot
    # retroactively change the current interpreter's hash seed.
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
