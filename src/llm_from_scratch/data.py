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
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(tokens) <= block_size:
        raise ValueError("tokens length must exceed block_size")

    ix = torch.randint(0, len(tokens) - block_size, (batch_size,), device=tokens.device)
    starts = ix.tolist()
    x = torch.stack([tokens[i : i + block_size] for i in starts])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def toy_instruction_examples() -> list[tuple[str, str]]:
    return [
        ("Define attention in one sentence.", "Attention mixes information across token positions."),
        ("What does a language model predict?", "It predicts a distribution over the next token."),
        ("Why use a causal mask?", "The mask prevents tokens from reading future tokens."),
    ]
