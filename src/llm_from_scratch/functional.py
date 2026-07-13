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
