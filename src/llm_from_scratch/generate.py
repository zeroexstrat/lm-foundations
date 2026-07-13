from __future__ import annotations

import torch
import torch.nn.functional as F


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k <= 0:
        return logits

    limit = min(k, logits.size(-1))
    values, _ = torch.topk(logits, limit, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    if p is None or p >= 1.0:
        return logits
    if p <= 0.0:
        raise ValueError("top_p must be greater than 0")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)

    remove = cumulative_probs > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    kept_sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_indices, kept_sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    scaled_logits = logits / temperature
    filtered_logits = top_k_filter(scaled_logits, top_k)
    filtered_logits = top_p_filter(filtered_logits, top_p)
    probs = F.softmax(filtered_logits, dim=-1)
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
    was_training = model.training
    model.eval()

    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        idx = torch.cat((idx, next_token), dim=1)

    if was_training:
        model.train()
    return idx
