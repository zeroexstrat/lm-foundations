from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingResult:
    topk_indices: torch.Tensor
    combine_weights: torch.Tensor
    accepted_mask: torch.Tensor
    position_in_expert: torch.Tensor
    gate_probs: torch.Tensor
    attempted: torch.Tensor
    load: torch.Tensor
    dropped: torch.Tensor
    capacity: int


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor of shape (tokens, experts)")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have at least one token and one expert")


def expert_capacity(
    num_tokens: int,
    num_experts: int,
    *,
    top_k: int,
    capacity_factor: float = 1.0,
    min_capacity: int = 1,
) -> int:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must lie between 1 and num_experts")
    if capacity_factor <= 0:
        raise ValueError("capacity_factor must be positive")
    if min_capacity <= 0:
        raise ValueError("min_capacity must be positive")

    mean_assignments = capacity_factor * num_tokens * top_k / num_experts
    return max(min_capacity, math.ceil(mean_assignments))


def route_tokens(
    logits: torch.Tensor,
    *,
    top_k: int,
    capacity_factor: float = 1.0,
    min_capacity: int = 1,
    normalize_topk: bool = True,
) -> RoutingResult:
    _validate_logits(logits)
    num_tokens, num_experts = logits.shape
    capacity = expert_capacity(
        num_tokens,
        num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        min_capacity=min_capacity,
    )

    gate_probs = torch.softmax(logits, dim=-1)
    raw_weights, raw_indices = torch.topk(gate_probs, k=top_k, dim=-1)
    combine_weights = raw_weights.clone()
    if normalize_topk:
        combine_weights = combine_weights / combine_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(combine_weights.dtype).eps
        )

    accepted_mask = torch.zeros_like(raw_indices, dtype=torch.bool)
    routed_indices = raw_indices.clone()
    position_in_expert = torch.full_like(raw_indices, fill_value=-1)
    load = torch.zeros(num_experts, dtype=torch.int64, device=logits.device)
    attempted = torch.zeros_like(load)

    # Higher-ranked experts claim capacity first across the batch.
    for rank in range(top_k):
        for token_idx in range(num_tokens):
            expert_idx = int(raw_indices[token_idx, rank].item())
            attempted[expert_idx] += 1
            if int(load[expert_idx].item()) >= capacity:
                routed_indices[token_idx, rank] = -1
                combine_weights[token_idx, rank] = 0.0
                continue

            accepted_mask[token_idx, rank] = True
            position_in_expert[token_idx, rank] = load[expert_idx]
            load[expert_idx] += 1

    if normalize_topk:
        row_sums = combine_weights.sum(dim=-1, keepdim=True)
        nonzero_rows = row_sums.squeeze(-1) > 0
        combine_weights[nonzero_rows] = combine_weights[nonzero_rows] / row_sums[nonzero_rows]

    dropped = attempted - load
    return RoutingResult(
        topk_indices=routed_indices,
        combine_weights=combine_weights,
        accepted_mask=accepted_mask,
        position_in_expert=position_in_expert,
        gate_probs=gate_probs,
        attempted=attempted,
        load=load,
        dropped=dropped,
        capacity=capacity,
    )


def load_balance_loss(gate_probs: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
    _validate_logits(gate_probs)
    if load.ndim != 1:
        raise ValueError("load must be a 1D tensor")
    if load.shape[0] != gate_probs.shape[1]:
        raise ValueError("load must have one entry per expert")

    total_assignments = int(load.sum().item())
    if total_assignments == 0:
        return torch.zeros((), dtype=gate_probs.dtype, device=gate_probs.device)

    mean_prob = gate_probs.mean(dim=0)
    load_fraction = load.to(dtype=gate_probs.dtype, device=gate_probs.device) / total_assignments
    return gate_probs.shape[1] * torch.sum(mean_prob * load_fraction)


def importance_loss(gate_probs: torch.Tensor) -> torch.Tensor:
    _validate_logits(gate_probs)
    num_experts = gate_probs.shape[1]
    mean_prob = gate_probs.mean(dim=0)
    uniform = torch.full_like(mean_prob, 1.0 / num_experts)
    return num_experts * torch.sum((mean_prob - uniform) ** 2)


def load_coefficient_of_variation_squared(load: torch.Tensor) -> torch.Tensor:
    if load.ndim != 1:
        raise ValueError("load must be a 1D tensor")
    if load.numel() == 0:
        raise ValueError("load must be non-empty")

    load_float = load.to(dtype=torch.float64)
    mean = load_float.mean()
    if float(mean.item()) == 0.0:
        return torch.zeros((), dtype=load_float.dtype, device=load_float.device)
    variance = torch.mean((load_float - mean) ** 2)
    return variance / (mean**2)
