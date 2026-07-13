from __future__ import annotations

import math

import torch


def _broadcast_mask(mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if mask.shape != scores.shape:
        mask = torch.broadcast_to(mask, scores.shape)
    return mask


def _masked_stable_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        shifted = logits - logits.max(dim=-1, keepdim=True).values
        exp = torch.exp(shifted)
        return exp / exp.sum(dim=-1, keepdim=True)

    broadcast_mask = _broadcast_mask(mask, logits)
    masked_logits = logits.masked_fill(~broadcast_mask, float("-inf"))
    row_max = masked_logits.max(dim=-1, keepdim=True).values
    safe_row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    shifted = masked_logits - safe_row_max
    exp = torch.exp(shifted) * broadcast_mask.to(dtype=logits.dtype)
    denom = exp.sum(dim=-1, keepdim=True)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    weights = exp / safe_denom
    return torch.where(denom > 0, weights, torch.zeros_like(weights))


def exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must have at least 2 dimensions")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share leading batch dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must share the key/value sequence length")

    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = q @ k.transpose(-2, -1) * scale
    if mask is not None:
        broadcast_mask = _broadcast_mask(mask, logits)
        if (~broadcast_mask.any(dim=-1)).any():
            raise ValueError("mask row has no valid key; every query row must keep at least one key")
        weights = _masked_stable_softmax(logits, mask=broadcast_mask)
        return weights @ v, weights
    weights = _masked_stable_softmax(logits, mask=mask)
    return weights @ v, weights


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    safe_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return -(weights * safe_weights.log()).sum(dim=-1)


def check_row_stochastic(weights: torch.Tensor, atol: float = 1e-6) -> bool:
    if not torch.isfinite(weights).all():
        return False
    if (weights < -atol).any():
        return False
    if (weights > 1.0 + atol).any():
        return False
    row_sums = weights.sum(dim=-1)
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol, rtol=0.0)


def _attention_objective(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    output, weights = exact_attention(q, k, v, mask=mask)
    return output.square().sum() + weights.square().sum()


def _central_difference_gradient(
    target: torch.Tensor,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    name: str,
    mask: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    grad = torch.zeros_like(target)
    flat_grad = grad.reshape(-1)
    for index in range(target.numel()):
        plus_q, plus_k, plus_v = q.clone(), k.clone(), v.clone()
        minus_q, minus_k, minus_v = q.clone(), k.clone(), v.clone()
        plus_target = {"q": plus_q, "k": plus_k, "v": plus_v}[name].reshape(-1)
        minus_target = {"q": minus_q, "k": minus_k, "v": minus_v}[name].reshape(-1)
        plus_target[index] += eps
        minus_target[index] -= eps
        plus_value = _attention_objective(plus_q, plus_k, plus_v, mask=mask)
        minus_value = _attention_objective(minus_q, minus_k, minus_v, mask=mask)
        flat_grad[index] = (plus_value - minus_value) / (2.0 * eps)
    return grad


def finite_difference_attention_gradients(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, dict[str, torch.Tensor | float]]:
    q_var = q.clone().detach().requires_grad_(True)
    k_var = k.clone().detach().requires_grad_(True)
    v_var = v.clone().detach().requires_grad_(True)
    objective = _attention_objective(q_var, k_var, v_var, mask=mask)
    objective.backward()

    diagnostics: dict[str, dict[str, torch.Tensor | float]] = {}
    analytic_by_name = {
        "q": q_var.grad.detach().clone(),
        "k": k_var.grad.detach().clone(),
        "v": v_var.grad.detach().clone(),
    }
    base_tensors = {
        "q": q.clone().detach(),
        "k": k.clone().detach(),
        "v": v.clone().detach(),
    }
    for name, analytic in analytic_by_name.items():
        numerical = _central_difference_gradient(
            base_tensors[name],
            q=base_tensors["q"],
            k=base_tensors["k"],
            v=base_tensors["v"],
            name=name,
            mask=mask,
            eps=eps,
        )
        diagnostics[name] = {
            "analytic": analytic,
            "numerical": numerical,
            "max_abs_error": float((analytic - numerical).abs().max().item()),
        }
    return diagnostics
