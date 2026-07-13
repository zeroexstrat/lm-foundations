from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


FeatureMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class LinearAttentionTrace:
    intermediate_shapes: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    sequence_square_matrices: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def record(self, name: str, tensor: torch.Tensor, *, sequence_length: int) -> None:
        shape = tuple(int(dim) for dim in tensor.shape)
        self.intermediate_shapes.append((name, shape))
        if tensor.ndim >= 2 and shape[-2:] == (sequence_length, sequence_length):
            self.sequence_square_matrices.append((name, shape))


def _record(
    trace: LinearAttentionTrace | None,
    name: str,
    tensor: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    if trace is not None:
        trace.record(name, tensor, sequence_length=sequence_length)
    return tensor


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> None:
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must have at least 2 dimensions")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share leading batch dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("key/value sequence length must match")
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention requires matching query and key sequence lengths")


def elu_plus_one_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def kernel_attention_via_feature_map(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    feature_map: FeatureMap = elu_plus_one_feature_map,
    causal: bool = False,
    eps: float = 1e-9,
) -> torch.Tensor:
    _validate_attention_inputs(q, k, v, causal=causal)

    phi_q = feature_map(q)
    phi_k = feature_map(k)
    weights = phi_q @ phi_k.transpose(-2, -1)
    if causal:
        mask = torch.tril(
            torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device)
        )
        weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (weights / denom) @ v


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    feature_map: FeatureMap = elu_plus_one_feature_map,
    eps: float = 1e-9,
    trace: LinearAttentionTrace | None = None,
) -> torch.Tensor:
    _validate_attention_inputs(q, k, v, causal=False)
    sequence_length = q.shape[-2]

    phi_q = _record(trace, "feature_queries", feature_map(q), sequence_length=sequence_length)
    phi_k = _record(trace, "feature_keys", feature_map(k), sequence_length=sequence_length)
    feature_key_sum = _record(
        trace,
        "feature_key_sum",
        phi_k.sum(dim=-2),
        sequence_length=sequence_length,
    )
    feature_kv = _record(
        trace,
        "feature_kv",
        torch.einsum("...tm,...td->...md", phi_k, v),
        sequence_length=sequence_length,
    )
    numerator = _record(
        trace,
        "numerator",
        torch.einsum("...tm,...md->...td", phi_q, feature_kv),
        sequence_length=sequence_length,
    )
    denominator = _record(
        trace,
        "denominator",
        torch.einsum("...tm,...m->...t", phi_q, feature_key_sum).clamp_min(eps),
        sequence_length=sequence_length,
    )
    return numerator / denominator.unsqueeze(-1)


def causal_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    feature_map: FeatureMap = elu_plus_one_feature_map,
    eps: float = 1e-9,
    trace: LinearAttentionTrace | None = None,
) -> torch.Tensor:
    _validate_attention_inputs(q, k, v, causal=True)
    sequence_length = q.shape[-2]

    phi_q = _record(trace, "feature_queries", feature_map(q), sequence_length=sequence_length)
    phi_k = _record(trace, "feature_keys", feature_map(k), sequence_length=sequence_length)
    feature_kv_terms = _record(
        trace,
        "feature_kv_terms",
        phi_k.unsqueeze(-1) * v.unsqueeze(-2),
        sequence_length=sequence_length,
    )
    feature_kv_prefix = _record(
        trace,
        "feature_kv_prefix",
        torch.cumsum(feature_kv_terms, dim=-3),
        sequence_length=sequence_length,
    )
    feature_key_prefix = _record(
        trace,
        "feature_key_prefix",
        torch.cumsum(phi_k, dim=-2),
        sequence_length=sequence_length,
    )
    numerator = _record(
        trace,
        "numerator",
        torch.einsum("...tm,...tmd->...td", phi_q, feature_kv_prefix),
        sequence_length=sequence_length,
    )
    denominator = _record(
        trace,
        "denominator",
        torch.einsum("...tm,...tm->...t", phi_q, feature_key_prefix).clamp_min(eps),
        sequence_length=sequence_length,
    )
    return numerator / denominator.unsqueeze(-1)


def gaussian_random_projection(
    input_dim: int,
    num_features: int,
    *,
    seed: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if num_features <= 0:
        raise ValueError("num_features must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(num_features, input_dim, generator=generator, dtype=torch.float64)
    if dtype is not None or device is not None:
        projection = projection.to(device=device, dtype=dtype)
    return projection


def softmax_positive_random_feature_map(
    x: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    if projection.ndim != 2:
        raise ValueError("projection must be a 2D tensor")
    if projection.shape[-1] != x.shape[-1]:
        raise ValueError("projection input dimension must match x.shape[-1]")

    scaled = x / math.sqrt(math.sqrt(x.shape[-1]))
    projected = scaled @ projection.transpose(0, 1)
    squared_norm = 0.5 * scaled.square().sum(dim=-1, keepdim=True)
    return torch.exp(projected - squared_norm) / math.sqrt(projection.shape[0])


def softmax_random_feature_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_features: int,
    seed: int,
    causal: bool = False,
    eps: float = 1e-9,
    trace: LinearAttentionTrace | None = None,
) -> torch.Tensor:
    _validate_attention_inputs(q, k, v, causal=causal)
    projection = gaussian_random_projection(
        q.shape[-1],
        num_features,
        seed=seed,
        device=q.device,
        dtype=q.dtype,
    )

    def feature_map(x: torch.Tensor) -> torch.Tensor:
        return softmax_positive_random_feature_map(x, projection)

    if causal:
        return causal_linear_attention(q, k, v, feature_map=feature_map, eps=eps, trace=trace)
    return linear_attention(q, k, v, feature_map=feature_map, eps=eps, trace=trace)
