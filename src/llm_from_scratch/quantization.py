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
    return q.to(torch.int32), QuantizationParams(
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
    )


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
    return (
        keys_and_values
        * n_layer
        * n_head
        * head_dim
        * seq_len
        * batch_size
        * bytes_per_value
    )
