from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class AdamWReferenceState:
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


@dataclass(frozen=True)
class GradientAccumulationStats:
    micro_batch_size: int
    accumulation_steps: int
    data_parallel_size: int
    effective_batch_size: int
    tokens_per_update: int | None
    optimizer_steps_per_epoch: int | None
    remainder_micro_batches: int | None


@dataclass(frozen=True)
class OptimizerMemoryEstimate:
    parameter_bytes: int
    gradient_bytes: int
    first_moment_bytes: int
    second_moment_bytes: int
    master_weights_bytes: int
    total_bytes: int


def _validate_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_same_shape(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{name} must match parameter shape")


def _validate_betas(betas: tuple[float, float]) -> tuple[float, float]:
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must lie in [0, 1)")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must lie in [0, 1)")
    return float(beta1), float(beta2)


def adamw_reference_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    *,
    state: AdamWReferenceState | None = None,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
) -> tuple[torch.Tensor, AdamWReferenceState]:
    if parameter.ndim == 0:
        parameter_tensor = parameter.reshape(1).clone()
        gradient_tensor = gradient.reshape(1).clone()
        scalar_input = True
    else:
        parameter_tensor = parameter.clone()
        gradient_tensor = gradient.clone()
        scalar_input = False

    _validate_same_shape("gradient", parameter_tensor, gradient_tensor)
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    beta1, beta2 = _validate_betas(betas)

    if state is None:
        step = 0
        exp_avg = torch.zeros_like(parameter_tensor)
        exp_avg_sq = torch.zeros_like(parameter_tensor)
    else:
        _validate_same_shape("state.exp_avg", parameter_tensor, state.exp_avg)
        _validate_same_shape("state.exp_avg_sq", parameter_tensor, state.exp_avg_sq)
        step = state.step
        exp_avg = state.exp_avg.clone()
        exp_avg_sq = state.exp_avg_sq.clone()

    step += 1
    parameter_tensor.mul_(1.0 - lr * weight_decay)
    exp_avg.lerp_(gradient_tensor, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient_tensor, gradient_tensor, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    step_size = lr / bias_correction1
    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
    parameter_tensor.addcdiv_(exp_avg, denom, value=-step_size)

    updated_state = AdamWReferenceState(
        step=step,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
    )
    if scalar_input:
        return parameter_tensor.reshape(()), updated_state
    return parameter_tensor, updated_state


def accumulate_mean_gradients(
    gradients: Sequence[torch.Tensor],
    *,
    sample_counts: Sequence[int] | None = None,
) -> torch.Tensor:
    if not gradients:
        raise ValueError("gradients must be non-empty")

    reference = gradients[0]
    total = torch.zeros_like(reference)
    total_weight = 0

    if sample_counts is not None and len(sample_counts) != len(gradients):
        raise ValueError("sample_counts must match gradients length")

    for index, gradient in enumerate(gradients):
        _validate_same_shape("all gradients", reference, gradient)
        weight = 1 if sample_counts is None else _validate_positive_int("sample_counts", int(sample_counts[index]))
        total.add_(gradient, alpha=float(weight))
        total_weight += weight

    return total / total_weight


def gradient_accumulation_stats(
    *,
    micro_batch_size: int,
    accumulation_steps: int,
    data_parallel_size: int = 1,
    sequence_length: int | None = None,
    micro_batches_per_epoch: int | None = None,
) -> GradientAccumulationStats:
    micro_batch_size = _validate_positive_int("micro_batch_size", micro_batch_size)
    accumulation_steps = _validate_positive_int("accumulation_steps", accumulation_steps)
    data_parallel_size = _validate_positive_int("data_parallel_size", data_parallel_size)

    if sequence_length is not None:
        sequence_length = _validate_positive_int("sequence_length", sequence_length)
    if micro_batches_per_epoch is not None:
        micro_batches_per_epoch = _validate_positive_int("micro_batches_per_epoch", micro_batches_per_epoch)

    effective_batch_size = micro_batch_size * accumulation_steps * data_parallel_size
    tokens_per_update = None if sequence_length is None else effective_batch_size * sequence_length
    optimizer_steps_per_epoch = None
    remainder_micro_batches = None
    if micro_batches_per_epoch is not None:
        remainder_micro_batches = micro_batches_per_epoch % accumulation_steps
        optimizer_steps_per_epoch = micro_batches_per_epoch // accumulation_steps
        if remainder_micro_batches:
            optimizer_steps_per_epoch += 1

    return GradientAccumulationStats(
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        data_parallel_size=data_parallel_size,
        effective_batch_size=effective_batch_size,
        tokens_per_update=tokens_per_update,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        remainder_micro_batches=remainder_micro_batches,
    )


def estimate_optimizer_memory(
    *,
    num_parameters: int,
    parameter_bytes: int,
    gradient_bytes: int | None = None,
    state_bytes: int | None = None,
    use_master_weights: bool = False,
    master_weight_bytes: int = 4,
) -> OptimizerMemoryEstimate:
    """Estimate tracked training-state bytes for AdamW-style training tensors."""
    num_parameters = _validate_positive_int("num_parameters", num_parameters)
    parameter_bytes = _validate_positive_int("parameter_bytes", parameter_bytes)
    gradient_bytes = parameter_bytes if gradient_bytes is None else _validate_positive_int("gradient_bytes", gradient_bytes)
    state_bytes = parameter_bytes if state_bytes is None else _validate_positive_int("state_bytes", state_bytes)
    if use_master_weights:
        master_weight_bytes = _validate_positive_int("master_weight_bytes", master_weight_bytes)
    else:
        master_weight_bytes = 0

    parameter_total = num_parameters * parameter_bytes
    gradient_total = num_parameters * gradient_bytes
    first_moment_total = num_parameters * state_bytes
    second_moment_total = num_parameters * state_bytes
    master_weight_total = num_parameters * master_weight_bytes
    total = parameter_total + gradient_total + first_moment_total + second_moment_total + master_weight_total

    return OptimizerMemoryEstimate(
        parameter_bytes=parameter_total,
        gradient_bytes=gradient_total,
        first_moment_bytes=first_moment_total,
        second_moment_bytes=second_moment_total,
        master_weights_bytes=master_weight_total,
        total_bytes=total,
    )
