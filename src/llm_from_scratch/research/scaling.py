from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PowerLawFit:
    coefficient: float
    exponent: float
    irreducible_loss: float
    predicted_losses: torch.Tensor
    log_mse: float


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    median: float
    upper: float


@dataclass(frozen=True)
class BootstrapFitSummary:
    coefficient: ConfidenceInterval
    exponent: ConfidenceInterval
    irreducible_loss: ConfidenceInterval
    samples: torch.Tensor


def _validate_positive_1d(name: str, value: torch.Tensor, *, min_points: int = 1) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    if value.numel() < min_points:
        raise ValueError(f"{name} must contain at least {min_points} points")
    if torch.any(value <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return value.to(dtype=torch.float64)


def _validate_matching_inputs(scales: torch.Tensor, losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales_tensor = _validate_positive_1d("scales", torch.as_tensor(scales), min_points=3)
    losses_tensor = _validate_positive_1d("losses", torch.as_tensor(losses), min_points=3)
    if scales_tensor.shape != losses_tensor.shape:
        raise ValueError("scales and losses must have the same shape")
    if torch.unique(scales_tensor).numel() < 2:
        raise ValueError("scales must contain at least two distinct values")
    return scales_tensor, losses_tensor


def power_law_losses(
    scales: torch.Tensor,
    *,
    coefficient: float,
    exponent: float,
    irreducible_loss: float = 0.0,
) -> torch.Tensor:
    scales_tensor = _validate_positive_1d("scales", torch.as_tensor(scales))
    if coefficient <= 0:
        raise ValueError("coefficient must be positive")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if irreducible_loss < 0:
        raise ValueError("irreducible_loss must be non-negative")
    return irreducible_loss + coefficient * torch.pow(scales_tensor, -exponent)


def _fit_with_fixed_floor(
    scales: torch.Tensor,
    losses: torch.Tensor,
    *,
    irreducible_loss: float,
) -> PowerLawFit:
    floor = float(irreducible_loss)
    if floor < 0:
        raise ValueError("irreducible_loss must be non-negative")

    adjusted = losses - floor
    if torch.any(adjusted <= 0):
        raise ValueError("irreducible_loss must be smaller than every observed loss")

    log_scales = torch.log(scales)
    log_adjusted = torch.log(adjusted)
    centered_scales = log_scales - log_scales.mean()
    denominator = torch.dot(centered_scales, centered_scales)
    if float(denominator.item()) == 0.0:
        raise ValueError("scales must vary to fit a power law")

    slope = torch.dot(centered_scales, log_adjusted - log_adjusted.mean()) / denominator
    intercept = log_adjusted.mean() - slope * log_scales.mean()
    exponent = float((-slope).item())
    coefficient = float(torch.exp(intercept).item())
    predicted_losses = floor + coefficient * torch.pow(scales, -exponent)
    residual = log_adjusted - (intercept + slope * log_scales)
    log_mse = float(torch.mean(residual.square()).item())
    return PowerLawFit(
        coefficient=coefficient,
        exponent=exponent,
        irreducible_loss=floor,
        predicted_losses=predicted_losses,
        log_mse=log_mse,
    )


def _candidate_floors(
    losses: torch.Tensor,
    *,
    floor_grid_size: int,
    max_floor_fraction: float,
) -> torch.Tensor:
    if floor_grid_size <= 1:
        raise ValueError("floor_grid_size must be greater than 1")
    if not 0.0 <= max_floor_fraction <= 1.0:
        raise ValueError("max_floor_fraction must lie in [0, 1]")

    min_loss = losses.min()
    admissible_upper = torch.nextafter(min_loss, torch.zeros_like(min_loss))
    max_floor = admissible_upper * max_floor_fraction
    if float(max_floor.item()) == 0.0:
        return torch.zeros(1, dtype=losses.dtype, device=losses.device)

    gap_lower = min_loss - max_floor
    gap_upper = min_loss
    log_gaps = torch.logspace(
        math.log10(float(gap_lower.item())),
        math.log10(float(gap_upper.item())),
        steps=floor_grid_size,
        dtype=losses.dtype,
        device=losses.device,
    )
    candidate_floors = torch.sort(torch.unique(torch.clamp(min_loss - log_gaps, min=0.0))).values
    return torch.cat(
        (
            torch.zeros(1, dtype=losses.dtype, device=losses.device),
            candidate_floors,
        )
    ).unique(sorted=True)


def fit_power_law(
    scales: torch.Tensor,
    losses: torch.Tensor,
    *,
    irreducible_loss: float | None = None,
    floor_grid_size: int = 256,
    max_floor_fraction: float = 1.0,
) -> PowerLawFit:
    scales_tensor, losses_tensor = _validate_matching_inputs(scales, losses)

    if irreducible_loss is not None:
        return _fit_with_fixed_floor(
            scales_tensor,
            losses_tensor,
            irreducible_loss=irreducible_loss,
        )

    candidate_floors = _candidate_floors(
        losses_tensor,
        floor_grid_size=floor_grid_size,
        max_floor_fraction=max_floor_fraction,
    )

    best_fit: PowerLawFit | None = None
    for candidate in candidate_floors.tolist():
        fit = _fit_with_fixed_floor(
            scales_tensor,
            losses_tensor,
            irreducible_loss=float(candidate),
        )
        if best_fit is None or fit.log_mse < best_fit.log_mse:
            best_fit = fit

    assert best_fit is not None
    return best_fit


def _quantile_interval(values: torch.Tensor, confidence: float) -> ConfidenceInterval:
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        lower=float(torch.quantile(values, alpha).item()),
        median=float(torch.quantile(values, 0.5).item()),
        upper=float(torch.quantile(values, 1.0 - alpha).item()),
    )


def bootstrap_power_law_fit(
    scales: torch.Tensor,
    losses: torch.Tensor,
    *,
    num_resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 0,
    irreducible_loss: float | None = None,
    floor_grid_size: int = 256,
    max_floor_fraction: float = 1.0,
) -> BootstrapFitSummary:
    scales_tensor, losses_tensor = _validate_matching_inputs(scales, losses)
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    generator = torch.Generator(device=scales_tensor.device)
    generator.manual_seed(seed)
    draws: list[torch.Tensor] = []
    attempts = 0
    max_attempts = num_resamples * 20

    while len(draws) < num_resamples:
        if attempts >= max_attempts:
            raise RuntimeError("bootstrap_power_law_fit could not draw enough valid resamples")
        attempts += 1
        indices = torch.randint(
            low=0,
            high=scales_tensor.numel(),
            size=(scales_tensor.numel(),),
            generator=generator,
            device=scales_tensor.device,
        )
        sampled_scales = scales_tensor[indices]
        if torch.unique(sampled_scales).numel() < 2:
            continue
        sampled_losses = losses_tensor[indices]
        fit = fit_power_law(
            sampled_scales,
            sampled_losses,
            irreducible_loss=irreducible_loss,
            floor_grid_size=floor_grid_size,
            max_floor_fraction=max_floor_fraction,
        )
        draws.append(
            torch.tensor(
                [fit.coefficient, fit.exponent, fit.irreducible_loss],
                dtype=torch.float64,
            )
        )

    samples = torch.stack(draws)
    return BootstrapFitSummary(
        coefficient=_quantile_interval(samples[:, 0], confidence),
        exponent=_quantile_interval(samples[:, 1], confidence),
        irreducible_loss=_quantile_interval(samples[:, 2], confidence),
        samples=samples,
    )


def _validate_positive_intlike(name: str, value: int | float) -> float:
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def training_compute_flops(
    num_parameters: int | float,
    num_tokens: int | float,
    *,
    flops_per_parameter_token: float = 6.0,
) -> float:
    parameters = _validate_positive_intlike("num_parameters", num_parameters)
    tokens = _validate_positive_intlike("num_tokens", num_tokens)
    factor = _validate_positive_intlike("flops_per_parameter_token", flops_per_parameter_token)
    return parameters * tokens * factor


def tokens_for_compute_budget(
    num_parameters: int | float,
    compute_budget_flops: int | float,
    *,
    flops_per_parameter_token: float = 6.0,
) -> float:
    parameters = _validate_positive_intlike("num_parameters", num_parameters)
    budget = _validate_positive_intlike("compute_budget_flops", compute_budget_flops)
    factor = _validate_positive_intlike("flops_per_parameter_token", flops_per_parameter_token)
    return budget / (parameters * factor)


def parameters_for_compute_budget(
    num_tokens: int | float,
    compute_budget_flops: int | float,
    *,
    flops_per_parameter_token: float = 6.0,
) -> float:
    tokens = _validate_positive_intlike("num_tokens", num_tokens)
    budget = _validate_positive_intlike("compute_budget_flops", compute_budget_flops)
    factor = _validate_positive_intlike("flops_per_parameter_token", flops_per_parameter_token)
    return budget / (tokens * factor)


def chinchilla_optimal_tokens(
    num_parameters: int | float,
    *,
    tokens_per_parameter: float = 20.0,
) -> float:
    parameters = _validate_positive_intlike("num_parameters", num_parameters)
    ratio = _validate_positive_intlike("tokens_per_parameter", tokens_per_parameter)
    return parameters * ratio


def chinchilla_optimal_parameters(
    num_tokens: int | float,
    *,
    tokens_per_parameter: float = 20.0,
) -> float:
    tokens = _validate_positive_intlike("num_tokens", num_tokens)
    ratio = _validate_positive_intlike("tokens_per_parameter", tokens_per_parameter)
    return tokens / ratio
