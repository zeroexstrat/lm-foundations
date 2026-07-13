from __future__ import annotations

import pytest
import torch

from llm_from_scratch.research.scaling import (
    bootstrap_power_law_fit,
    chinchilla_optimal_parameters,
    chinchilla_optimal_tokens,
    fit_power_law,
    parameters_for_compute_budget,
    power_law_losses,
    tokens_for_compute_budget,
    training_compute_flops,
)


def test_fit_power_law_recovers_known_parameters() -> None:
    scales = torch.tensor([1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0, 32_000.0], dtype=torch.float64)
    true_coefficient = 18.0
    true_exponent = 0.37
    true_irreducible_loss = 1.45
    multiplicative_noise = torch.tensor([1.00, 1.02, 0.99, 1.01, 0.985, 1.015], dtype=torch.float64)
    losses = true_irreducible_loss + true_coefficient * torch.pow(scales, -true_exponent)
    losses = true_irreducible_loss + (losses - true_irreducible_loss) * multiplicative_noise

    fit = fit_power_law(scales, losses)

    assert fit.exponent == pytest.approx(true_exponent, abs=0.03)
    assert fit.coefficient == pytest.approx(true_coefficient, rel=0.15)
    assert fit.irreducible_loss == pytest.approx(true_irreducible_loss, abs=0.05)
    assert fit.predicted_losses.shape == scales.shape


def test_fit_power_law_recovers_near_floor_synthetic_sweep() -> None:
    scales = torch.tensor(
        [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
        dtype=torch.float64,
    )
    true_coefficient = 10.0
    true_exponent = 0.5
    true_irreducible_loss = 1.2
    losses = power_law_losses(
        scales,
        coefficient=true_coefficient,
        exponent=true_exponent,
        irreducible_loss=true_irreducible_loss,
    )

    fit = fit_power_law(scales, losses)

    assert fit.exponent == pytest.approx(true_exponent, abs=0.02)
    assert fit.coefficient == pytest.approx(true_coefficient, rel=0.1)
    assert fit.irreducible_loss == pytest.approx(true_irreducible_loss, abs=2e-2)


def test_bootstrap_power_law_fit_returns_ordered_confidence_intervals() -> None:
    scales = torch.tensor([768.0, 1_536.0, 3_072.0, 6_144.0, 12_288.0, 24_576.0, 49_152.0], dtype=torch.float64)
    true_exponent = 0.29
    true_coefficient = 11.0
    true_irreducible_loss = 1.1
    multiplicative_noise = torch.tensor([1.01, 0.99, 1.02, 0.985, 1.015, 0.995, 1.005], dtype=torch.float64)
    losses = true_irreducible_loss + true_coefficient * torch.pow(scales, -true_exponent)
    losses = true_irreducible_loss + (losses - true_irreducible_loss) * multiplicative_noise

    summary = bootstrap_power_law_fit(scales, losses, num_resamples=200, seed=7)

    assert summary.exponent.lower < summary.exponent.median < summary.exponent.upper
    assert summary.coefficient.lower < summary.coefficient.median < summary.coefficient.upper
    assert summary.irreducible_loss.lower < summary.irreducible_loss.median < summary.irreducible_loss.upper
    assert summary.exponent.lower <= true_exponent <= summary.exponent.upper
    assert summary.samples.shape == (200, 3)


def test_compute_accounting_helpers_are_inverse_consistent() -> None:
    num_parameters = 70_000_000
    num_tokens = 1_400_000_000
    compute_budget = training_compute_flops(num_parameters, num_tokens)

    assert tokens_for_compute_budget(num_parameters, compute_budget) == pytest.approx(float(num_tokens))
    assert parameters_for_compute_budget(num_tokens, compute_budget) == pytest.approx(float(num_parameters))
    assert chinchilla_optimal_tokens(num_parameters) == pytest.approx(1_400_000_000.0)
    assert chinchilla_optimal_parameters(num_tokens) == pytest.approx(70_000_000.0)


def test_power_law_losses_supports_single_scale_evaluation() -> None:
    loss = power_law_losses(
        torch.tensor([1_048_576.0], dtype=torch.float64),
        coefficient=18.0,
        exponent=0.37,
        irreducible_loss=1.45,
    )

    assert loss.shape == (1,)
    assert float(loss.item()) > 1.45


@pytest.mark.parametrize(
    ("num_parameters", "num_tokens"),
    [
        (0, 128),
        (128, 0),
    ],
)
def test_training_compute_flops_rejects_nonpositive_arguments(
    num_parameters: int,
    num_tokens: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        training_compute_flops(num_parameters, num_tokens)
