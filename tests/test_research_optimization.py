from __future__ import annotations

import pytest
import torch

from llm_from_scratch.research.optimization import (
    accumulate_mean_gradients,
    adamw_reference_step,
    estimate_optimizer_memory,
    gradient_accumulation_stats,
)


def test_adamw_reference_step_matches_torch_adamw_for_two_updates() -> None:
    parameter = torch.tensor([1.25, -0.75, 0.5], dtype=torch.float64)
    gradients = [
        torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64),
        torch.tensor([-0.3, 0.15, 0.4], dtype=torch.float64),
    ]
    kwargs = {
        "lr": 1e-2,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }

    actual_parameter = torch.nn.Parameter(parameter.clone())
    optimizer = torch.optim.AdamW([actual_parameter], **kwargs)

    reference_parameter = parameter.clone()
    reference_state = None

    for gradient in gradients:
        optimizer.zero_grad()
        actual_parameter.grad = gradient.clone()
        optimizer.step()

        reference_parameter, reference_state = adamw_reference_step(
            reference_parameter,
            gradient,
            state=reference_state,
            **kwargs,
        )

    assert reference_state is not None

    torch_state = optimizer.state[actual_parameter]

    assert torch.allclose(reference_parameter, actual_parameter.detach(), atol=1e-12, rtol=1e-12)
    assert torch.allclose(reference_state.exp_avg, torch_state["exp_avg"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(reference_state.exp_avg_sq, torch_state["exp_avg_sq"], atol=1e-12, rtol=1e-12)
    assert reference_state.step == int(torch_state["step"].item())


def test_accumulate_mean_gradients_matches_full_batch_gradient() -> None:
    features = torch.tensor(
        [
            [1.0, -1.0],
            [0.5, 0.25],
            [-0.75, 1.5],
            [1.25, 0.5],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0.1], [0.2], [-0.3], [0.4]], dtype=torch.float64)
    weight = torch.tensor([[0.4, -0.2]], dtype=torch.float64, requires_grad=True)

    full_prediction = features @ weight.t()
    full_loss = torch.mean((full_prediction - targets) ** 2)
    full_gradient = torch.autograd.grad(full_loss, weight)[0]

    micro_gradients: list[torch.Tensor] = []
    for micro_features, micro_targets in zip(features.chunk(2), targets.chunk(2), strict=True):
        micro_weight = weight.detach().clone().requires_grad_(True)
        micro_prediction = micro_features @ micro_weight.t()
        micro_loss = torch.mean((micro_prediction - micro_targets) ** 2)
        micro_gradients.append(torch.autograd.grad(micro_loss, micro_weight)[0])

    accumulated = accumulate_mean_gradients(micro_gradients)

    assert torch.allclose(accumulated, full_gradient, atol=1e-12, rtol=1e-12)


def test_accumulate_mean_gradients_with_sample_counts_matches_full_batch_gradient() -> None:
    features = torch.tensor(
        [
            [1.0, -1.0],
            [0.5, 0.25],
            [-0.75, 1.5],
            [1.25, 0.5],
            [0.1, -0.4],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0.1], [0.2], [-0.3], [0.4], [0.6]], dtype=torch.float64)
    weight = torch.tensor([[0.4, -0.2]], dtype=torch.float64, requires_grad=True)

    full_prediction = features @ weight.t()
    full_loss = torch.mean((full_prediction - targets) ** 2)
    full_gradient = torch.autograd.grad(full_loss, weight)[0]

    micro_gradients: list[torch.Tensor] = []
    sample_counts: list[int] = []
    for micro_features, micro_targets in (
        (features[:2], targets[:2]),
        (features[2:3], targets[2:3]),
        (features[3:], targets[3:]),
    ):
        micro_weight = weight.detach().clone().requires_grad_(True)
        micro_prediction = micro_features @ micro_weight.t()
        micro_loss = torch.mean((micro_prediction - micro_targets) ** 2)
        micro_gradients.append(torch.autograd.grad(micro_loss, micro_weight)[0])
        sample_counts.append(len(micro_features))

    weighted = accumulate_mean_gradients(micro_gradients, sample_counts=sample_counts)
    unweighted = accumulate_mean_gradients(micro_gradients)

    assert torch.allclose(weighted, full_gradient, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(unweighted, full_gradient, atol=1e-12, rtol=1e-12)


def test_gradient_accumulation_stats_flushes_partial_accumulation_into_final_step() -> None:
    stats = gradient_accumulation_stats(
        micro_batch_size=4,
        accumulation_steps=8,
        data_parallel_size=2,
        sequence_length=128,
        micro_batches_per_epoch=21,
    )

    assert stats.effective_batch_size == 64
    assert stats.tokens_per_update == 8_192
    assert stats.optimizer_steps_per_epoch == 3
    assert stats.remainder_micro_batches == 5


def test_estimate_optimizer_memory_accounts_for_master_weights() -> None:
    estimate = estimate_optimizer_memory(
        num_parameters=1_000,
        parameter_bytes=2,
        gradient_bytes=2,
        state_bytes=4,
        use_master_weights=True,
        master_weight_bytes=4,
    )

    assert estimate.parameter_bytes == 2_000
    assert estimate.gradient_bytes == 2_000
    assert estimate.first_moment_bytes == 4_000
    assert estimate.second_moment_bytes == 4_000
    assert estimate.master_weights_bytes == 4_000
    assert estimate.total_bytes == 16_000


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("micro_batch_size", {"micro_batch_size": 0, "accumulation_steps": 4}),
        ("accumulation_steps", {"micro_batch_size": 4, "accumulation_steps": 0}),
        ("num_parameters", {"num_parameters": 0, "parameter_bytes": 2}),
    ],
)
def test_optimization_helpers_reject_nonpositive_arguments(name: str, kwargs: dict[str, int]) -> None:
    if name == "num_parameters":
        with pytest.raises(ValueError, match="positive"):
            estimate_optimizer_memory(**kwargs)
    else:
        with pytest.raises(ValueError, match="positive"):
            gradient_accumulation_stats(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": 0.0}, "lr must be positive"),
        ({"eps": 0.0}, "eps must be positive"),
        ({"betas": (1.0, 0.999)}, "beta1 must lie in \\[0, 1\\)"),
        ({"betas": (0.9, 1.0)}, "beta2 must lie in \\[0, 1\\)"),
    ],
)
def test_adamw_reference_step_rejects_invalid_hyperparameters(kwargs: dict[str, object], message: str) -> None:
    parameter = torch.tensor([1.0], dtype=torch.float64)
    gradient = torch.tensor([0.1], dtype=torch.float64)

    with pytest.raises(ValueError, match=message):
        adamw_reference_step(parameter, gradient, **kwargs)


def test_accumulate_mean_gradients_rejects_empty_gradients() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        accumulate_mean_gradients([])


def test_accumulate_mean_gradients_rejects_mismatched_sample_counts() -> None:
    gradients = [torch.tensor([1.0]), torch.tensor([2.0])]

    with pytest.raises(ValueError, match="sample_counts must match gradients length"):
        accumulate_mean_gradients(gradients, sample_counts=[2])
