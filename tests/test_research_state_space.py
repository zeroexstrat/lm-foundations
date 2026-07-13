from __future__ import annotations

import math

import pytest
import torch

from llm_from_scratch.research.state_space import (
    causal_convolution,
    diagonal_impulse_response,
    diagonal_linear_recurrence,
    diagonal_recurrence_convolution_outputs,
    discretize_diagonal_state_space,
    selective_scalar_linear_recurrence,
    selective_sigmoid_gates,
    scalar_impulse_response,
    scalar_linear_recurrence,
    scalar_recurrence_convolution_outputs,
)


def test_scalar_impulse_response_matches_closed_form() -> None:
    impulse = scalar_impulse_response(
        transition=torch.tensor(0.5, dtype=torch.float64),
        input_gain=torch.tensor(2.0, dtype=torch.float64),
        output_gain=torch.tensor(1.5, dtype=torch.float64),
        length=4,
    )

    expected = torch.tensor([3.0, 1.5, 0.75, 0.375], dtype=torch.float64)

    assert torch.allclose(impulse, expected, atol=1e-12, rtol=1e-12)


def test_scalar_recurrence_matches_convolution_and_states() -> None:
    inputs = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
    states, outputs = scalar_linear_recurrence(
        inputs,
        transition=torch.tensor(0.5, dtype=torch.float64),
        input_gain=torch.tensor(2.0, dtype=torch.float64),
        output_gain=torch.tensor(1.5, dtype=torch.float64),
    )
    recurrent, convolution = scalar_recurrence_convolution_outputs(
        inputs,
        transition=torch.tensor(0.5, dtype=torch.float64),
        input_gain=torch.tensor(2.0, dtype=torch.float64),
        output_gain=torch.tensor(1.5, dtype=torch.float64),
    )

    expected_states = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    expected_outputs = torch.tensor([3.0, -1.5, 0.75], dtype=torch.float64)

    assert torch.allclose(states, expected_states, atol=1e-12, rtol=1e-12)
    assert torch.allclose(outputs, expected_outputs, atol=1e-12, rtol=1e-12)
    assert torch.allclose(recurrent, convolution, atol=1e-12, rtol=1e-12)
    assert torch.allclose(recurrent, expected_outputs, atol=1e-12, rtol=1e-12)


def test_diagonal_recurrence_matches_convolution() -> None:
    inputs = torch.tensor([1.0, 0.0, -2.0, 1.5], dtype=torch.float64)
    transition = torch.tensor([0.9, 0.5, -0.25], dtype=torch.float64)
    input_gain = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    output_gain = torch.tensor([0.2, 1.0, -1.5], dtype=torch.float64)

    states, outputs = diagonal_linear_recurrence(
        inputs,
        transition=transition,
        input_gain=input_gain,
        output_gain=output_gain,
    )
    recurrent, convolution = diagonal_recurrence_convolution_outputs(
        inputs,
        transition=transition,
        input_gain=input_gain,
        output_gain=output_gain,
    )
    impulse = diagonal_impulse_response(
        transition=transition,
        input_gain=input_gain,
        output_gain=output_gain,
        length=inputs.numel(),
    )

    expected_first_state = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    expected_impulse = torch.tensor(
        [
            -3.3,
            0.68,
            -0.1505,
            0.130175,
        ],
        dtype=torch.float64,
    )

    assert states.shape == (inputs.numel(), transition.numel())
    assert torch.allclose(states[0], expected_first_state, atol=1e-12, rtol=1e-12)
    assert torch.allclose(impulse, expected_impulse, atol=1e-12, rtol=1e-12)
    assert torch.allclose(recurrent, convolution, atol=1e-12, rtol=1e-12)
    assert torch.allclose(recurrent, outputs, atol=1e-12, rtol=1e-12)


def test_discretize_diagonal_state_space_matches_closed_form_integral() -> None:
    continuous_transition = torch.tensor([0.0, -1.0, 0.3], dtype=torch.float64)
    continuous_input = torch.tensor([2.0, 3.0, -4.0], dtype=torch.float64)

    discrete_transition, discrete_input = discretize_diagonal_state_space(
        continuous_transition,
        continuous_input,
        step=torch.tensor(0.5, dtype=torch.float64),
    )

    expected_transition = torch.tensor(
        [1.0, math.exp(-0.5), math.exp(0.15)],
        dtype=torch.float64,
    )
    expected_input = torch.tensor(
        [
            1.0,
            3.0 * (1.0 - math.exp(-0.5)),
            -4.0 * (math.exp(0.15) - 1.0) / 0.3,
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(discrete_transition, expected_transition, atol=1e-12, rtol=1e-12)
    assert torch.allclose(discrete_input, expected_input, atol=1e-12, rtol=1e-12)


def test_selective_sigmoid_gates_are_bounded() -> None:
    gates = selective_sigmoid_gates(
        torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=torch.float64),
        weight=torch.tensor(2.0, dtype=torch.float64),
        bias=torch.tensor(-0.5, dtype=torch.float64),
    )

    assert gates.shape == (5,)
    assert torch.all(gates >= 0.0)
    assert torch.all(gates <= 1.0)


def test_selective_recurrence_matches_fixed_recurrence_for_constant_gates() -> None:
    inputs = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    gates = torch.full_like(inputs, 0.75)

    selective_states, selective_outputs = selective_scalar_linear_recurrence(
        inputs,
        gates=gates,
        transition=torch.tensor(0.8, dtype=torch.float64),
        input_gain=torch.tensor(1.6, dtype=torch.float64),
        output_gain=torch.tensor(1.2, dtype=torch.float64),
    )
    fixed_states, fixed_outputs = scalar_linear_recurrence(
        inputs,
        transition=torch.tensor(0.75 * 0.8, dtype=torch.float64),
        input_gain=torch.tensor((1.0 - 0.75) * 1.6, dtype=torch.float64),
        output_gain=torch.tensor(1.2, dtype=torch.float64),
    )

    assert torch.allclose(selective_states, fixed_states, atol=1e-12, rtol=1e-12)
    assert torch.allclose(selective_outputs, fixed_outputs, atol=1e-12, rtol=1e-12)


def test_scalar_recurrence_convolution_outputs_rejects_nonzero_initial_state() -> None:
    with pytest.raises(ValueError, match="zero initial state"):
        scalar_recurrence_convolution_outputs(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            transition=torch.tensor(0.5, dtype=torch.float64),
            initial_state=torch.tensor(1.0, dtype=torch.float64),
        )


def test_scalar_recurrence_convolution_outputs_rejects_vector_zero_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state must be scalar"):
        scalar_recurrence_convolution_outputs(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            transition=torch.tensor(0.5, dtype=torch.float64),
            initial_state=torch.tensor([0.0, 0.0], dtype=torch.float64),
        )


def test_scalar_linear_recurrence_rejects_vector_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state must be scalar"):
        scalar_linear_recurrence(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            transition=torch.tensor(0.5, dtype=torch.float64),
            initial_state=torch.tensor([0.0, 0.0], dtype=torch.float64),
        )


@pytest.mark.parametrize("argument", ["transition", "input_gain", "output_gain"])
def test_scalar_linear_recurrence_rejects_vector_parameters(argument: str) -> None:
    kwargs: dict[str, torch.Tensor] = {
        "transition": torch.tensor(0.5, dtype=torch.float64),
        "input_gain": torch.tensor(1.0, dtype=torch.float64),
        "output_gain": torch.tensor(1.0, dtype=torch.float64),
    }
    kwargs[argument] = torch.tensor([0.5, 0.25], dtype=torch.float64)

    with pytest.raises(ValueError, match=f"{argument} must be scalar"):
        scalar_linear_recurrence(torch.tensor([1.0, -1.0], dtype=torch.float64), **kwargs)


@pytest.mark.parametrize("argument", ["transition", "input_gain", "output_gain"])
def test_scalar_impulse_response_rejects_vector_parameters(argument: str) -> None:
    kwargs: dict[str, torch.Tensor | int] = {
        "transition": torch.tensor(0.5, dtype=torch.float64),
        "input_gain": torch.tensor(1.0, dtype=torch.float64),
        "output_gain": torch.tensor(1.0, dtype=torch.float64),
        "length": 3,
    }
    kwargs[argument] = torch.tensor([0.5, 0.25], dtype=torch.float64)

    with pytest.raises(ValueError, match=f"{argument} must be scalar"):
        scalar_impulse_response(**kwargs)


def test_scalar_recurrence_convolution_outputs_rejects_vector_transition() -> None:
    with pytest.raises(ValueError, match="transition must be scalar"):
        scalar_recurrence_convolution_outputs(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            transition=torch.tensor([0.5, 0.25], dtype=torch.float64),
        )


def test_diagonal_recurrence_convolution_outputs_rejects_nonzero_initial_state() -> None:
    with pytest.raises(ValueError, match="zero initial state"):
        diagonal_recurrence_convolution_outputs(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            transition=torch.tensor([0.5, 0.25], dtype=torch.float64),
            input_gain=torch.tensor([1.0, -2.0], dtype=torch.float64),
            output_gain=torch.tensor([0.25, 0.5], dtype=torch.float64),
            initial_state=torch.tensor([0.0, 1.0], dtype=torch.float64),
        )


def test_selective_scalar_linear_recurrence_rejects_out_of_range_gates() -> None:
    with pytest.raises(ValueError, match="gates must lie in \\[0, 1\\]"):
        selective_scalar_linear_recurrence(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            gates=torch.tensor([0.5, 1.25], dtype=torch.float64),
            transition=torch.tensor(0.8, dtype=torch.float64),
        )


def test_scalar_linear_recurrence_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="inputs must be non-empty"):
        scalar_linear_recurrence(
            torch.tensor([], dtype=torch.float64),
            transition=torch.tensor(0.5, dtype=torch.float64),
        )


def test_diagonal_linear_recurrence_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="inputs must be non-empty"):
        diagonal_linear_recurrence(
            torch.tensor([], dtype=torch.float64),
            transition=torch.tensor([0.5, 0.25], dtype=torch.float64),
            input_gain=torch.tensor([1.0, -2.0], dtype=torch.float64),
            output_gain=torch.tensor([0.25, 0.5], dtype=torch.float64),
        )


def test_causal_convolution_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="inputs must be non-empty"):
        causal_convolution(
            torch.tensor([], dtype=torch.float64),
            torch.tensor([], dtype=torch.float64),
        )
