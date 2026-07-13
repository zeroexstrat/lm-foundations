from __future__ import annotations

import torch


def _validate_length(length: int) -> None:
    if length <= 0:
        raise ValueError("length must be positive")


def _validate_vector(name: str, value: torch.Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def _validate_nonempty_1d(name: str, value: torch.Tensor) -> None:
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")


def _validate_scalar(name: str, value: torch.Tensor) -> None:
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")


def _validate_same_shape(*values: tuple[str, torch.Tensor]) -> None:
    base_name, base_value = values[0]
    for name, value in values[1:]:
        if value.shape != base_value.shape:
            raise ValueError(f"{name} must have shape {tuple(base_value.shape)} like {base_name}")


def _validate_zero_initial_state(name: str, value: torch.Tensor | float | None) -> None:
    if value is None:
        return

    initial_state = torch.as_tensor(value)
    if torch.any(initial_state != 0):
        raise ValueError(f"{name} only supports zero initial state")


def _validate_zero_scalar_initial_state(name: str, value: torch.Tensor | float | None) -> None:
    if value is None:
        return

    initial_state = torch.as_tensor(value)
    if initial_state.ndim != 0:
        raise ValueError(f"{name} initial_state must be scalar")
    if torch.any(initial_state != 0):
        raise ValueError(f"{name} only supports zero initial state")


def scalar_linear_recurrence(
    inputs: torch.Tensor,
    *,
    transition: torch.Tensor | float,
    input_gain: torch.Tensor | float = 1.0,
    output_gain: torch.Tensor | float = 1.0,
    initial_state: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 1:
        raise ValueError("inputs must be a 1D tensor")
    _validate_nonempty_1d("inputs", inputs)

    transition_tensor = torch.as_tensor(transition, dtype=inputs.dtype, device=inputs.device)
    input_gain_tensor = torch.as_tensor(input_gain, dtype=inputs.dtype, device=inputs.device)
    output_gain_tensor = torch.as_tensor(output_gain, dtype=inputs.dtype, device=inputs.device)
    _validate_scalar("transition", transition_tensor)
    _validate_scalar("input_gain", input_gain_tensor)
    _validate_scalar("output_gain", output_gain_tensor)
    state = torch.as_tensor(initial_state, dtype=inputs.dtype, device=inputs.device)
    if state.ndim != 0:
        raise ValueError("initial_state must be scalar")

    states: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for input_t in inputs:
        state = transition_tensor * state + input_gain_tensor * input_t
        states.append(state)
        outputs.append(output_gain_tensor * state)
    return torch.stack(states), torch.stack(outputs)


def selective_sigmoid_gates(
    signals: torch.Tensor,
    *,
    weight: torch.Tensor | float,
    bias: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    if signals.ndim != 1:
        raise ValueError("signals must be a 1D tensor")
    _validate_nonempty_1d("signals", signals)

    weight_tensor = torch.as_tensor(weight, dtype=signals.dtype, device=signals.device)
    bias_tensor = torch.as_tensor(bias, dtype=signals.dtype, device=signals.device)
    _validate_scalar("weight", weight_tensor)
    _validate_scalar("bias", bias_tensor)
    return torch.sigmoid(weight_tensor * signals + bias_tensor)


def selective_scalar_linear_recurrence(
    inputs: torch.Tensor,
    *,
    gates: torch.Tensor,
    transition: torch.Tensor | float = 1.0,
    input_gain: torch.Tensor | float = 1.0,
    output_gain: torch.Tensor | float = 1.0,
    initial_state: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 1:
        raise ValueError("inputs must be a 1D tensor")
    _validate_nonempty_1d("inputs", inputs)

    gates_tensor = torch.as_tensor(gates, dtype=inputs.dtype, device=inputs.device)
    transition_tensor = torch.as_tensor(transition, dtype=inputs.dtype, device=inputs.device)
    input_gain_tensor = torch.as_tensor(input_gain, dtype=inputs.dtype, device=inputs.device)
    output_gain_tensor = torch.as_tensor(output_gain, dtype=inputs.dtype, device=inputs.device)

    _validate_vector("gates", gates_tensor)
    _validate_same_shape(("inputs", inputs), ("gates", gates_tensor))
    if torch.any((gates_tensor < 0) | (gates_tensor > 1)):
        raise ValueError("gates must lie in [0, 1]")

    _validate_scalar("transition", transition_tensor)
    _validate_scalar("input_gain", input_gain_tensor)
    _validate_scalar("output_gain", output_gain_tensor)
    state = torch.as_tensor(initial_state, dtype=inputs.dtype, device=inputs.device)
    if state.ndim != 0:
        raise ValueError("initial_state must be scalar")

    states: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for input_t, gate_t in zip(inputs, gates_tensor):
        state = gate_t * transition_tensor * state + (1.0 - gate_t) * input_gain_tensor * input_t
        states.append(state)
        outputs.append(output_gain_tensor * state)
    return torch.stack(states), torch.stack(outputs)


def scalar_impulse_response(
    *,
    transition: torch.Tensor | float,
    input_gain: torch.Tensor | float = 1.0,
    output_gain: torch.Tensor | float = 1.0,
    length: int,
) -> torch.Tensor:
    _validate_length(length)

    transition_tensor = torch.as_tensor(transition)
    _validate_scalar("transition", transition_tensor)
    dtype = transition_tensor.dtype
    device = transition_tensor.device
    input_gain_tensor = torch.as_tensor(input_gain, dtype=dtype, device=device)
    output_gain_tensor = torch.as_tensor(output_gain, dtype=dtype, device=device)
    _validate_scalar("input_gain", input_gain_tensor)
    _validate_scalar("output_gain", output_gain_tensor)
    lags = torch.arange(length, device=device)
    return output_gain_tensor * input_gain_tensor * torch.pow(transition_tensor, lags)


def diagonal_linear_recurrence(
    inputs: torch.Tensor,
    *,
    transition: torch.Tensor,
    input_gain: torch.Tensor,
    output_gain: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 1:
        raise ValueError("inputs must be a 1D tensor")
    _validate_nonempty_1d("inputs", inputs)

    transition_tensor = torch.as_tensor(transition, dtype=inputs.dtype, device=inputs.device)
    input_gain_tensor = torch.as_tensor(input_gain, dtype=inputs.dtype, device=inputs.device)
    output_gain_tensor = torch.as_tensor(output_gain, dtype=inputs.dtype, device=inputs.device)

    _validate_vector("transition", transition_tensor)
    _validate_vector("input_gain", input_gain_tensor)
    _validate_vector("output_gain", output_gain_tensor)
    _validate_same_shape(
        ("transition", transition_tensor),
        ("input_gain", input_gain_tensor),
        ("output_gain", output_gain_tensor),
    )

    if initial_state is None:
        state = torch.zeros_like(transition_tensor)
    else:
        state = torch.as_tensor(initial_state, dtype=inputs.dtype, device=inputs.device)
        _validate_vector("initial_state", state)
        _validate_same_shape(("transition", transition_tensor), ("initial_state", state))

    states: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for input_t in inputs:
        state = transition_tensor * state + input_gain_tensor * input_t
        states.append(state)
        outputs.append(torch.sum(output_gain_tensor * state))
    return torch.stack(states), torch.stack(outputs)


def diagonal_impulse_response(
    *,
    transition: torch.Tensor,
    input_gain: torch.Tensor,
    output_gain: torch.Tensor,
    length: int,
) -> torch.Tensor:
    _validate_length(length)

    transition_tensor = torch.as_tensor(transition)
    input_gain_tensor = torch.as_tensor(
        input_gain,
        dtype=transition_tensor.dtype,
        device=transition_tensor.device,
    )
    output_gain_tensor = torch.as_tensor(
        output_gain,
        dtype=transition_tensor.dtype,
        device=transition_tensor.device,
    )
    _validate_vector("transition", transition_tensor)
    _validate_vector("input_gain", input_gain_tensor)
    _validate_vector("output_gain", output_gain_tensor)
    _validate_same_shape(
        ("transition", transition_tensor),
        ("input_gain", input_gain_tensor),
        ("output_gain", output_gain_tensor),
    )

    lags = torch.arange(length, device=transition_tensor.device).unsqueeze(1)
    powers = torch.pow(transition_tensor.unsqueeze(0), lags)
    return (output_gain_tensor * input_gain_tensor * powers).sum(dim=1)


def causal_convolution(inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 1 or kernel.ndim != 1:
        raise ValueError("inputs and kernel must be 1D tensors")
    _validate_nonempty_1d("inputs", inputs)
    if kernel.numel() < inputs.numel():
        raise ValueError("kernel must be at least as long as inputs")

    outputs: list[torch.Tensor] = []
    for timestep in range(inputs.numel()):
        outputs.append(torch.dot(kernel[: timestep + 1], inputs[: timestep + 1].flip(0)))
    return torch.stack(outputs)


def scalar_recurrence_convolution_outputs(
    inputs: torch.Tensor,
    *,
    transition: torch.Tensor | float,
    input_gain: torch.Tensor | float = 1.0,
    output_gain: torch.Tensor | float = 1.0,
    initial_state: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_zero_scalar_initial_state("scalar_recurrence_convolution_outputs", initial_state)
    _, recurrent_output = scalar_linear_recurrence(
        inputs,
        transition=transition,
        input_gain=input_gain,
        output_gain=output_gain,
        initial_state=initial_state,
    )
    convolution_output = causal_convolution(
        inputs,
        scalar_impulse_response(
            transition=transition,
            input_gain=input_gain,
            output_gain=output_gain,
            length=inputs.numel(),
        ).to(dtype=inputs.dtype, device=inputs.device),
    )
    return recurrent_output, convolution_output


def diagonal_recurrence_convolution_outputs(
    inputs: torch.Tensor,
    *,
    transition: torch.Tensor,
    input_gain: torch.Tensor,
    output_gain: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_zero_initial_state("diagonal_recurrence_convolution_outputs", initial_state)
    _, recurrent_output = diagonal_linear_recurrence(
        inputs,
        transition=transition,
        input_gain=input_gain,
        output_gain=output_gain,
        initial_state=initial_state,
    )
    convolution_output = causal_convolution(
        inputs,
        diagonal_impulse_response(
            transition=transition,
            input_gain=input_gain,
            output_gain=output_gain,
            length=inputs.numel(),
        ).to(dtype=inputs.dtype, device=inputs.device),
    )
    return recurrent_output, convolution_output


def discretize_diagonal_state_space(
    continuous_transition: torch.Tensor,
    continuous_input: torch.Tensor,
    *,
    step: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    transition_tensor = torch.as_tensor(continuous_transition)
    input_tensor = torch.as_tensor(
        continuous_input,
        dtype=transition_tensor.dtype,
        device=transition_tensor.device,
    )
    step_tensor = torch.as_tensor(step, dtype=transition_tensor.dtype, device=transition_tensor.device)

    _validate_vector("continuous_transition", transition_tensor)
    _validate_vector("continuous_input", input_tensor)
    _validate_same_shape(
        ("continuous_transition", transition_tensor),
        ("continuous_input", input_tensor),
    )
    if step_tensor.ndim != 0:
        raise ValueError("step must be a scalar")
    if step_tensor <= 0:
        raise ValueError("step must be positive")

    scaled_transition = transition_tensor * step_tensor
    discrete_transition = torch.exp(scaled_transition)
    integral = torch.where(
        transition_tensor.abs() > 1e-12,
        torch.expm1(scaled_transition) / transition_tensor,
        torch.ones_like(transition_tensor) * step_tensor,
    )
    discrete_input = integral * input_tensor
    return discrete_transition, discrete_input
