from __future__ import annotations

import torch


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_square_bool_mask(mask: torch.Tensor) -> None:
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("mask must be a square 2D tensor")


def local_causal_mask(
    length: int,
    window_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    _validate_positive_int("length", length)
    _validate_positive_int("window_size", window_size)

    positions = torch.arange(length, device=device)
    offsets = positions[:, None] - positions[None, :]
    return (offsets >= 0) & (offsets < window_size)


def block_causal_mask(
    length: int,
    block_size: int,
    *,
    num_previous_blocks: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    _validate_positive_int("length", length)
    _validate_positive_int("block_size", block_size)
    if num_previous_blocks < 0:
        raise ValueError("num_previous_blocks must be nonnegative")

    positions = torch.arange(length, device=device)
    query_blocks = torch.div(positions[:, None], block_size, rounding_mode="floor")
    key_blocks = torch.div(positions[None, :], block_size, rounding_mode="floor")
    block_distance = query_blocks - key_blocks
    causal = positions[:, None] >= positions[None, :]
    return causal & (block_distance >= 0) & (block_distance <= num_previous_blocks)


def dilated_causal_mask(
    length: int,
    dilation: int,
    window_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    _validate_positive_int("length", length)
    _validate_positive_int("dilation", dilation)
    _validate_positive_int("window_size", window_size)

    positions = torch.arange(length, device=device)
    offsets = positions[:, None] - positions[None, :]
    return (offsets >= 0) & (offsets % dilation == 0) & (offsets // dilation < window_size)


def global_causal_mask(
    length: int,
    global_indices: list[int] | tuple[int, ...],
    *,
    local_window: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    _validate_positive_int("length", length)
    _validate_positive_int("local_window", local_window)

    base = local_causal_mask(length, local_window, device=device)
    if not global_indices:
        return base

    global_positions = torch.zeros(length, dtype=torch.bool, device=device)
    for index in global_indices:
        if index < 0 or index >= length:
            raise ValueError("global index out of range")
        global_positions[index] = True

    positions = torch.arange(length, device=device)
    causal = positions[:, None] >= positions[None, :]
    attends_to_global_keys = causal & global_positions.unsqueeze(0)
    global_queries = causal & global_positions.unsqueeze(1)
    return base | attends_to_global_keys | global_queries


def random_causal_mask(
    length: int,
    num_random: int,
    *,
    seed: int,
    local_window: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    _validate_positive_int("length", length)
    if num_random < 0:
        raise ValueError("num_random must be nonnegative")
    _validate_positive_int("local_window", local_window)

    mask = local_causal_mask(length, local_window, device=torch.device("cpu"))
    if num_random == 0:
        return mask.to(device) if device is not None else mask

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for query_index in range(length):
        eligible = torch.nonzero(~mask[query_index, :query_index], as_tuple=False).flatten()
        if eligible.numel() == 0:
            continue
        sample_size = min(num_random, eligible.numel())
        chosen = eligible[torch.randperm(eligible.numel(), generator=generator)[:sample_size]]
        mask[query_index, chosen] = True

    return mask.to(device) if device is not None else mask


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    if not masks:
        raise ValueError("at least one mask is required")

    reference_shape = masks[0].shape
    combined = masks[0].clone()
    _validate_square_bool_mask(combined)
    for mask in masks[1:]:
        _validate_square_bool_mask(mask)
        if mask.shape != reference_shape:
            raise ValueError("all masks must share the same shape")
        combined |= mask
    return combined


def _boolean_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.unsqueeze(-1) & right.unsqueeze(-3)).any(dim=-2)


def stacked_reachability(mask: torch.Tensor, layers: int) -> torch.Tensor:
    _validate_square_bool_mask(mask)
    _validate_positive_int("layers", layers)

    reach = mask.clone()
    for _ in range(1, layers):
        reach = _boolean_matmul(reach, mask)
    return reach


def receptive_field_sizes(mask: torch.Tensor, max_layers: int) -> torch.Tensor:
    _validate_square_bool_mask(mask)
    _validate_positive_int("max_layers", max_layers)

    reach = mask.clone()
    counts: list[torch.Tensor] = []
    for _ in range(max_layers):
        counts.append(reach.sum(dim=-1))
        reach = _boolean_matmul(reach, mask)
    return torch.stack(counts)


def mask_density(mask: torch.Tensor) -> float:
    _validate_square_bool_mask(mask)
    return float(mask.to(dtype=torch.float32).mean().item())
