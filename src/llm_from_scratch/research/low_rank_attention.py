from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SVDDiagnostics:
    singular_values: torch.Tensor
    cumulative_energy: torch.Tensor


def _validate_rank(rank: int) -> None:
    if rank <= 0:
        raise ValueError("rank must be positive")


def _validate_square_matrix(matrix: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D tensor")


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must have at least 2 dimensions")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share leading batch dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("key/value sequence length must match")


def truncated_svd_approximation(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor")
    _validate_rank(rank)

    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    kept = min(rank, singular_values.shape[0])
    return (u[:, :kept] * singular_values[:kept]) @ vh[:kept, :]


def svd_diagnostics(matrix: torch.Tensor) -> SVDDiagnostics:
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D tensor")

    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    cumulative_energy = energy.cumsum(dim=0) / energy.sum().clamp_min(torch.finfo(energy.dtype).tiny)
    return SVDDiagnostics(
        singular_values=singular_values,
        cumulative_energy=cumulative_energy,
    )


def frobenius_relative_error(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must share the same shape")

    numerator = torch.linalg.vector_norm((reference - approximation).reshape(-1), ord=2)
    denominator = torch.linalg.vector_norm(reference.reshape(-1), ord=2).clamp_min(
        torch.finfo(reference.dtype).tiny
    )
    return float((numerator / denominator).item())


def make_mean_pool_projection(
    sequence_length: int,
    projected_length: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if projected_length <= 0:
        raise ValueError("projected_length must be positive")
    if projected_length > sequence_length:
        raise ValueError("projected_length cannot exceed sequence_length")

    projection = torch.zeros(projected_length, sequence_length, device=device, dtype=dtype)
    boundaries = torch.linspace(0, sequence_length, steps=projected_length + 1, dtype=torch.float64)
    starts = torch.floor(boundaries[:-1]).to(dtype=torch.int64)
    ends = torch.floor(boundaries[1:]).to(dtype=torch.int64)
    ends[-1] = sequence_length

    for row_index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        if end <= start:
            end = start + 1
        width = end - start
        projection[row_index, start:end] = 1.0 / width
    return projection


def projected_sequence_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    projection: torch.Tensor,
    value_projection: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_attention_inputs(q, k, v)
    if projection.ndim != 2:
        raise ValueError("projection must be a 2D tensor")
    if projection.shape[-1] != k.shape[-2]:
        raise ValueError("projection input dimension must match key sequence length")

    if value_projection is None:
        value_projection = projection
    if value_projection.ndim != 2:
        raise ValueError("value_projection must be a 2D tensor")
    if value_projection.shape[-1] != v.shape[-2]:
        raise ValueError("value_projection input dimension must match value sequence length")
    if value_projection.shape[0] != projection.shape[0]:
        raise ValueError("projection and value_projection must share the projected length")

    projected_k = torch.einsum("rt,...td->...rd", projection.to(device=k.device, dtype=k.dtype), k)
    projected_v = torch.einsum(
        "rt,...td->...rd",
        value_projection.to(device=v.device, dtype=v.dtype),
        v,
    )

    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = q @ projected_k.transpose(-2, -1) * scale
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    weights = torch.exp(shifted)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights @ projected_v, weights


def uniform_landmark_indices(sequence_length: int, num_landmarks: int) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_landmarks <= 0:
        raise ValueError("num_landmarks must be positive")
    if num_landmarks > sequence_length:
        raise ValueError("num_landmarks cannot exceed sequence_length")

    if num_landmarks == 1:
        return torch.tensor([sequence_length // 2], dtype=torch.int64)

    raw = torch.linspace(0, sequence_length - 1, steps=num_landmarks, dtype=torch.float64)
    indices: list[int] = []
    used: set[int] = set()
    for candidate in raw.round().to(dtype=torch.int64).tolist():
        if candidate not in used:
            indices.append(candidate)
            used.add(candidate)
            continue
        for fallback in range(sequence_length):
            if fallback not in used:
                indices.append(fallback)
                used.add(fallback)
                break
    indices.sort()
    return torch.tensor(indices, dtype=torch.int64)


def nystrom_matrix_approximation(
    matrix: torch.Tensor,
    *,
    num_landmarks: int,
    landmark_indices: torch.Tensor | None = None,
    ridge: float = 1e-6,
) -> torch.Tensor:
    _validate_square_matrix(matrix)
    if num_landmarks <= 0:
        raise ValueError("num_landmarks must be positive")
    if num_landmarks > matrix.shape[0]:
        raise ValueError("num_landmarks cannot exceed matrix size")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")

    if landmark_indices is None:
        landmark_indices = uniform_landmark_indices(matrix.shape[0], num_landmarks)
    else:
        if landmark_indices.ndim != 1:
            raise ValueError("landmark_indices must be a 1D tensor")
        if landmark_indices.numel() != num_landmarks:
            raise ValueError("landmark_indices length must match num_landmarks")
        if landmark_indices.min().item() < 0 or landmark_indices.max().item() >= matrix.shape[0]:
            raise ValueError("landmark_indices must stay within matrix bounds")
        if torch.unique(landmark_indices).numel() != landmark_indices.numel():
            raise ValueError("landmark_indices must be unique")

    indices = landmark_indices.to(device=matrix.device, dtype=torch.int64)
    columns = matrix[:, indices]
    rows = matrix[indices, :]
    intersection = matrix[indices][:, indices]
    if ridge > 0.0 and num_landmarks < matrix.shape[0]:
        intersection = intersection + ridge * torch.eye(
            num_landmarks,
            device=matrix.device,
            dtype=matrix.dtype,
        )
    return columns @ torch.linalg.pinv(intersection) @ rows
