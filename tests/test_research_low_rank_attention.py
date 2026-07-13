from __future__ import annotations

import torch

from llm_from_scratch.research.attention_math import exact_attention
from llm_from_scratch.research.low_rank_attention import (
    frobenius_relative_error,
    make_mean_pool_projection,
    nystrom_matrix_approximation,
    projected_sequence_attention,
    svd_diagnostics,
    truncated_svd_approximation,
)


def test_truncated_svd_reconstruction_error_decreases_with_rank() -> None:
    left = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    singular_values = torch.diag(torch.tensor([5.0, 2.0, 0.5], dtype=torch.float64))
    matrix = left @ singular_values @ right

    rank_one = truncated_svd_approximation(matrix, rank=1)
    rank_two = truncated_svd_approximation(matrix, rank=2)
    rank_three = truncated_svd_approximation(matrix, rank=3)

    error_one = frobenius_relative_error(matrix, rank_one)
    error_two = frobenius_relative_error(matrix, rank_two)
    error_three = frobenius_relative_error(matrix, rank_three)

    assert error_two < error_one
    assert error_three < error_two
    assert error_three < 1e-12


def test_projected_sequence_attention_matches_exact_attention_for_identity_projection() -> None:
    q = torch.tensor(
        [[[0.2, -0.4], [0.1, 0.3], [-0.5, 0.7]]],
        dtype=torch.float64,
    )
    k = torch.tensor(
        [[[0.6, -0.2], [-0.3, 0.1], [0.4, 0.5]]],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [[[1.0, 0.0], [0.5, 1.0], [-0.5, 0.25]]],
        dtype=torch.float64,
    )
    projection = torch.eye(q.shape[-2], dtype=torch.float64)

    exact_output, exact_weights = exact_attention(q, k, v)
    projected_output, projected_weights = projected_sequence_attention(
        q,
        k,
        v,
        projection=projection,
    )

    assert torch.allclose(projected_output, exact_output, atol=1e-10, rtol=1e-10)
    assert torch.allclose(projected_weights, exact_weights, atol=1e-10, rtol=1e-10)


def test_svd_diagnostics_reports_sorted_spectrum_and_normalized_energy() -> None:
    matrix = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))

    diagnostics = svd_diagnostics(matrix)

    assert torch.allclose(
        diagnostics.singular_values,
        torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.all(diagnostics.singular_values[:-1] >= diagnostics.singular_values[1:])
    assert torch.all(diagnostics.cumulative_energy[:-1] < diagnostics.cumulative_energy[1:])
    assert torch.allclose(
        diagnostics.cumulative_energy,
        torch.tensor([16.0 / 21.0, 20.0 / 21.0, 1.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert diagnostics.cumulative_energy[-1].item() == 1.0


def test_mean_pool_projection_computes_true_block_means() -> None:
    sequence = torch.tensor(
        [[[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]]],
        dtype=torch.float64,
    )
    projection = make_mean_pool_projection(sequence_length=6, projected_length=3, dtype=torch.float64)

    pooled = torch.einsum("rt,btd->brd", projection, sequence)

    assert torch.allclose(
        pooled,
        torch.tensor([[[2.0], [6.0], [10.0]]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_mean_pool_projection_preserves_smooth_sequences_better_with_more_slots() -> None:
    sequence = torch.tensor(
        [[[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]]],
        dtype=torch.float64,
    )

    coarse = make_mean_pool_projection(sequence_length=6, projected_length=2, dtype=torch.float64)
    medium = make_mean_pool_projection(sequence_length=6, projected_length=3, dtype=torch.float64)
    fine = make_mean_pool_projection(sequence_length=6, projected_length=6, dtype=torch.float64)

    def reconstruct(projection: torch.Tensor) -> torch.Tensor:
        compressed = torch.einsum("rt,btd->brd", projection, sequence)
        block_membership = projection.transpose(0, 1).ne(0.0).to(dtype=sequence.dtype)
        return torch.einsum("tr,brd->btd", block_membership, compressed)

    coarse_error = frobenius_relative_error(sequence, reconstruct(coarse))
    medium_error = frobenius_relative_error(sequence, reconstruct(medium))
    fine_error = frobenius_relative_error(sequence, reconstruct(fine))

    assert medium_error < coarse_error
    assert fine_error < medium_error
    assert fine_error < 1e-12


def test_nystrom_reconstruction_error_decreases_with_more_landmarks() -> None:
    positions = torch.linspace(0.0, 1.0, steps=8, dtype=torch.float64).unsqueeze(-1)
    distances = positions - positions.transpose(0, 1)
    matrix = torch.exp(-(distances.square()) / 0.08)

    two_landmarks = nystrom_matrix_approximation(matrix, num_landmarks=2)
    four_landmarks = nystrom_matrix_approximation(matrix, num_landmarks=4)
    eight_landmarks = nystrom_matrix_approximation(matrix, num_landmarks=8)

    error_two = frobenius_relative_error(matrix, two_landmarks)
    error_four = frobenius_relative_error(matrix, four_landmarks)
    error_eight = frobenius_relative_error(matrix, eight_landmarks)

    assert error_four < error_two
    assert error_eight < error_four
    assert error_eight < 1e-12
