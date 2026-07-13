from __future__ import annotations

import math

import torch
import pytest

from llm_from_scratch.functional import causal_mask
from llm_from_scratch.research.attention_math import (
    attention_entropy,
    check_row_stochastic,
    exact_attention,
    finite_difference_attention_gradients,
)


def test_exact_attention_preserves_shapes_and_row_sums() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 3)
    k = torch.randn(2, 4, 3)
    v = torch.randn(2, 4, 5)

    output, weights = exact_attention(q, k, v)

    assert output.shape == (2, 4, 5)
    assert weights.shape == (2, 4, 4)
    assert check_row_stochastic(weights)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_exact_attention_respects_causal_mask() -> None:
    q = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]])
    k = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]])
    v = torch.tensor([[[1.0], [10.0], [100.0]]])

    _, masked_weights = exact_attention(q, k, v, mask=causal_mask(3).unsqueeze(0))
    _, full_weights = exact_attention(q, k, v)

    assert torch.allclose(masked_weights[0].triu(diagonal=1), torch.zeros(3, 3), atol=1e-8)
    assert full_weights[0, 0, 2] > 0.0
    assert masked_weights[0, 0, 2] == 0.0
    assert torch.allclose(masked_weights.sum(dim=-1), torch.ones_like(masked_weights.sum(dim=-1)), atol=1e-6)


def test_exact_attention_rejects_fully_masked_rows() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[1.0], [2.0]]])
    mask = torch.tensor([[[True, False], [False, False]]])

    with pytest.raises(ValueError, match="no valid key"):
        exact_attention(q, k, v, mask=mask)


@pytest.mark.parametrize(
    ("q", "k", "v", "expected_message"),
    [
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[[1.0, 2.0]]]),
            torch.tensor([[[1.0]]]),
            "at least 2 dimensions",
        ),
        (
            torch.randn(1, 2, 3),
            torch.randn(2, 2, 3),
            torch.randn(1, 2, 4),
            "share leading batch dimensions",
        ),
        (
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 4),
            torch.randn(1, 2, 4),
            "same feature dimension",
        ),
        (
            torch.randn(1, 2, 3),
            torch.randn(1, 3, 3),
            torch.randn(1, 2, 4),
            "share the key/value sequence length",
        ),
    ],
)
def test_exact_attention_rejects_invalid_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        exact_attention(q, k, v)


def test_attention_entropy_matches_known_limits() -> None:
    weights = torch.tensor(
        [
            [[0.5, 0.5], [1.0, 0.0]],
            [[0.25, 0.25], [0.75, 0.25]],
        ]
    )

    entropy = attention_entropy(weights)

    assert torch.allclose(entropy[0, 0], torch.tensor(math.log(2.0)), atol=1e-6)
    assert torch.allclose(entropy[0, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.all(entropy >= 0.0)
    assert torch.all(entropy <= math.log(weights.shape[-1]) + 1e-6)


def test_check_row_stochastic_rejects_invalid_rows() -> None:
    valid = torch.tensor([[[0.2, 0.8], [0.5, 0.5]]])
    invalid = torch.tensor([[[0.2, 0.9], [-0.1, 1.0]]])

    assert check_row_stochastic(valid)
    assert not check_row_stochastic(invalid)


def test_finite_difference_attention_gradients_stay_within_tolerance() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 3, 2, dtype=torch.float64)
    k = torch.randn(1, 3, 2, dtype=torch.float64)
    v = torch.randn(1, 3, 2, dtype=torch.float64)

    diagnostics = finite_difference_attention_gradients(q, k, v, eps=1e-6)

    assert set(diagnostics) == {"q", "k", "v"}
    for stats in diagnostics.values():
        assert stats["analytic"].shape == stats["numerical"].shape
        assert stats["max_abs_error"] < 5e-4
