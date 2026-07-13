from __future__ import annotations

import pytest
import torch

from llm_from_scratch.research.moe import expert_capacity, route_tokens


@pytest.mark.parametrize("top_k", [1, 2])
def test_route_tokens_supports_top1_and_top2_shapes(top_k: int) -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0, -2.0],
            [0.5, 3.0, 2.0],
            [-1.0, 0.0, 5.0],
        ],
        dtype=torch.float64,
    )

    result = route_tokens(logits, top_k=top_k, capacity_factor=8.0)

    assert result.topk_indices.shape == (logits.shape[0], top_k)
    assert result.combine_weights.shape == (logits.shape[0], top_k)
    assert result.accepted_mask.shape == (logits.shape[0], top_k)
    assert result.position_in_expert.shape == (logits.shape[0], top_k)
    assert result.gate_probs.shape == logits.shape
    assert result.capacity == expert_capacity(
        logits.shape[0],
        logits.shape[1],
        top_k=top_k,
        capacity_factor=8.0,
    )

    expected = torch.topk(logits, k=top_k, dim=-1).indices
    torch.testing.assert_close(result.topk_indices, expected)

    expected_row_sums = torch.ones(logits.shape[0], dtype=torch.float64)
    torch.testing.assert_close(result.combine_weights.sum(dim=-1), expected_row_sums)
    assert torch.all(result.accepted_mask)


def test_route_tokens_enforces_capacity_and_drops_overflow() -> None:
    logits = torch.tensor(
        [
            [8.0, 0.0],
            [7.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=torch.float64,
    )

    result = route_tokens(logits, top_k=1, capacity_factor=0.5)

    assert result.capacity == 1
    torch.testing.assert_close(result.load, torch.tensor([1, 0]))
    torch.testing.assert_close(result.dropped, torch.tensor([3, 0]))
    assert result.accepted_mask.sum().item() == 1

    dropped_tokens = (~result.accepted_mask).squeeze(-1)
    assert torch.all(result.topk_indices[dropped_tokens] == -1)
    assert torch.all(result.combine_weights[dropped_tokens] == 0.0)
    assert torch.all(result.position_in_expert[dropped_tokens] == -1)


def test_route_tokens_accounts_for_attempted_and_accepted_loads() -> None:
    logits = torch.tensor(
        [
            [2.0, 1.0],
            [1.5, 1.0],
            [1.0, 0.5],
        ],
        dtype=torch.float64,
    )

    result = route_tokens(logits, top_k=2, capacity_factor=2.0 / 3.0)

    assert result.capacity == 2
    torch.testing.assert_close(result.attempted, torch.tensor([3, 3]))
    torch.testing.assert_close(result.load, torch.tensor([2, 2]))
    torch.testing.assert_close(result.dropped, torch.tensor([1, 1]))
    torch.testing.assert_close(result.load + result.dropped, result.attempted)
    assert result.accepted_mask.sum().item() == 4
