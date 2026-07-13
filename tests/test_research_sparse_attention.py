from __future__ import annotations

import torch

from llm_from_scratch.research.sparse_attention import (
    block_causal_mask,
    combine_masks,
    dilated_causal_mask,
    global_causal_mask,
    local_causal_mask,
    random_causal_mask,
    receptive_field_sizes,
    stacked_reachability,
)


def assert_causal(mask: torch.Tensor) -> None:
    assert mask.dtype is torch.bool
    assert torch.equal(mask, torch.tril(mask))
    assert torch.all(torch.diagonal(mask))


def test_local_causal_mask_matches_sliding_window() -> None:
    mask = local_causal_mask(5, window_size=2)

    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, True, True, False, False],
            [False, False, True, True, False],
            [False, False, False, True, True],
        ]
    )

    assert torch.equal(mask, expected)
    assert_causal(mask)


def test_block_causal_mask_covers_current_and_previous_block() -> None:
    mask = block_causal_mask(6, block_size=2, num_previous_blocks=1)

    expected = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [False, False, True, True, True, False],
            [False, False, True, True, True, True],
        ]
    )

    assert torch.equal(mask, expected)
    assert_causal(mask)


def test_dilated_causal_mask_keeps_stride_aligned_predecessors() -> None:
    mask = dilated_causal_mask(7, dilation=2, window_size=3)

    expected = torch.tensor(
        [
            [True, False, False, False, False, False, False],
            [False, True, False, False, False, False, False],
            [True, False, True, False, False, False, False],
            [False, True, False, True, False, False, False],
            [True, False, True, False, True, False, False],
            [False, True, False, True, False, True, False],
            [False, False, True, False, True, False, True],
        ]
    )

    assert torch.equal(mask, expected)
    assert_causal(mask)


def test_global_causal_mask_adds_prefix_globals_and_dense_global_queries() -> None:
    mask = global_causal_mask(6, global_indices=[0, 3], local_window=2)

    expected = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, False, False, True, True, False],
            [True, False, False, True, True, True],
        ]
    )

    assert torch.equal(mask, expected)
    assert_causal(mask)


def test_random_causal_mask_is_seeded_and_preserves_causality() -> None:
    first = random_causal_mask(8, num_random=2, seed=123, local_window=1)
    second = random_causal_mask(8, num_random=2, seed=123, local_window=1)
    third = random_causal_mask(8, num_random=2, seed=124, local_window=1)

    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert_causal(first)
    assert torch.all(first.sum(dim=-1) <= 3)
    assert first[7].sum().item() == 3


def test_combined_masks_and_reachability_expand_with_layers() -> None:
    local = local_causal_mask(6, window_size=2)
    global_mask = global_causal_mask(6, global_indices=[0, 3], local_window=1)
    combined = combine_masks(local, global_mask)

    local_counts = receptive_field_sizes(local, max_layers=4)
    combined_counts = receptive_field_sizes(combined, max_layers=2)

    assert torch.equal(local_counts[:, 5], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(combined_counts[:, 5], torch.tensor([4, 6]))
    assert stacked_reachability(local, layers=5)[5, 0].item() is True
    assert stacked_reachability(combined, layers=2)[5, 0].item() is True
