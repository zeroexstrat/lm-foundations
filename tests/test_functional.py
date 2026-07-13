import math

import torch
import torch.nn.functional as F

from llm_from_scratch.functional import (
    causal_mask,
    cross_entropy_from_logits,
    scaled_dot_product_attention,
    stable_softmax,
)


def test_stable_softmax_rows_sum_to_one() -> None:
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
    probs = stable_softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
    assert torch.isfinite(probs).all()


def test_cross_entropy_matches_pytorch() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    targets = torch.tensor([2, 0])
    expected = F.cross_entropy(logits, targets)
    actual = cross_entropy_from_logits(logits, targets)
    assert torch.allclose(actual, expected)


def test_causal_mask_is_lower_triangular() -> None:
    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_scaled_dot_product_attention_shapes_and_masking() -> None:
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 5)
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(3))
    assert out.shape == (2, 3, 5)
    assert weights.shape == (2, 3, 3)
    assert torch.allclose(weights[0, 0, 1:], torch.zeros(2), atol=1e-7)
    assert math.isclose(weights[0, 0].sum().item(), 1.0, rel_tol=1e-6)
