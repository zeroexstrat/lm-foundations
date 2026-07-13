from __future__ import annotations

import pytest
import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

from llm_from_scratch.research.attention_math import exact_attention
from llm_from_scratch.research.linear_attention import (
    LinearAttentionTrace,
    causal_linear_attention,
    elu_plus_one_feature_map,
    kernel_attention_via_feature_map,
    linear_attention,
    softmax_random_feature_attention,
)


class TensorOutputShapeObserver(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_outputs: list[tuple[str, tuple[int, ...]]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        if kwargs is None:
            kwargs = {}
        output = func(*args, **kwargs)
        flat_output, _ = tree_flatten(output)
        for value in flat_output:
            if isinstance(value, torch.Tensor):
                self.tensor_outputs.append((str(func), tuple(int(dim) for dim in value.shape)))
        return output


def make_causal_mask(length: int) -> torch.Tensor:
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def test_elu_plus_one_feature_map_is_positive() -> None:
    x = torch.tensor([[-2.0, 0.0, 1.5]], dtype=torch.float64)

    features = elu_plus_one_feature_map(x)

    assert features.shape == x.shape
    assert torch.all(features > 0)
    assert torch.allclose(features[0, 1:], torch.tensor([1.0, 2.5], dtype=torch.float64))


def test_linear_attention_matches_explicit_feature_kernel_attention() -> None:
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

    expected = kernel_attention_via_feature_map(
        q,
        k,
        v,
        feature_map=elu_plus_one_feature_map,
        causal=False,
    )
    actual = linear_attention(q, k, v, feature_map=elu_plus_one_feature_map)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_causal_linear_attention_matches_explicit_kernel_attention() -> None:
    q = torch.tensor(
        [
            [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2], [-0.4, 0.3, 0.6], [0.5, -0.3, 0.1]],
            [[-0.1, 0.2, 0.0], [0.6, -0.5, 0.2], [0.4, 0.1, -0.3], [0.0, 0.5, 0.4]],
        ],
        dtype=torch.float64,
    )
    k = torch.tensor(
        [
            [[0.3, 0.0, -0.2], [0.5, -0.1, 0.1], [-0.2, 0.4, 0.2], [0.1, 0.3, -0.4]],
            [[0.0, -0.3, 0.5], [0.2, 0.6, -0.1], [-0.5, 0.2, 0.3], [0.4, 0.0, 0.1]],
        ],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [
            [[0.5, -0.2], [1.0, 0.4], [-0.3, 0.7], [0.1, 0.9]],
            [[-0.4, 0.3], [0.6, -0.2], [0.8, 0.5], [-0.1, 0.2]],
        ],
        dtype=torch.float64,
    )
    trace = LinearAttentionTrace()

    expected = kernel_attention_via_feature_map(
        q,
        k,
        v,
        feature_map=elu_plus_one_feature_map,
        causal=True,
    )
    actual = causal_linear_attention(
        q,
        k,
        v,
        feature_map=elu_plus_one_feature_map,
        trace=trace,
    )

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert any(name == "feature_kv_prefix" for name, _ in trace.intermediate_shapes)


def test_causal_linear_attention_never_emits_sequence_square_tensor_outputs() -> None:
    q = torch.tensor(
        [[[0.2, -0.1, 0.3], [0.1, 0.4, -0.2], [-0.4, 0.3, 0.6], [0.5, -0.3, 0.1]]],
        dtype=torch.float64,
    )
    k = torch.tensor(
        [[[0.3, 0.0, -0.2], [0.5, -0.1, 0.1], [-0.2, 0.4, 0.2], [0.1, 0.3, -0.4]]],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [[[0.5, -0.2], [1.0, 0.4], [-0.3, 0.7], [0.1, 0.9]]],
        dtype=torch.float64,
    )
    observer = TensorOutputShapeObserver()

    with observer:
        output = causal_linear_attention(
            q,
            k,
            v,
            feature_map=elu_plus_one_feature_map,
        )

    sequence_length = q.shape[-2]
    sequence_square_outputs = [
        (name, shape)
        for name, shape in observer.tensor_outputs
        if len(shape) >= 2 and shape[-2:] == (sequence_length, sequence_length)
    ]

    assert output.shape == v.shape
    assert observer.tensor_outputs
    assert sequence_square_outputs == []


def test_random_feature_attention_tracks_exact_softmax_better_with_more_features() -> None:
    q = torch.tensor(
        [[[0.1, -0.2, 0.0], [0.3, 0.2, -0.1], [-0.4, 0.1, 0.2], [0.2, -0.3, 0.5]]],
        dtype=torch.float64,
    )
    k = torch.tensor(
        [[[0.0, 0.2, -0.1], [0.4, -0.2, 0.3], [-0.3, 0.5, 0.1], [0.2, 0.1, -0.4]]],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [[[0.5, -0.1], [0.2, 0.7], [-0.6, 0.3], [0.4, 0.2]]],
        dtype=torch.float64,
    )
    mask = make_causal_mask(q.shape[-2])

    exact_output, _ = exact_attention(q, k, v, mask=mask)
    low_rank_output = softmax_random_feature_attention(q, k, v, num_features=8, seed=0, causal=True)
    high_rank_output = softmax_random_feature_attention(q, k, v, num_features=128, seed=0, causal=True)

    low_error = (low_rank_output - exact_output).abs().mean().item()
    high_error = (high_rank_output - exact_output).abs().mean().item()

    assert high_error < low_error
    assert high_error < 0.08


def test_linear_attention_rejects_mismatched_shapes() -> None:
    q = torch.randn(2, 4, 3, dtype=torch.float64)
    k = torch.randn(2, 5, 3, dtype=torch.float64)
    v = torch.randn(2, 4, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="key/value sequence length"):
        linear_attention(q, k, v)
