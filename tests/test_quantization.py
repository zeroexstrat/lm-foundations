import torch

from llm_from_scratch.quantization import (
    dequantize_tensor,
    estimate_kv_cache_bytes,
    fake_quantize_tensor,
    quantization_error,
    quantize_per_channel,
    quantize_tensor,
)


def test_asymmetric_quantize_tensor_matches_expected_math() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    q, params = quantize_tensor(x, num_bits=2, symmetric=False)
    x_hat = dequantize_tensor(q, params)
    assert params.qmin == 0
    assert params.qmax == 3
    assert params.scale == 2.0 / 3.0
    assert params.zero_point == 2
    assert torch.equal(q, torch.tensor([0, 2, 3], dtype=torch.int32))
    assert torch.allclose(
        x_hat,
        torch.tensor([-4.0 / 3.0, 0.0, 2.0 / 3.0], dtype=torch.float32),
    )


def test_symmetric_quantize_tensor_matches_expected_math() -> None:
    x = torch.tensor([-2.0, -4.0 / 7.0, 0.0, 4.0 / 7.0, 2.0], dtype=torch.float32)
    q, params = quantize_tensor(x, num_bits=4, symmetric=True)
    x_hat = dequantize_tensor(q, params)
    assert params.qmin == -7
    assert params.qmax == 7
    assert params.scale == 2.0 / 7.0
    assert params.zero_point == 0
    assert torch.equal(q, torch.tensor([-7, -2, 0, 2, 7], dtype=torch.int32))
    assert torch.allclose(
        x_hat,
        torch.tensor(
            [-2.0, -4.0 / 7.0, 0.0, 4.0 / 7.0, 2.0],
            dtype=torch.float32,
        ),
    )


def test_quantization_error_reports_mae_and_max_abs() -> None:
    x = torch.tensor([0.0, 1.0, 2.0])
    x_hat = torch.tensor([0.0, 1.1, 1.8])
    err = quantization_error(x, x_hat)
    assert set(err) == {"mae", "max_abs"}
    assert err["max_abs"] > 0


def test_fake_quantize_preserves_float_dtype_and_shape() -> None:
    x = torch.randn(3, 4)
    x_hat = fake_quantize_tensor(x, num_bits=4, symmetric=True)
    assert x_hat.shape == x.shape
    assert x_hat.dtype == torch.float32


def test_quantize_per_channel_axis_0_uses_row_specific_params() -> None:
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
    q, params = quantize_per_channel(x, axis=0, num_bits=2)
    assert q.shape == x.shape
    assert len(params) == 2
    assert params[0].scale == 2.0 / 3.0
    assert params[0].zero_point == 0
    assert params[1].scale == 4.0 / 3.0
    assert params[1].zero_point == 0
    assert torch.equal(q[0], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert torch.equal(q[1], torch.tensor([0, 2, 3], dtype=torch.int32))


def test_quantize_per_channel_axis_1_uses_column_specific_params() -> None:
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
    q, params = quantize_per_channel(x, axis=1, num_bits=2)
    assert q.shape == x.shape
    assert len(params) == 3
    assert params[0].scale == 1e-12
    assert params[0].zero_point == 0
    assert params[1].scale == 1.0 / 3.0
    assert params[1].zero_point == 0
    assert params[2].scale == 2.0 / 3.0
    assert params[2].zero_point == 0
    assert torch.equal(q[:, 0], torch.tensor([0, 0], dtype=torch.int32))
    assert torch.equal(q[:, 1], torch.tensor([3, 3], dtype=torch.int32))
    assert torch.equal(q[:, 2], torch.tensor([3, 3], dtype=torch.int32))


def test_quantize_per_channel_negative_axis_matches_last_dimension() -> None:
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
    q_axis_1, params_axis_1 = quantize_per_channel(x, axis=1, num_bits=2)
    q_axis_neg1, params_axis_neg1 = quantize_per_channel(x, axis=-1, num_bits=2)
    assert torch.equal(q_axis_neg1, q_axis_1)
    assert len(params_axis_neg1) == len(params_axis_1)
    for actual, expected in zip(params_axis_neg1, params_axis_1):
        assert actual.scale == expected.scale
        assert actual.zero_point == expected.zero_point
        assert actual.qmin == expected.qmin
        assert actual.qmax == expected.qmax


def test_estimate_kv_cache_bytes_counts_keys_and_values() -> None:
    bytes_used = estimate_kv_cache_bytes(
        n_layer=2,
        n_head=4,
        head_dim=8,
        seq_len=16,
        batch_size=1,
        bytes_per_value=2,
    )
    assert bytes_used == 2 * 2 * 4 * 8 * 16 * 1 * 2
