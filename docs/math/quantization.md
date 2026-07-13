# Quantization

Uniform affine quantization maps a real value `x` to an integer `q` using

```text
q = round(x / scale + zero_point).
```

Dequantization maps back to an approximate real value:

```text
x_hat = scale * (q - zero_point).
```

The scale controls bin width. The zero-point represents real zero exactly when possible. Symmetric quantization fixes `zero_point = 0`; asymmetric quantization uses the observed range to place zero inside an unsigned integer interval.

The teaching helpers in `llm_from_scratch.quantization` expose this flow directly:

- `QuantizationParams` stores `scale`, `zero_point`, `qmin`, and `qmax`.
- `quantize_tensor` performs per-tensor affine quantization.
- `dequantize_tensor` reconstructs an approximate floating-point tensor.
- `fake_quantize_tensor` quantizes and immediately dequantizes for analysis.
- `quantize_per_channel` applies per-channel quantization along one axis.
- `quantization_error` reports mean absolute and maximum absolute error.
- `estimate_kv_cache_bytes` estimates KV-cache memory from model dimensions and storage precision.

Per-channel quantization gives each output channel its own range. Groupwise quantization is a compromise: it stores several scales instead of one scale per tensor, but fewer scales than one per channel.

Fake quantization quantizes and immediately dequantizes during analysis or training. It lets a float model experience quantization error without requiring integer kernels.

For LLMs, weight-only quantization often saves memory with less calibration complexity than full activation quantization. Activation and KV-cache quantization can further reduce bandwidth or memory pressure, but the accuracy and hardware behavior depend strongly on kernels, calibration data, and device support.
