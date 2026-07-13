# Quantization Exercises

## Range And Scale

1. For unsigned 8-bit asymmetric quantization, what are `qmin` and `qmax`?
2. Given `x_min = -1`, `x_max = 3`, `qmin = 0`, and `qmax = 255`, compute the scale.

## Implementation

3. Fill in the missing dequantization formula used by `dequantize_tensor`: `x_hat = ____`.
4. Write one sentence explaining what `fake_quantize_tensor` is useful for.

## Error Analysis

5. Quantize `[-1.0, 0.0, 1.0]` to 2 bits with `quantize_tensor`. Why should you expect visible reconstruction error?
6. Compare per-tensor quantization with `quantize_tensor` and per-channel quantization with `quantize_per_channel` for a weight matrix whose rows have very different ranges.

## LLM Systems

7. Explain why weight-only quantization primarily targets model memory.
8. Explain why KV-cache quantization affects long-context generation.
9. Use `estimate_kv_cache_bytes` to explain one reason long sequences are expensive.
10. Name one reason a quantization method may work on CUDA but not Apple Silicon.
