# Quantization Solutions

## Range And Scale

1. `qmin = 0` and `qmax = 255`.
2. `scale = (3 - (-1)) / (255 - 0) = 4 / 255`.

## Implementation

3. `scale * (q - zero_point)`.
4. `fake_quantize_tensor` is useful when you want a floating-point tensor to experience quantization error without requiring integer-only execution.

## Error Analysis

5. Two bits provide only four representable integer levels, so several real values must share coarse bins and reconstruct imprecisely.
6. Per-tensor quantization uses one range for the whole matrix; per-channel quantization lets each row or column use a tighter range, often reducing error when channel ranges differ.

## LLM Systems

7. Weight-only quantization stores large parameter matrices in fewer bits while often leaving activations in floating point.
8. During autoregressive decoding, the KV cache grows with layers, heads, head dimension, batch size, and sequence length, so lower precision can reduce long-context memory pressure.
9. `estimate_kv_cache_bytes` grows linearly with `seq_len`, so long contexts increase the memory footprint of both keys and values at every layer.
10. Quantization speedups depend on kernels and hardware instructions; CUDA may have optimized kernels that are unavailable or immature on Apple Silicon.
