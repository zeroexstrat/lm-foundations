# KV-Cache, Memory Bandwidth, And Inference Complexity Solutions

## KV-Cache Accounting

1. The stored bytes per token per layer are `2 * B * H_kv * d_h * s`.
2. The factor of `2` accounts for storing both keys and values.
3. After `T` cached tokens and `L` layers, total KV-cache bytes are `L * T * 2 * B * H_kv * d_h * s`.

## MHA, MQA, And GQA

4. MHA uses `H_kv = H_q`, MQA uses `H_kv = 1`, and GQA uses an intermediate grouped count with `1 < H_kv < H_q`.
5. They share keys and values across multiple query heads, so the stored KV tensors shrink even though the query-side model width stays the same.
6. Each key/value head serves `16 / 4 = 4` query heads.

## Prefill Versus Decode

7. Prefill evaluates all prompt-query against prompt-key interactions, so the score workload is quadratic in prompt length. Single-token decode evaluates one new query against all cached keys, so the score workload is linear in current context length.
8. Each extra token adds one more cached key and value slice that future decode steps must read, so the total read traffic per generated token increases with context even though the write increment stays constant.
9. Approximately `B * H_q * T` score elements are evaluated for that token.

## Systems Framing

10. FlashAttention optimizes IO and memory traffic for exact attention kernels. It does not change the exact softmax-attention math or the KV-cache storage formula.
11. `torch.int8` only tells us what one-byte storage would cost. Real KV-cache quantization also needs scale handling, kernel support, and an accuracy tradeoff study.
12. Hold the cache write precision and per-token cache growth fixed, then sweep context length while measuring only the decode step’s `kv_read_bytes_total`.
