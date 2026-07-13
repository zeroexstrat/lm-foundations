# KV-Cache, Memory Bandwidth, And Inference Complexity Exercises

## KV-Cache Accounting

1. Derive the KV-cache bytes stored per token per layer for a decoder layer with batch size `B`, `H_kv` key/value heads, head dimension `d_h`, and `s` bytes per stored element.
2. Why does the formula include a factor of `2`?
3. Write the total cache formula after `T` cached tokens and `L` transformer layers.

## MHA, MQA, And GQA

4. For fixed query-head count `H_q`, how do `H_kv` differ across MHA, MQA, and GQA?
5. Why can MQA and GQA reduce KV-cache memory without changing the hidden width of the model?
6. If `H_q = 16` and `H_kv = 4`, how many query heads share each key/value head?

## Prefill Versus Decode

7. In one sentence each, describe the dominant attention-shape cost during prefill and during single-token decode.
8. Why does decode-time KV read bandwidth grow with context length even though each new token adds a constant number of cache bytes?
9. For one generated token at context length `T`, what is the approximate number of attention score elements evaluated across a batch of size `B` and `H_q` query heads?

## Systems Framing

10. What does FlashAttention optimize, and what does it not change about the exact attention equations?
11. Why is `torch.int8` in this notebook only a storage proxy and not, by itself, a complete production KV-cache quantization story?
12. Give one experiment that isolates cache-capacity growth from decode-time cache-read bandwidth.
