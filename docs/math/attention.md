# Scaled Dot-Product Attention

Given queries `Q`, keys `K`, and values `V`, attention computes

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
```

For causal language modeling, token position `t` may read positions `<= t` and must not read positions `> t`. The causal mask replaces forbidden logits with `-inf` before softmax, forcing their probabilities to zero.

The attention score matrix has shape `T x T`, so vanilla self-attention has quadratic sequence-length cost in both score memory and score computation. This fact is the bridge to sparse and subquadratic attention methods later in the curriculum.
