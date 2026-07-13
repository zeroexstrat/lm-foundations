# Softmax And Cross-Entropy

For logits `z in R^V`, softmax defines

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Subtracting `max(z)` before exponentiation leaves the probabilities unchanged and prevents overflow.

For a target class `y`, the negative log-likelihood is

```text
L = -log p_y = -z_y + log(sum_j exp(z_j)).
```

The gradient with respect to each logit is

```text
dL/dz_i = p_i - 1[i = y].
```

This is why the output layer receives a dense probability-error signal even though the target is a single class index.
