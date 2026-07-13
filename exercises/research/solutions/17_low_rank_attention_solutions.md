# Low-Rank And Randomized Attention Approximations Solutions

## Eckart-Young Diagnostics

1. The Eckart-Young theorem says the truncated SVD `A_k = U_k Sigma_k V_k^T` is the best rank-`k` approximation to `A` in Frobenius norm and spectral norm.
2. The spectrum shows how quickly matrix energy concentrates. Fast decay supports a low-rank assumption; slow decay warns that aggressive compression may discard important structure.
3. It is the squared Frobenius error of the best rank-`k` approximation: `||A - A_k||_F^2`.

## Sequence Projection

4. It compresses the sequence dimension. Keys and values are projected from `T` positions down to `k` projected positions.
5. `Q in R^{T x d}`, `K in R^{T x d}`, `V in R^{T x d_v}`, `E in R^{k x T}`, and `F in R^{k x T}`. The projected approximation is `softmax(Q(EK)^T / sqrt(d))(FV)`.
6. Local smoothing is redundant across nearby positions, so pooling can preserve the dominant behavior. Exact lookup depends on a specific token remaining distinct, and pooling can erase that distinction.

## Landmark / Nyström Approximation

7. The landmarks choose which rows and columns anchor the reconstruction. They define the sampled submatrix whose pseudoinverse links the sampled columns back to the full matrix.
8. More landmarks retain more of the original matrix geometry, so the sampled columns and rows span a richer subspace.
9. `A_{L,L}` can be ill-conditioned or nearly singular, so ridge regularization stabilizes the pseudoinverse.

## Failure Modes

10. A decisive entry in one query row can control the correct retrieved token while contributing little to the total squared matrix error.
11. Frobenius error averages over every entry. If only one row is mission-critical, its failure can be diluted by many easy rows that are reconstructed well.
12. Compare both a global reconstruction metric and a row-level retrieval metric such as the target query’s top-attended token or exact copied value.
