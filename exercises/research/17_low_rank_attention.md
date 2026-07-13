# Low-Rank And Randomized Attention Approximations Exercises

## Eckart-Young Diagnostics

1. State the Eckart-Young theorem for a matrix `A` and explain what quantity it minimizes.
2. Why is the singular value spectrum a useful diagnostic before adopting a low-rank attention approximation?
3. If `A_k = U_k Sigma_k V_k^T`, what does `sum_{r > k} sigma_r^2` represent?

## Sequence Projection

4. In a Linformer-style approximation, what object is being compressed: the feature dimension or the sequence dimension?
5. Write the projected attention approximation `softmax(Q(EK)^T / sqrt(d))(FV)` and identify the shapes of `E` and `F`.
6. Why can coarse sequence pooling work for local smoothing but fail for exact token lookup?

## Landmark / Nyström Approximation

7. In a landmark approximation `A approximately A_{:,L} A_{L,L}^+ A_{L,:}`, what role do the landmark indices `L` play?
8. Why can increasing the number of landmarks reduce reconstruction error even when the basic approximation formula stays the same?
9. What numerical issue motivates adding a small ridge term before taking a pseudoinverse?

## Failure Modes

10. Give one concrete reason low Frobenius error does not guarantee correct retrieval on every token.
11. Suppose one query row is rare but operationally critical. Why can a global matrix metric understate the real task failure?
12. Name one experiment that cleanly separates “the approximation is globally close” from “the task behavior is preserved.”
