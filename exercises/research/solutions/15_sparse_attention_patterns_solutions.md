# Sparse Attention Patterns Solutions

## Mask Semantics

1. Local masks assume nearby context matters most, block masks assume dense interaction is mainly useful inside or near a chunk, and global-token masks assume a small set of hub positions should aggregate or broadcast information broadly.
2. The diagonal self-edge lets each token preserve its current state, which keeps rows nonempty and makes layered reachability monotone.
3. Good: next-token prediction with mostly short-range syntax or local smoothing. Bad: a task where the answer depends on one marker far away with no local clue.

## Reachability And Complexity

4. The edge count is about `T w`, so the score computation scales linearly in `T` for fixed `w` instead of quadratically.
5. A token can attend to an intermediate token on layer 1 and then inherit that intermediate token's predecessors on layer 2, so paths compose across layers.
6. With dilation `2`, odd positions only connect to earlier odd positions and even positions only to earlier even positions, so the final odd-index token never enters the source token's even-index residue class.

## Paper Patterns

7. They act as routing hubs: every token can write to or read from a small set of shared positions, which shortens long-range dependency paths.
8. Random edges reduce graph diameter and break regular routing dead zones, so they create shortcuts that pure locality often cannot.
9. They are trying to improve graph diameter or effective path length so far-apart tokens can interact in fewer layers.

## Experimental Design

10. Put the target token at position `0`, ask the model to reproduce it at position `T-1`, and choose `T` large enough that a local window needs many hops while a pattern with one well-placed global token needs only two.
11. Useful choices include first reaching layer, receptive-field size by layer, source mass under normalized sparse propagation, or average path length to the answer token.
12. Reachability only tells you that a path exists; it does not tell you whether attention weights, optimization, or representation quality make that path usable in practice.
