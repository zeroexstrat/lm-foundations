# Curriculum Index

## Core Build

1. Project orientation
2. Tensors, autograd, and probability
3. Tokenization
4. Embeddings and language modeling
5. Attention from raw tensors
6. Transformer block
7. Training loop
8. Generation and evaluation
9. Toy supervised fine-tuning

## Abstraction And Orientation

10. Hugging Face translation layer
11. Quantization deep dive
12. Modern LLM orientation
13. Beyond transformers and world models

## Research-Level Extension

The next layer is specified in [`research_level_curriculum.md`](research_level_curriculum.md) and mirrored in [`research_notebook_index.md`](research_notebook_index.md). It extends the college-level build into advanced attention mathematics, sparse and kernelized attention, state-space models, MoE, scaling laws, optimization at scale, representation learning, JEPA-style objectives, world models, retrieval, interpretability, evaluation validity, compression, and research methodology.

Research notebooks are added incrementally. The curriculum index is the navigation layer, not a guarantee that every notebook file already exists.

## Verification

Run:

```bash
uv run python -m pytest
uv run python scripts/smoke_notebooks.py --quick
```
