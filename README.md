# Mathematical Foundations of Language Model Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeroexstrat/lm-foundations/blob/main/notebooks/99_complete_college_level_walkthrough.ipynb)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Text: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-blue.svg)](LICENSE-TEXT.md)

*A derivations-first textbook on language models, written in the style of mathematical
physics — with the Gibbs variational principle as the recurring thread.*

**Read it three ways:**

1. **[Interactive textbook](output/interactive_textbook.html)** — a single self-contained
   HTML file (open it in any browser, no server, no dependencies): 1,000+ rendered
   equations each with a plain-English "how to read this" toggle, and eight interactive
   panels running on real traces from an actual training run of the included model.
2. **[The notebook](notebooks/99_complete_college_level_walkthrough.ipynb)** — the source
   of truth. 100 cells, runnable locally or on Google Colab (the first code cell
   handles Colab setup automatically).
3. **The code** — `src/llm_from_scratch/`, a ~700-line teaching implementation of a
   decoder-only transformer that every formula in the book lands in.

## What this book is

The reader in mind is mathematically trained (linear algebra, multivariable calculus,
probability, proof-style reasoning) but has used language models only from the outside —
API calls, temperature knobs, loss curves — without seeing the backend. The book fills
exactly that gap, with the standards of a mathematics text:

- **Everything is typed.** Every map is introduced with an explicit domain and codomain,
  from the tokenizer pair $\mathrm{enc}: \mathcal{A}^* \to \mathcal{V}^*$ to the full model as one
  composition $\mathcal{V}^T \to \mathbb{R}^{T\times C} \to \cdots \to (\mathring{\Delta}^{V-1})^T$.
- **Everything is derived.** 40+ numbered definitions, propositions, and theorems with
  proofs: the softmax Jacobian, the maximum-entropy characterization of softmax, the
  permutation-equivariance theorem that justifies position embeddings, the $1/\sqrt{D}$
  variance lemma, the AdamW update line by line, the quantization error bound
  (achieved, not just stated), the DPO loss from the Gibbs tilt.
- **Everything is computed.** Twelve worked examples carry one tiny running example —
  the word `"cat"` over a four-token vocabulary — through every mechanism with each
  digit visible, and a code cell re-verifies each one against the real implementation.
- **One structure recurs.** The Gibbs variational principle is proved once (Theorem 1.6)
  and recognized three times: as the softmax output head, inside every attention row
  (temperature $\sqrt{D}$), and as the closed-form optimum of KL-regularized preference
  alignment. One theorem, three scales.
- **The narrative is explicit.** Each chapter ends with a "Where We Stand" bridge — what
  we now have, restated as an evolving *definition* of an autoregressive language model,
  what is still missing, and how the next chapter closes the gap. The definition is
  refined thirteen times, from "an assignment of a next-token distribution to every
  prefix" to the full system object of modern practice.

## Contents

| Ch. | Topic | Ch. | Topic |
| --- | --- | --- | --- |
| 0 | What a language model computes | 7 | Generation and evaluation |
| 1 | Tensors, autograd, and probability | 8 | Supervised fine-tuning |
| 2 | Tokenization | 9 | Hugging Face translation layer |
| 3 | Embeddings and label shifting | 10 | Quantization |
| 4 | Attention from raw tensors | 11 | Modern LLM practice |
| 5 | The transformer block | 12 | Beyond transformers |
| 6 | Training | 13 | How to study from here |

## Quickstart

```bash
git clone https://github.com/zeroexstrat/lm-foundations.git
cd lm-foundations
python -m venv .venv && source .venv/bin/activate
pip install -e .            # torch, tokenizers, transformers, datasets, ...
jupyter lab notebooks/99_complete_college_level_walkthrough.ipynb
```

Or open the notebook directly in Colab — the first code cell clones this repo and
installs dependencies.

## Rebuilding the interactive textbook

```bash
npm install                                  # KaTeX (server-side rendering assets)
python scripts/viz_data.py                   # ~1 min: trains the toy model, exports real traces
python scripts/build_interactive_textbook.py # emits output/interactive_textbook.html
```

`viz_data.json` is committed, so a rebuild works without retraining; regenerate it to
refresh the traces.

## Tests

```bash
python -m pytest tests/
```

## Sources and acknowledgments

The curriculum cross-references Stanford CS124 and Jurafsky & Martin, *Speech and
Language Processing* (3rd ed. draft), and folds in emphases from Tong Xiao and Jingbo
Zhu, *Foundations of Large Language Models* (arXiv:2501.09223, CC BY 4.0). Source notes
appear inline where material is integrated. The supplementary research notebooks
(14–24) cover attention variants, state-space models, MoE, scaling laws, and
optimization at scale.

## License

Dual-licensed by content type: all **code** (`src/`, `scripts/`, `tests/`, notebook code
cells) under the [MIT License](LICENSE); all **text** (prose, mathematics, worked
examples, the rendered HTML textbook, `docs/`) under
[CC BY 4.0](LICENSE-TEXT.md). Reuse either freely with attribution.

## Author

Rafael Almeida — [0xstrategies.com](https://0xstrategies.com) ·
[zeroexstrat@gmail.com](mailto:zeroexstrat@gmail.com)
