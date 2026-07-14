# The Machine That Predicts the Next Token

[![Open the flagship notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zeroexstrat/lm-foundations/blob/main/notebooks/the_machine_that_predicts_the_next_token.ipynb)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Text: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-blue.svg)](LICENSE-TEXT.md)

*A build-it-till-it-breaks course in the mathematics of language models — every idea built
from raw tensors in self-contained NumPy, then hardened to correctness.*

## Start here

**[`notebooks/the_machine_that_predicts_the_next_token.ipynb`](notebooks/the_machine_that_predicts_the_next_token.ipynb)**
is the flagship notebook and the recommended way to read this project. It builds a language
model the way you'd actually come to understand it: try the obvious thing, watch it break,
and let the break tell you what the next idea has to be — across all fourteen chapters, from
*why counting cannot produce the map* to attention, training, generation, alignment, and
beyond.

Every exploration cell is self-contained **`numpy` / `scipy` / `matplotlib`** — no
repository, no GPU, no framework — so any claim can be broken by deleting a line and
re-running. `pip install numpy scipy matplotlib` is all it needs; open it in Jupyter or
Colab and go.

**Read it rendered** (sidebar navigation, KaTeX math, live plots) at
**[0xstrategies.com/writing/llm-from-scratch.html](https://0xstrategies.com/writing/llm-from-scratch.html)**.

### How it was made

The flagship began as a *merge of two earlier drafts of this same course*: the
derivations-first PyTorch walkthrough (`99`, below) and a narrative "build-it-till-it-breaks"
rewrite. It was then **hardened through adversarial cross-model auditing** — two models took
turns auditing each other's work and each other's audits against a frozen rubric, resolving
33 findings (wrong claims, dropped rigor, a quantizer calibration bug present in *both*
sources) until both independently signed off on the same version with zero open findings. The
result keeps the rigor of the first draft and the motion of the second, on a substrate anyone
can run.

### What's inside

- **The break→repair method.** Each chapter poses a question you can feel, builds the naive
  answer in runnable code, watches it fail on something it should handle, and derives the
  repair as the thing that *had* to come next.
- **Rigor, ported and executable.** The $\sqrt D$ variance derivation; cross-entropy =
  entropy + KL (the loss floor); an end-to-end multi-head causal transformer *assembled and
  trained* in NumPy; the affine-quantizer half-step error bound (with the calibration trap
  that both source drafts got wrong); weight tying; held-out validation that actually shows
  generalization-then-overfitting; the DPO loss from the Gibbs tilt.
- **One recurring structure.** The Gibbs / maximum-entropy principle appears at three scales —
  the softmax output head, each attention row (temperature $\sqrt D$), and KL-regularized
  preference alignment. One idea, three homes.

| Ch. | Topic | Ch. | Topic |
| --- | --- | --- | --- |
| 0 | The one map, and why counting can't produce it | 7 | Generation: the knobs are the policy |
| 1 | Scores, the simplex, softmax as max-entropy | 8 | Fine-tuning: same loss, different measure |
| 2 | Choosing the atoms (tokenization) | 9 | Translation to PyTorch / Hugging Face |
| 3 | Coordinates for symbols (embeddings) | 10 | Quantization |
| 4 | Attention: one position reads another | 11 | Alignment |
| 5 | The block: new features, surviving depth | 12 | Beyond the transformer |
| 6 | Training: turning the loss into motion | 13 | How to keep going |

## The companion — derivations-first, PyTorch, package-backed

The original **[`notebooks/99_complete_college_level_walkthrough.ipynb`](notebooks/99_complete_college_level_walkthrough.ipynb)**
remains here as the companion volume: the same fourteen-chapter arc in a mathematical-physics
idiom — 40+ numbered definitions, propositions, and theorems with proofs — built on the
`src/llm_from_scratch/` PyTorch teaching implementation, with a Colab setup cell that handles
dependencies automatically.

Read the **flagship** for the runnable, self-contained, build-it-till-it-breaks path; read
**`99`** for the fully typed, package-backed treatment. They cover the same course. The
`notebooks/00`–`24` sequence is the original per-topic study path and the supplementary
research notebooks (attention variants, state-space models, MoE, scaling laws).

## Who this is for

A mathematically trained reader (linear algebra, multivariable calculus, probability,
proof-style reasoning) who has used language models only from the outside — API calls,
temperature knobs, loss curves — and wants to see the machine from raw tensors up, without
the algebra hidden.

## Quickstart

The flagship needs nothing but a scientific-Python stack:

```bash
pip install numpy scipy matplotlib
jupyter lab notebooks/the_machine_that_predicts_the_next_token.ipynb
```

The companion `99` uses the PyTorch package:

```bash
git clone https://github.com/zeroexstrat/lm-foundations.git
cd lm-foundations
python -m venv .venv && source .venv/bin/activate
pip install -e .            # torch, tokenizers, transformers, datasets, ...
jupyter lab notebooks/99_complete_college_level_walkthrough.ipynb
```

## Tests

```bash
python -m pytest tests/
```

## Sources and acknowledgments

The curriculum cross-references Stanford CS124 and Jurafsky & Martin, *Speech and Language
Processing* (3rd ed. draft), and folds in emphases from Tong Xiao and Jingbo Zhu,
*Foundations of Large Language Models* (arXiv:2501.09223, CC BY 4.0). Source notes appear
inline where material is integrated.

## License

Dual-licensed by content type: all **code** (`src/`, `scripts/`, `tests/`, notebook code
cells) under the [MIT License](LICENSE); all **text** (prose, mathematics, worked examples,
the rendered HTML textbook, `docs/`) under [CC BY 4.0](LICENSE-TEXT.md). Reuse either freely
with attribution.

## Author

Rafael Almeida — [0xstrategies.com](https://0xstrategies.com) ·
[zeroexstrat@gmail.com](mailto:zeroexstrat@gmail.com)
