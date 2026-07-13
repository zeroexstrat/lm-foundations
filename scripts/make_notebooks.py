from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


SCAFFOLD_ONLY_WARNING = (
    "This script is a one-time scaffold generator. After curriculum content is added, "
    "notebooks/*.ipynb are the source of truth."
)


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "00_project_orientation.ipynb": [
        (
            "markdown",
            "# Project Orientation\n\n"
            "This notebook explains the curriculum path, runtime profiles, and verification commands.",
        ),
        (
            "code",
            "from llm_from_scratch.configs import get_device, profile_config\n"
            "get_device(), profile_config('quick')",
        ),
    ],
    "01_tensors_autograd_and_probability.ipynb": [
        (
            "markdown",
            "# Tensors, Autograd, And Probability\n\n"
            "We connect tensor operations to categorical next-token prediction.",
        ),
        (
            "code",
            "import torch\n"
            "x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
            "y = (x**2).sum()\n"
            "y.backward()\n"
            "x.grad",
        ),
    ],
    "02_text_tokenization_char_to_subword.ipynb": [
        (
            "markdown",
            "# Text Tokenization: Character To Subword\n\n"
            "Start with character IDs, then compare the idea to BPE tokenization.",
        ),
        (
            "code",
            "from pathlib import Path\n"
            "from llm_from_scratch.tokenization import CharTokenizer\n"
            "text = Path('../data/tiny_corpus.txt').read_text()\n"
            "tok = CharTokenizer.from_text(text)\n"
            "ids = tok.encode(text[:40])\n"
            "tok.decode(ids), tok.vocab_size",
        ),
    ],
    "03_embeddings_and_language_modeling.ipynb": [
        (
            "markdown",
            "# Embeddings And Language Modeling\n\n"
            "Derive next-token modeling and inspect embedding lookup tensors.",
        ),
        (
            "code",
            "import torch\n"
            "embedding = torch.nn.Embedding(10, 4)\n"
            "ids = torch.tensor([[1, 2, 3]])\n"
            "embedding(ids).shape",
        ),
    ],
    "04_attention_from_raw_tensors.ipynb": [
        (
            "markdown",
            "# Attention From Raw Tensors\n\n"
            "Build scaled dot-product attention and expose the O(T^2) score matrix.",
        ),
        (
            "code",
            "import torch\n"
            "from llm_from_scratch.functional import causal_mask, scaled_dot_product_attention\n"
            "q = torch.randn(1, 4, 8)\n"
            "k = torch.randn(1, 4, 8)\n"
            "v = torch.randn(1, 4, 8)\n"
            "out, weights = scaled_dot_product_attention(q, k, v, causal_mask(4))\n"
            "out.shape, weights.shape",
        ),
    ],
    "05_transformer_block_from_scratch.ipynb": [
        (
            "markdown",
            "# Transformer Block From Scratch\n\n"
            "Assemble attention, residual paths, layer norm, and MLP layers.",
        ),
        (
            "code",
            "import torch\n"
            "from llm_from_scratch.configs import ModelConfig\n"
            "from llm_from_scratch.model import TinyGPT\n"
            "cfg = ModelConfig(vocab_size=32, block_size=8, n_embd=16, n_head=4, n_layer=1)\n"
            "model = TinyGPT(cfg)\n"
            "model(torch.randint(0, 32, (2, 8)))[0].shape",
        ),
    ],
    "06_training_loop_loss_and_optimization.ipynb": [
        (
            "markdown",
            "# Training Loop, Loss, And Optimization\n\n"
            "Train a tiny model and inspect loss movement.",
        ),
        (
            "code",
            "import torch\n"
            "from llm_from_scratch.configs import ModelConfig\n"
            "from llm_from_scratch.model import TinyGPT\n"
            "from llm_from_scratch.train import overfit_tiny_batch\n"
            "cfg = ModelConfig(vocab_size=8, block_size=4, n_embd=16, n_head=4, n_layer=1, dropout=0.0)\n"
            "model = TinyGPT(cfg)\n"
            "x = torch.randint(0, 8, (4, 4))\n"
            "y = torch.randint(0, 8, (4, 4))\n"
            "overfit_tiny_batch(model, x, y, steps=5, lr=1e-2)",
        ),
    ],
    "07_generation_sampling_and_evaluation.ipynb": [
        (
            "markdown",
            "# Generation, Sampling, And Evaluation\n\n"
            "Compare temperature, top-k, top-p, validation loss, and perplexity.",
        ),
        (
            "code",
            "from llm_from_scratch.evaluate import perplexity\n"
            "perplexity(1.0)",
        ),
    ],
    "08_finetuning_toy_instruction_task.ipynb": [
        (
            "markdown",
            "# Fine-Tuning A Toy Instruction Task\n\n"
            "Format prompt-response pairs and discuss supervised fine-tuning.",
        ),
        (
            "code",
            "from llm_from_scratch.data import toy_instruction_examples\n"
            "toy_instruction_examples()[:2]",
        ),
    ],
    "09_hugging_face_translation_layer.ipynb": [
        (
            "markdown",
            "# Hugging Face Translation Layer\n\n"
            "Map handmade data, tokenizer, model, training, and generation pieces to library abstractions.",
        ),
        (
            "code",
            "import datasets\n"
            "import tokenizers\n"
            "import transformers\n"
            "transformers.__version__, datasets.__version__, tokenizers.__version__",
        ),
    ],
    "10_quantization_deep_dive.ipynb": [
        (
            "markdown",
            "# Quantization Deep Dive\n\n"
            "Derive scale and zero-point, implement toy quantization, then compare to modern APIs.\n\n"
            "Task 7 keeps this scaffold self-contained so smoke tests pass before "
            "`llm_from_scratch.quantization` exists. Task 8 replaces this with the real module-backed demo.",
        ),
        (
            "code",
            "import torch\n"
            "x = torch.tensor([-1.0, 0.0, 1.0])\n"
            "qmin, qmax = 0, 255\n"
            "scale = (x.max() - x.min()) / (qmax - qmin)\n"
            "zero_point = int(round(qmin - x.min().item() / scale.item()))\n"
            "q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax).to(torch.int32)\n"
            "x_hat = (q.float() - zero_point) * scale\n"
            "q, x_hat",
        ),
    ],
    "11_modern_llm_orientation.ipynb": [
        (
            "markdown",
            "# Modern LLM Orientation\n\n"
            "Orient from the toy GPT to instruction tuning, RAG, deployment, and systems concerns.",
        ),
        ("code", "print('orientation notebook')"),
    ],
    "12_beyond_transformers_and_world_models.ipynb": [
        (
            "markdown",
            "# Beyond Transformers And World Models\n\n"
            "Use the transformer build to reason about sparse attention, post-transformer techniques, JEPA, and world models.",
        ),
        (
            "code",
            "sequence_lengths = [128, 512, 2048]\n"
            "[(n, n * n) for n in sequence_lengths]",
        ),
    ],
}


def make_notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(source) if kind == "markdown" else nbf.v4.new_code_cell(source)
        for kind, source in cells
    ]
    notebook.metadata["kernelspec"] = {
        "display_name": "LLM From Scratch",
        "language": "python",
        "name": "llm-from-scratch",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


def write_notebooks(out_dir: Path) -> None:
    existing = sorted(name for name in NOTEBOOKS if (out_dir / name).exists())
    if existing:
        raise FileExistsError("Refusing to overwrite existing notebooks: " + ", ".join(existing))

    for name, cells in NOTEBOOKS.items():
        nbf.write(make_notebook(cells), out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    print(SCAFFOLD_ONLY_WARNING)
    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)
    write_notebooks(out_dir)


if __name__ == "__main__":
    main()
