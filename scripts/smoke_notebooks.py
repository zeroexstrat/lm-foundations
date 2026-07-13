from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
NOTEBOOKS_DIR = ROOT / "notebooks"

QUICK_NOTEBOOKS = [
    "00_project_orientation.ipynb",
    "01_tensors_autograd_and_probability.ipynb",
    "02_text_tokenization_char_to_subword.ipynb",
    "03_embeddings_and_language_modeling.ipynb",
    "04_attention_from_raw_tensors.ipynb",
    "05_transformer_block_from_scratch.ipynb",
    "06_training_loop_loss_and_optimization.ipynb",
    "07_generation_sampling_and_evaluation.ipynb",
    "08_finetuning_toy_instruction_task.ipynb",
    "09_hugging_face_translation_layer.ipynb",
    "10_quantization_deep_dive.ipynb",
    "11_modern_llm_orientation.ipynb",
    "12_beyond_transformers_and_world_models.ipynb",
]

RESEARCH_NOTEBOOKS = [
    "13_research_orientation.ipynb",
    "14_attention_as_kernel_operator.ipynb",
    "15_sparse_attention_patterns.ipynb",
    "16_linear_kernel_attention.ipynb",
    "17_low_rank_attention.ipynb",
    "18_kv_cache_inference.ipynb",
    "19_long_context_evaluation.ipynb",
    "20_state_space_models.ipynb",
    "21_selective_recurrence_mamba_toy.ipynb",
    "22_moe_conditional_compute.ipynb",
    "23_scaling_laws_lab.ipynb",
    "24_optimization_at_scale.ipynb",
    "25_self_supervised_objectives.ipynb",
    "26_jepa_predictive_representations.ipynb",
    "27_world_models_and_planning.ipynb",
    "28_retrieval_memory_systems.ipynb",
    "29_mechanistic_interpretability.ipynb",
    "30_evaluation_validity.ipynb",
    "31_compression_research.ipynb",
    "32_research_capstone_workshop.ipynb",
]

NOTEBOOK_TIMEOUTS = {
    "13_research_orientation.ipynb": 180,
}


def available_notebook_names() -> list[str]:
    return sorted(path.name for path in NOTEBOOKS_DIR.glob("*.ipynb"))


def selected_notebook_names(quick: bool, research: bool, notebook: str | None) -> list[str]:
    available = set(available_notebook_names())
    if notebook is not None:
        names = [notebook]
    elif research:
        names = [name for name in RESEARCH_NOTEBOOKS if name in available]
        if not names:
            return []
    elif quick:
        names = QUICK_NOTEBOOKS
    else:
        names = available_notebook_names()

    available = set(available_notebook_names())
    duplicates = sorted({name for name in names if names.count(name) > 1})
    missing = sorted(name for name in names if name not in available)
    if duplicates or missing:
        details: list[str] = []
        if duplicates:
            details.append(f"duplicate notebook names: {', '.join(duplicates)}")
        if missing:
            details.append(f"missing notebooks under {NOTEBOOKS_DIR}: {', '.join(missing)}")
        raise SystemExit(f"invalid notebook selection: {'; '.join(details)}")
    if not names:
        raise SystemExit(f"no notebooks found under {NOTEBOOKS_DIR}")
    return names


def execute_notebook(path: Path, timeout: int = 120) -> None:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "import sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})"
        ),
    )
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run the quick notebook subset.")
    group.add_argument("--research", action="store_true", help="Run existing research notebooks.")
    group.add_argument("--notebook", help="Run one notebook by file name.")
    args = parser.parse_args()
    names = selected_notebook_names(args.quick, args.research, args.notebook)
    for name in names:
        execute_notebook(NOTEBOOKS_DIR / name, timeout=NOTEBOOK_TIMEOUTS.get(name, 120))


if __name__ == "__main__":
    main()
