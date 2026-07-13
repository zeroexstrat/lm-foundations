from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
NOTEBOOKS_DIR = ROOT / "notebooks"

EXPECTED_NOTEBOOKS = [
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


def load_script_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_notebook_scripts_exist() -> None:
    assert (SCRIPTS_DIR / "check_research_notebooks.py").exists()
    assert (SCRIPTS_DIR / "make_notebooks.py").exists()
    assert (SCRIPTS_DIR / "smoke_notebooks.py").exists()
    assert (SCRIPTS_DIR / "strip_notebook_outputs.py").exists()
    assert (SCRIPTS_DIR / "verify_research_references.py").exists()


def test_make_notebooks_defines_all_curriculum_notebooks() -> None:
    make_notebooks = load_script_module("make_notebooks")

    assert sorted(make_notebooks.NOTEBOOKS) == EXPECTED_NOTEBOOKS


def test_make_notebook_sets_expected_metadata() -> None:
    make_notebooks = load_script_module("make_notebooks")

    notebook = make_notebooks.make_notebook(
        [
            ("markdown", "# Title"),
            ("code", "value = 1"),
        ]
    )

    assert notebook.metadata["kernelspec"]["display_name"] == "LLM From Scratch"
    assert notebook.metadata["kernelspec"]["name"] == "llm-from-scratch"
    assert notebook.metadata["language_info"]["name"] == "python"
    assert [cell.cell_type for cell in notebook.cells] == ["markdown", "code"]


def test_make_notebooks_refuses_to_overwrite_existing_files_by_default(tmp_path: Path) -> None:
    make_notebooks = load_script_module("make_notebooks")

    output_dir = tmp_path / "notebooks"
    output_dir.mkdir()
    existing_path = output_dir / EXPECTED_NOTEBOOKS[0]
    existing_path.write_text("keep me")

    try:
        make_notebooks.write_notebooks(output_dir)
    except FileExistsError as exc:
        assert EXPECTED_NOTEBOOKS[0] in str(exc)
    else:
        raise AssertionError("expected FileExistsError when notebook already exists")

    assert existing_path.read_text() == "keep me"


def test_make_notebooks_writes_all_expected_notebooks(tmp_path: Path) -> None:
    make_notebooks = load_script_module("make_notebooks")
    output_dir = tmp_path / "generated"
    output_dir.mkdir()

    make_notebooks.write_notebooks(output_dir)

    assert sorted(path.name for path in output_dir.glob("*.ipynb")) == EXPECTED_NOTEBOOKS


def test_quantization_scaffold_mentions_real_module_arrives_in_task_8() -> None:
    make_notebooks = load_script_module("make_notebooks")

    quantization_cells = make_notebooks.NOTEBOOKS["10_quantization_deep_dive.ipynb"]
    markdown_sources = [source for kind, source in quantization_cells if kind == "markdown"]

    assert any("Task 8" in source and "llm_from_scratch.quantization" in source for source in markdown_sources)


def test_make_notebooks_cli_rejects_force_flag(tmp_path: Path, monkeypatch) -> None:
    make_notebooks = load_script_module("make_notebooks")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_notebooks.py", "--force"])

    with pytest.raises(SystemExit) as excinfo:
        make_notebooks.main()

    assert excinfo.value.code == 2
    assert not (tmp_path / "notebooks").exists()


def test_smoke_notebooks_quick_mode_covers_all_scaffolds() -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    assert smoke_notebooks.QUICK_NOTEBOOKS == EXPECTED_NOTEBOOKS


def test_smoke_notebooks_defines_research_sequence() -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    assert smoke_notebooks.RESEARCH_NOTEBOOKS == [
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


def test_smoke_notebooks_research_mode_only_selects_existing_research_notebooks(monkeypatch) -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    monkeypatch.setattr(
        smoke_notebooks,
        "available_notebook_names",
        lambda: [
            "00_project_orientation.ipynb",
            "13_research_orientation.ipynb",
            "15_sparse_attention_patterns.ipynb",
        ],
    )

    assert smoke_notebooks.selected_notebook_names(quick=False, research=True, notebook=None) == [
        "13_research_orientation.ipynb",
        "15_sparse_attention_patterns.ipynb",
    ]


def test_smoke_notebooks_research_mode_noops_when_no_research_notebooks_exist(monkeypatch) -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    monkeypatch.setattr(
        smoke_notebooks,
        "available_notebook_names",
        lambda: [
            "00_project_orientation.ipynb",
            "01_tensors_autograd_and_probability.ipynb",
        ],
    )

    assert smoke_notebooks.selected_notebook_names(quick=False, research=True, notebook=None) == []


def test_smoke_notebooks_notebook_flag_selects_single_notebook(monkeypatch) -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    monkeypatch.setattr(
        smoke_notebooks,
        "available_notebook_names",
        lambda: ["00_project_orientation.ipynb", "01_tensors_autograd_and_probability.ipynb"],
    )

    assert smoke_notebooks.selected_notebook_names(
        quick=False,
        research=False,
        notebook="00_project_orientation.ipynb",
    ) == ["00_project_orientation.ipynb"]


def test_smoke_notebooks_notebook_flag_rejects_unknown_name(monkeypatch) -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    monkeypatch.setattr(smoke_notebooks, "available_notebook_names", lambda: EXPECTED_NOTEBOOKS)

    with pytest.raises(SystemExit) as excinfo:
        smoke_notebooks.selected_notebook_names(
            quick=False,
            research=False,
            notebook="13_research_orientation.ipynb",
        )

    assert "missing notebooks" in str(excinfo.value)


def test_execute_notebook_can_import_project_package_from_repo_root(tmp_path: Path) -> None:
    smoke_notebooks = load_script_module("smoke_notebooks")

    notebook_path = tmp_path / "import_check.ipynb"
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "from llm_from_scratch.configs import profile_config\n"
            "profile_config('quick')"
        )
    ]
    nbformat.write(notebook, notebook_path)

    smoke_notebooks.execute_notebook(notebook_path, timeout=120)


def test_strip_notebook_clears_outputs_and_widget_metadata(tmp_path: Path) -> None:
    strip_notebook_outputs = load_script_module("strip_notebook_outputs")

    notebook_path = tmp_path / "sample.ipynb"
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("text"),
        nbformat.v4.new_code_cell(
            "print('hello')",
            execution_count=3,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="hello\n")],
        ),
    ]
    notebook.metadata["widgets"] = {"application/vnd.jupyter.widget-state+json": {}}
    nbformat.write(notebook, notebook_path)

    strip_notebook_outputs.strip_notebook(notebook_path)

    stripped = nbformat.read(notebook_path, as_version=4)
    code_cell = stripped.cells[1]
    assert code_cell.outputs == []
    assert code_cell.execution_count is None
    assert "widgets" not in stripped.metadata


def test_generated_notebooks_are_output_free_when_present() -> None:
    if not NOTEBOOKS_DIR.exists():
        return

    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        notebook = nbformat.read(notebook_path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.outputs == []
                assert cell.execution_count is None
