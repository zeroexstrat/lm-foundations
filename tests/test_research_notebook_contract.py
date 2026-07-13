from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def load_script_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_notebook(path: Path, markdown_cells: list[str]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [nbformat.v4.new_markdown_cell(source) for source in markdown_cells]
    nbformat.write(notebook, path)


def test_contract_check_passes_when_no_research_notebooks_exist(tmp_path: Path) -> None:
    checker = load_script_module("check_research_notebooks")

    assert checker.collect_contract_failures(tmp_path) == []


def test_contract_check_reports_missing_required_sections(tmp_path: Path) -> None:
    checker = load_script_module("check_research_notebooks")
    notebook_path = tmp_path / "13_research_orientation.ipynb"
    write_notebook(
        notebook_path,
        [
            "# Motivation",
            "## Mathematical derivation",
            "## PyTorch implementation",
        ],
    )

    failures = checker.collect_contract_failures(tmp_path)

    assert len(failures) == 1
    failure = failures[0]
    assert failure.notebook_name == "13_research_orientation.ipynb"
    assert failure.missing_headings == ["Numerical checks", "Exercises", "References"]


def test_contract_check_accepts_numbered_markdown_headings(tmp_path: Path) -> None:
    checker = load_script_module("check_research_notebooks")
    notebook_path = tmp_path / "14_attention_as_kernel_operator.ipynb"
    write_notebook(
        notebook_path,
        [
            "## 1. Motivation",
            "## 5. Mathematical derivation",
            "## 6. PyTorch implementation",
            "## 7. Numerical checks",
            "## 12. Exercises",
            "## 13. References",
        ],
    )

    assert checker.collect_contract_failures(tmp_path) == []


def test_contract_check_accepts_explicit_notebook_paths(tmp_path: Path) -> None:
    checker = load_script_module("check_research_notebooks")
    notebook_path = tmp_path / "13_research_orientation.ipynb"
    write_notebook(
        notebook_path,
        [
            "# Motivation",
            "## Mathematical derivation",
            "## PyTorch implementation",
            "## Numerical checks",
            "## Exercises",
            "## References",
        ],
    )

    assert checker.collect_contract_failures(notebook_paths=[notebook_path]) == []
