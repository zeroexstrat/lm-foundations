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


def write_references_yaml(path: Path) -> None:
    path.write_text(
        "references:\n"
        "  attention_is_all_you_need:\n"
        "    title: Attention Is All You Need\n"
        "    authors: Vaswani et al.\n"
        "    year: 2017\n"
        "    primary_url: https://arxiv.org/abs/1706.03762\n"
        "    arxiv_id: '1706.03762'\n"
        "    modules: [R01]\n",
        encoding="utf-8",
    )


def test_reference_check_passes_when_no_research_notebooks_exist(tmp_path: Path) -> None:
    verifier = load_script_module("verify_research_references")
    references_path = tmp_path / "references.yaml"
    write_references_yaml(references_path)

    assert verifier.collect_reference_failures(tmp_path, references_path) == []


def test_reference_check_reports_unknown_citation_keys(tmp_path: Path) -> None:
    verifier = load_script_module("verify_research_references")
    references_path = tmp_path / "references.yaml"
    write_references_yaml(references_path)
    write_notebook(
        tmp_path / "13_research_orientation.ipynb",
        [
            "# Motivation",
            "We revisit exact attention [@attention_is_all_you_need] and a missing paper [@unknown_key].",
        ],
    )

    failures = verifier.collect_reference_failures(tmp_path, references_path)

    assert len(failures) == 1
    failure = failures[0]
    assert failure.notebook_name == "13_research_orientation.ipynb"
    assert failure.missing_keys == ["unknown_key"]


def test_reference_check_ignores_non_research_notebooks(tmp_path: Path) -> None:
    verifier = load_script_module("verify_research_references")
    references_path = tmp_path / "references.yaml"
    write_references_yaml(references_path)
    write_notebook(
        tmp_path / "00_project_orientation.ipynb",
        ["# Orientation", "This text mentions [@unknown_key] but is outside the research layer."],
    )

    assert verifier.collect_reference_failures(tmp_path, references_path) == []
