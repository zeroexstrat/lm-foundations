from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"

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

REQUIRED_HEADINGS = [
    "Motivation",
    "Mathematical derivation",
    "PyTorch implementation",
    "Numerical checks",
    "Exercises",
    "References",
]

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<heading>.+?)\s*$")
NUMBER_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")


@dataclass(frozen=True)
class ContractFailure:
    notebook_name: str
    missing_headings: list[str]


def existing_research_notebook_paths(notebooks_dir: Path = NOTEBOOKS_DIR) -> list[Path]:
    return [notebooks_dir / name for name in RESEARCH_NOTEBOOKS if (notebooks_dir / name).exists()]


def explicit_notebook_paths(notebook_paths: list[Path]) -> list[Path]:
    return notebook_paths


def normalize_heading(heading: str) -> str:
    normalized = heading.strip()
    normalized = NUMBER_PREFIX_RE.sub("", normalized)
    return normalized.rstrip(":")


def extract_markdown_headings(notebook_path: Path) -> set[str]:
    notebook = nbformat.read(notebook_path, as_version=4)
    headings: set[str] = set()
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            match = HEADING_RE.match(line)
            if match:
                headings.add(normalize_heading(match.group("heading")))
    return headings


def collect_contract_failures(
    notebooks_dir: Path = NOTEBOOKS_DIR,
    notebook_paths: list[Path] | None = None,
) -> list[ContractFailure]:
    failures: list[ContractFailure] = []
    paths_to_check = (
        explicit_notebook_paths(notebook_paths)
        if notebook_paths is not None
        else existing_research_notebook_paths(notebooks_dir)
    )
    for notebook_path in paths_to_check:
        headings = extract_markdown_headings(notebook_path)
        missing = [heading for heading in REQUIRED_HEADINGS if heading not in headings]
        if missing:
            failures.append(
                ContractFailure(notebook_name=notebook_path.name, missing_headings=missing)
            )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Explicit notebook paths to scan.",
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=NOTEBOOKS_DIR,
        help="Notebook directory to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = collect_contract_failures(
        args.notebooks_dir,
        args.notebooks or None,
    )
    if not failures:
        return

    details = "\n".join(
        f"{failure.notebook_name}: missing headings: {', '.join(failure.missing_headings)}"
        for failure in failures
    )
    raise SystemExit(details)


if __name__ == "__main__":
    main()
