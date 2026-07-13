from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
REFERENCES_PATH = ROOT / "docs" / "research" / "references.yaml"

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

CITATION_RE = re.compile(r"\[@(?P<key>[a-z0-9_]+)\]|(?<![\w/])@(?P<bare>[a-z0-9_]+)")


@dataclass(frozen=True)
class ReferenceFailure:
    notebook_name: str
    missing_keys: list[str]


def existing_research_notebook_paths(notebooks_dir: Path = NOTEBOOKS_DIR) -> list[Path]:
    return [notebooks_dir / name for name in RESEARCH_NOTEBOOKS if (notebooks_dir / name).exists()]


def load_reference_registry(references_path: Path = REFERENCES_PATH) -> dict[str, dict]:
    payload = yaml.safe_load(references_path.read_text(encoding="utf-8")) or {}
    references = payload.get("references", {})
    if not isinstance(references, dict):
        raise ValueError(f"invalid references registry in {references_path}")
    return references


def extract_citation_keys(notebook_path: Path) -> list[str]:
    notebook = nbformat.read(notebook_path, as_version=4)
    keys: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for match in CITATION_RE.finditer(cell.source):
            key = match.group("key") or match.group("bare")
            if key is not None:
                keys.append(key)
    return keys


def collect_reference_failures(
    notebooks_dir: Path = NOTEBOOKS_DIR,
    references_path: Path = REFERENCES_PATH,
) -> list[ReferenceFailure]:
    known_keys = load_reference_registry(references_path)
    failures: list[ReferenceFailure] = []
    for notebook_path in existing_research_notebook_paths(notebooks_dir):
        missing = sorted({key for key in extract_citation_keys(notebook_path) if key not in known_keys})
        if missing:
            failures.append(ReferenceFailure(notebook_name=notebook_path.name, missing_keys=missing))
    return failures


def verify_reference_urls_live(references_path: Path = REFERENCES_PATH, timeout: int = 10) -> list[str]:
    failures: list[str] = []
    for key, metadata in load_reference_registry(references_path).items():
        url = metadata.get("primary_url")
        if not url:
            failures.append(f"{key}: missing primary_url")
            continue
        try:
            with urlopen(url, timeout=timeout):
                pass
        except URLError as exc:
            failures.append(f"{key}: {url} ({exc.reason})")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=NOTEBOOKS_DIR,
        help="Notebook directory to scan.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=REFERENCES_PATH,
        help="Reference registry YAML file.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also verify that each primary URL is reachable over the network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_failures = collect_reference_failures(args.notebooks_dir, args.references)
    live_failures = verify_reference_urls_live(args.references) if args.live else []
    if not reference_failures and not live_failures:
        return

    details: list[str] = []
    for failure in reference_failures:
        details.append(f"{failure.notebook_name}: unknown citation keys: {', '.join(failure.missing_keys)}")
    details.extend(live_failures)
    raise SystemExit("\n".join(details))


if __name__ == "__main__":
    main()
