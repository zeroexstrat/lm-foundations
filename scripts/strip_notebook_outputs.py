from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def strip_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop("widgets", None)
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["notebooks"])
    args = parser.parse_args()
    for raw_path in args.paths:
        path = Path(raw_path)
        notebooks = [path] if path.is_file() else sorted(path.glob("*.ipynb"))
        for notebook_path in notebooks:
            strip_notebook(notebook_path)


if __name__ == "__main__":
    main()
