#!/usr/bin/env python3
"""Export notebook 99 to a print-ready PDF.

The exporter intentionally uses nbconvert's LaTeX path for native math rendering,
then patches the generated TeX so long code/path spans wrap cleanly on paper.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "99_complete_college_level_walkthrough.ipynb"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "language-models-from-first-principles.pdf"
DEFAULT_WORKDIR = PROJECT_ROOT / "tmp" / "pdfs" / "notebook99_print_work"
DEFAULT_TITLE = "Language Models From First Principles"


def find_pandoc(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None

    found = shutil.which("pandoc")
    if found:
        return Path(found)

    try:
        import pypandoc  # type: ignore

        path = Path(pypandoc.get_pandoc_path())
        return path if path.exists() else None
    except Exception:
        pass

    for candidate in (
        Path.home() / "chema-env/lib/python3.12/site-packages/pypandoc/files/pandoc",
        PROJECT_ROOT / ".venv/lib/python3.11/site-packages/pypandoc/files/pandoc",
    ):
        if candidate.exists():
            return candidate

    return None


def run(cmd: list[str], cwd: Path, env_path_prefix: Path | None = None) -> None:
    env = None
    if env_path_prefix:
        import os

        env = os.environ.copy()
        env["PATH"] = f"{env_path_prefix}{os.pathsep}{env['PATH']}"
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def pathify_long_texttt(tex: str) -> str:
    pattern = re.compile(r"\\texttt\{([^{}]*)\}")

    def replace(match: re.Match[str]) -> str:
        content = match.group(1)
        raw = (
            content.replace(r"\_", "_")
            .replace(r"\#", "#")
            .replace(r"\%", "%")
            .replace(r"\&", "&")
        )
        only_known_escapes = re.sub(r"\\[_#%&]", "", content)
        if "\\" in only_known_escapes:
            return match.group(0)
        if "/" in raw or "_" in raw or len(raw) >= 24:
            delimiter = "|" if "|" not in raw else "!"
            return rf"\path{delimiter}{raw}{delimiter}"
        return match.group(0)

    return pattern.sub(replace, tex)


def patch_latex(tex_path: Path, title: str) -> None:
    tex = tex_path.read_text()

    tex = tex.replace(
        r"\title{99\_complete\_college\_level\_walkthrough}",
        rf"\title{{{title}}}" + "\n    " + r"\author{}",
    )
    tex = tex.replace(
        r"\geometry{verbose,tmargin=1in,bmargin=1in,lmargin=1in,rmargin=1in}",
        r"\geometry{letterpaper,tmargin=0.8in,bmargin=0.85in,lmargin=0.8in,rmargin=0.8in}",
    )
    tex = tex.replace(
        r"\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\}}",
        r"\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},fontsize=\footnotesize}",
    )
    tex = tex.replace(
        r"\begin{Verbatim}[commandchars=\\\{\}]",
        r"\begin{Verbatim}[commandchars=\\\{\},fontsize=\footnotesize]",
    )
    tex = tex.replace(
        "boxrule=1pt, pad at break*=1mm,colback=cellbackground, colframe=cellborder",
        "boxrule=0.45pt, arc=0.5mm, left=1mm, right=1mm, top=1mm, bottom=1mm, "
        "pad at break*=1mm,colback=cellbackground, colframe=cellborder",
    )
    tex = tex.replace(
        r"\usepackage{mathrsfs}" + "\n",
        r"\usepackage{mathrsfs}" + "\n"
        r"    \newcounter{none}" + "\n"
        r"    \setlength{\emergencystretch}{3em}" + "\n",
    )
    tex = pathify_long_texttt(tex)

    tex_path.write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--pandoc-path", help="Optional explicit path to pandoc.")
    args = parser.parse_args()

    notebook = args.notebook.resolve()
    output = args.output.resolve()
    workdir = args.workdir.resolve()
    pandoc = find_pandoc(args.pandoc_path)
    if pandoc is None:
        raise SystemExit(
            "pandoc was not found. Install pandoc or pass --pandoc-path to a local pandoc binary."
        )

    workdir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    base = "notebook99_print"
    run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "latex",
            "--no-prompt",
            str(notebook),
            "--output-dir",
            str(workdir),
            "--output",
            base,
        ],
        cwd=PROJECT_ROOT,
        env_path_prefix=pandoc.parent,
    )

    tex_path = workdir / f"{base}.tex"
    patch_latex(tex_path, args.title)

    for _ in range(2):
        run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=workdir)

    built_pdf = workdir / f"{base}.pdf"
    shutil.copy2(built_pdf, output)

    log_path = workdir / f"{base}.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    overfull_count = log_text.count("Overfull \\hbox")

    print(f"wrote {output}")
    print(f"source_tex {tex_path}")
    print(f"pandoc {pandoc}")
    print(f"overfull_hbox_count {overfull_count}")


if __name__ == "__main__":
    main()
