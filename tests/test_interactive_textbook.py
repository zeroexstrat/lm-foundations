from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def load_builder():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "build_interactive_textbook",
        SCRIPTS_DIR / "build_interactive_textbook.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_html_maps_navigation_to_real_chapter_ids() -> None:
    builder = load_builder()
    cells = [
        {"type": "markdown", "source": "## 0. First Chapter\n\nOpening text."},
        {"type": "markdown", "source": "## 1. Second Chapter\n\nClosing text."},
    ]

    html = builder.build_html(
        cells,
        css="",
        js=builder.build_js("[]"),
        katex_js_b64="",
    )

    assert 'const chStarts = [["first-chapter", 0], ["second-chapter", 1]];' in html
    assert "document.getElementById(slug)" in html
    assert "href === '#' + active" in html


def test_notebook_keeps_visible_space_glyph_outside_katex() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "99_complete_college_level_walkthrough.ipynb").read_text()
    )
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    math_expressions = re.findall(r"\$\$?(.+?)\$\$?", markdown, flags=re.DOTALL)

    assert all("␣" not in expression for expression in math_expressions)


def test_mobile_css_contains_content_overflow_guards() -> None:
    builder = load_builder()
    mobile_css = builder.build_css().split("@media (max-width: 1024px)", maxsplit=1)[1]

    assert "code { overflow-wrap: anywhere;" in mobile_css
    assert ".table-wrapper { overflow-x: auto;" in mobile_css


def test_visualization_sliders_have_programmatic_labels() -> None:
    builder = load_builder()
    containers = "".join(builder.VIZ_CONTAINERS.values())

    for control_id in (
        "temp-slider",
        "step-slider",
        "gen-temp",
        "quant-bits",
        "kv-l",
        "kv-h",
        "kv-d",
        "kv-t",
        "kv-g",
    ):
        assert f'for="{control_id}"' in containers


def test_toolbar_icon_buttons_have_accessible_names() -> None:
    builder = load_builder()
    html = builder.build_html([], css="", js="", katex_js_b64="")

    for label in (
        "Search textbook",
        "Decrease font size",
        "Increase font size",
        "Toggle color theme",
        "Back to top",
    ):
        assert f'aria-label="{label}"' in html


def test_artifact_counts_only_count_rendered_markup() -> None:
    builder = load_builder()
    assert hasattr(builder, "artifact_counts")
    html = """
    <style>.code-cell {} .viz-container {} .katex-inline {}</style>
    <li><a href="#chapter-one">Chapter one</a></li>
    <span class="katex-inline">x</span>
    <span class="katex-display">y</span>
    <div class="code-cell"><pre>code</pre></div>
    <div class="viz-container"></div>
    """

    assert builder.artifact_counts(html) == {
        "chapters": 1,
        "math": 2,
        "code_blocks": 1,
        "visualizations": 1,
    }


def test_committed_output_matches_in_memory_build() -> None:
    builder = load_builder()
    assert hasattr(builder, "render_textbook")
    assert builder.render_textbook() == (ROOT / "output" / "interactive_textbook.html").read_text()


def test_dependency_floor_matches_embedded_katex_version() -> None:
    package = json.loads((ROOT / "package.json").read_text())
    lock = json.loads((ROOT / "package-lock.json").read_text())
    locked_version = lock["packages"]["node_modules/katex"]["version"]

    assert package["dependencies"]["katex"] == f"^{locked_version}"
    assert lock["packages"][""]["dependencies"]["katex"] == f"^{locked_version}"
    assert f'version:"{locked_version}"' in (
        ROOT / "output" / "interactive_textbook.html"
    ).read_text()


def test_disclosure_controls_are_keyboard_and_screen_reader_accessible() -> None:
    builder = load_builder()
    builder.EQ_READINGS["x"] = "x"
    html = builder.build_html(
        [{"type": "markdown", "source": "Math $x$"}],
        css="",
        js=builder.build_js("[]"),
        katex_js_b64="",
    )

    assert '<button type="button" class="eq-reading-toggle"' in html
    assert 'aria-expanded="false"' in html
    assert "h3.tabIndex = 0" in html
    assert "h3.setAttribute('role', 'button')" in html
    assert "h3.setAttribute('aria-expanded', 'true')" in html
    assert "e.key === 'Enter' || e.key === ' '" in html


def test_visualization_canvases_have_accessible_names_and_keyboard_alternatives() -> None:
    builder = load_builder()
    containers = "".join(builder.VIZ_CONTAINERS.values())

    assert containers.count('<canvas ') == 8
    assert containers.count('role="img"') == 8
    assert containers.count('aria-label="') >= 8
    assert 'id="transformer-stage"' in containers
    assert 'id="embedding-token"' in containers
    assert 'id="attn-summary"' in containers
