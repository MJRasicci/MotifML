"""Execution coverage for the IR inspection notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ir_inspection.ipynb"
GOLDEN_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "ir" / "golden"
EXPECTED_NOTEBOOK_SVG_COUNT = 4


def test_ir_inspection_notebook_executes_for_single_fixture(monkeypatch) -> None:
    monkeypatch.setenv(
        "MOTIFML_IR_INPUT_PATH",
        str(GOLDEN_FIXTURE_ROOT / "ensemble_polyphony_controls.ir.json"),
    )
    executed = _execute_notebook()

    markdown_outputs = _markdown_outputs(executed)
    svg_outputs = _svg_outputs(executed)
    assert any("Loaded Document" in output for output in markdown_outputs)
    assert any("Structure Rollup" in output for output in markdown_outputs)
    assert any("Voice-Lane Onset Tables" in output for output in markdown_outputs)
    assert any("Control Events" in output for output in markdown_outputs)
    assert len(svg_outputs) == EXPECTED_NOTEBOOK_SVG_COUNT
    assert any("IR Timeline Plot" in output for output in svg_outputs)


def test_ir_inspection_notebook_executes_for_corpus_member(monkeypatch) -> None:
    monkeypatch.setenv("MOTIFML_IR_INPUT_PATH", str(GOLDEN_FIXTURE_ROOT))
    monkeypatch.setenv("MOTIFML_IR_MEMBER", "single_track_monophonic_pickup")
    executed = _execute_notebook()

    markdown_outputs = _markdown_outputs(executed)
    svg_outputs = _svg_outputs(executed)
    assert any(
        "single_track_monophonic_pickup" in output for output in markdown_outputs
    )
    assert any("Onset Note Tables" in output for output in markdown_outputs)
    assert any("tie_origin" in output for output in markdown_outputs)
    assert any("Note Relation Graph" in output for output in svg_outputs)


def _execute_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    return notebook


def _markdown_outputs(notebook: nbformat.NotebookNode) -> list[str]:
    outputs: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        for output in cell.get("outputs", []):
            data = output.get("data", {})
            markdown = data.get("text/markdown")
            if markdown is None:
                continue

            if isinstance(markdown, str):
                outputs.append(markdown)
                continue

            outputs.append("".join(markdown))

    return outputs


def _svg_outputs(notebook: nbformat.NotebookNode) -> list[str]:
    outputs: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        for output in cell.get("outputs", []):
            data = output.get("data", {})
            svg = data.get("image/svg+xml")
            if svg is None:
                continue

            if isinstance(svg, str):
                outputs.append(svg)
                continue

            outputs.append("".join(svg))

    return outputs
