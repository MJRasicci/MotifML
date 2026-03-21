"""Shared helpers for executing and inspecting project notebooks in tests."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from motifml.datasets.tokenized_model_input_dataset import TokenizedModelInputDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"
TRAINING_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "training"
GOLDEN_IR_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "ir" / "golden"


def execute_notebook(path: Path) -> nbformat.NotebookNode:
    """Execute one notebook from its parent directory."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    return notebook


def collect_output_text(
    notebook: nbformat.NotebookNode,
    *,
    mime_type: str,
) -> list[str]:
    """Collect stringified outputs for one MIME type from executed notebook cells."""
    outputs: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            payload = data.get(mime_type)
            if payload is None:
                continue
            if isinstance(payload, str):
                outputs.append(payload)
                continue
            if isinstance(payload, list):
                outputs.append("".join(str(item) for item in payload))
                continue
            outputs.append(str(payload))
    return outputs


def markdown_outputs(notebook: nbformat.NotebookNode) -> list[str]:
    """Collect Markdown display outputs from one executed notebook."""
    return collect_output_text(notebook, mime_type="text/markdown")


def text_outputs(notebook: nbformat.NotebookNode) -> list[str]:
    """Collect plain-text display outputs from one executed notebook."""
    return collect_output_text(notebook, mime_type="text/plain")


def plotly_outputs(notebook: nbformat.NotebookNode) -> list[dict[str, object]]:
    """Collect Plotly JSON outputs from one executed notebook."""
    outputs: list[dict[str, object]] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            payload = data.get("application/vnd.plotly.v1+json")
            if isinstance(payload, dict):
                outputs.append(payload)
    return outputs


def build_model_input_runtime_fixture(tmp_path: Path) -> Path:
    """Materialize a tiny Parquet-backed model-input root from tracked rows."""
    representative_row_root = TRAINING_FIXTURE_ROOT / "representative_rows"
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(representative_row_root.rglob("*.row.json"))
        if path.is_file()
    ]
    model_input_root = tmp_path / "model_input"
    TokenizedModelInputDataset(filepath=str(model_input_root)).save(
        {
            "parameters": _load_json(
                TRAINING_FIXTURE_ROOT / "model_input" / "parameters.json"
            ),
            "storage_schema": _load_json(
                TRAINING_FIXTURE_ROOT / "model_input" / "storage_schema.json"
            ),
            "records": records,
            "shard_id": "global",
        }
    )
    return model_input_root


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "GOLDEN_IR_FIXTURE_ROOT",
    "NOTEBOOK_ROOT",
    "REPO_ROOT",
    "TRAINING_FIXTURE_ROOT",
    "build_model_input_runtime_fixture",
    "execute_notebook",
    "markdown_outputs",
    "plotly_outputs",
    "text_outputs",
]
