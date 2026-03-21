"""Execution coverage for the training failure-analysis notebook."""

from __future__ import annotations

from tests.analysis.notebook_test_support import (
    NOTEBOOK_ROOT,
    TRAINING_FIXTURE_ROOT,
    execute_notebook,
    markdown_outputs,
)

NOTEBOOK_PATH = NOTEBOOK_ROOT / "training_failure_analysis.ipynb"


def test_training_failure_analysis_notebook_executes(monkeypatch) -> None:
    monkeypatch.setenv("MOTIFML_TRAINING_FIXTURE_ROOT", str(TRAINING_FIXTURE_ROOT))
    monkeypatch.setenv(
        "MOTIFML_TRAINING_SMOKE_BUNDLE_ROOT",
        str(TRAINING_FIXTURE_ROOT / "smoke_bundle"),
    )

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)

    assert any("Failure Mode Overview" in output for output in markdown)
    assert any("Unknown Token Review" in output for output in markdown)
    assert any("Vocabulary Coverage" in output for output in markdown)
    assert any("Structural Drift" in output for output in markdown)
    assert any("Document Pathologies" in output for output in markdown)
    assert any("transition" in output for output in markdown)
    assert any("ensemble_polyphony_controls.json" in output for output in markdown)
