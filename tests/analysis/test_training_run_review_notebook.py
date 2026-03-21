"""Execution coverage for the training-run review notebook."""

from __future__ import annotations

from tests.analysis.notebook_test_support import (
    NOTEBOOK_ROOT,
    TRAINING_FIXTURE_ROOT,
    execute_notebook,
    markdown_outputs,
    plotly_outputs,
)

NOTEBOOK_PATH = NOTEBOOK_ROOT / "training_run_review.ipynb"
EXPECTED_MINIMUM_PLOTLY_FIGURES = 2


def test_training_run_review_notebook_executes(monkeypatch) -> None:
    monkeypatch.setenv(
        "MOTIFML_TRAINING_SMOKE_BUNDLE_ROOT",
        str(TRAINING_FIXTURE_ROOT / "smoke_bundle"),
    )

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)
    plotly = plotly_outputs(executed)

    assert any("Training Run Overview" in output for output in markdown)
    assert any("Training Curves" in output for output in markdown)
    assert any("Evaluation Curves" in output for output in markdown)
    assert any("Qualitative Review" in output for output in markdown)
    assert any("Report Match: `True`" in output for output in markdown)
    assert any("ensemble_polyphony_controls.json" in output for output in markdown)
    assert len(plotly) >= EXPECTED_MINIMUM_PLOTLY_FIGURES
