"""Execution coverage for the model-input inspection notebook."""

from __future__ import annotations

from tests.analysis.notebook_test_support import (
    NOTEBOOK_ROOT,
    TRAINING_FIXTURE_ROOT,
    build_model_input_runtime_fixture,
    execute_notebook,
    markdown_outputs,
)

NOTEBOOK_PATH = NOTEBOOK_ROOT / "model_input_inspection.ipynb"


def test_model_input_inspection_notebook_executes_from_tracked_snapshots(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOTIFML_TRAINING_FIXTURE_ROOT", str(TRAINING_FIXTURE_ROOT))
    monkeypatch.setenv(
        "MOTIFML_MODEL_INPUT_DOCUMENT",
        "ensemble_polyphony_controls.json",
    )

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)

    assert any("Model-Input Overview" in output for output in markdown)
    assert any("Split Manifest" in output for output in markdown)
    assert any("Vocabulary Summary" in output for output in markdown)
    assert any("Tokenized Document Distribution" in output for output in markdown)
    assert any("Window Reconstruction" in output for output in markdown)
    assert any("ensemble_polyphony_controls.json" in output for output in markdown)


def test_model_input_inspection_notebook_executes_from_runtime_dataset(
    monkeypatch,
    tmp_path,
) -> None:
    model_input_root = build_model_input_runtime_fixture(tmp_path)
    monkeypatch.setenv("MOTIFML_TRAINING_FIXTURE_ROOT", str(TRAINING_FIXTURE_ROOT))
    monkeypatch.setenv("MOTIFML_MODEL_INPUT_ROOT", str(model_input_root))
    monkeypatch.setenv(
        "MOTIFML_MODEL_INPUT_DOCUMENT",
        "single_track_monophonic_pickup.json",
    )

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)

    assert any("Runtime Dataset" in output for output in markdown)
    assert any("single_track_monophonic_pickup.json" in output for output in markdown)
    assert any("Storage Schema" in output for output in markdown)
