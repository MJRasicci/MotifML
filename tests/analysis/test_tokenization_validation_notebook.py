"""Execution coverage for the tokenization validation notebook."""

from __future__ import annotations

from tests.analysis.notebook_test_support import (
    NOTEBOOK_ROOT,
    TRAINING_FIXTURE_ROOT,
    execute_notebook,
    markdown_outputs,
)

NOTEBOOK_PATH = NOTEBOOK_ROOT / "tokenization_validation.ipynb"


def test_tokenization_validation_notebook_executes_for_validation_document(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOTIFML_TRAINING_FIXTURE_ROOT", str(TRAINING_FIXTURE_ROOT))
    monkeypatch.setenv("MOTIFML_TOKENIZATION_DOCUMENT", "ensemble_polyphony_controls")

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)

    assert any("Tokenization Overview" in output for output in markdown)
    assert any("Tokenization Trace" in output for output in markdown)
    assert any("Unknown Token Mapping" in output for output in markdown)
    assert any("Persisted Row Comparison" in output for output in markdown)
    assert any("ensemble_polyphony_controls.json" in output for output in markdown)
    assert any("Persisted Row Match: `True`" in output for output in markdown)


def test_tokenization_validation_notebook_executes_for_training_document(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOTIFML_TRAINING_FIXTURE_ROOT", str(TRAINING_FIXTURE_ROOT))
    monkeypatch.setenv(
        "MOTIFML_TOKENIZATION_DOCUMENT",
        "single_track_monophonic_pickup",
    )

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)

    assert any("single_track_monophonic_pickup.json" in output for output in markdown)
    assert any("Unknown Token Count: `0`" in output for output in markdown)


def test_tokenization_validation_notebook_executes_from_runtime_artifact_root(
    monkeypatch,
    training_runtime_artifact_root,
) -> None:
    monkeypatch.setenv(
        "MOTIFML_TRAINING_ARTIFACT_ROOT",
        str(training_runtime_artifact_root),
    )
    monkeypatch.setenv(
        "MOTIFML_TOKENIZATION_DOCUMENT",
        "single_track_monophonic_pickup",
    )

    executed = execute_notebook(NOTEBOOK_PATH)
    markdown = markdown_outputs(executed)

    assert any("Runtime Kedro Outputs" in output for output in markdown)
    assert any("single_track_monophonic_pickup.json" in output for output in markdown)
    assert any("Persisted Row Match: `True`" in output for output in markdown)
