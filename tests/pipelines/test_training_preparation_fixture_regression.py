"""Fixture-backed regression tests for golden training-preparation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from motifml.datasets.tokenized_model_input_dataset import TokenizedModelInputDataset
from tests.pipelines.ir_test_support import load_json, run_session, write_test_conf
from tests.pipelines.training_test_support import (
    TRAINING_FIXTURE_ROOT,
    baseline_training_runtime_overrides,
    materialize_training_fixture_corpus,
)

TRACKED_JSON_OUTPUTS = (
    "split_manifest.json",
    "split_stats.json",
    "vocabulary.json",
    "vocabulary_version.json",
    "vocab_stats.json",
    "model_input_stats.json",
    "model_input/model_input_version.json",
    "model_input/parameters.json",
    "model_input/storage_schema.json",
)


def test_training_preparation_outputs_match_tracked_fixtures(tmp_path: Path) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(
        conf_source,
        ["__default__"],
        runtime_params=baseline_training_runtime_overrides(),
    )

    for relative_path in TRACKED_JSON_OUTPUTS:
        assert load_json(output_root / relative_path) == load_json(
            TRAINING_FIXTURE_ROOT / relative_path
        )

    actual_rows = TokenizedModelInputDataset(
        filepath=str(output_root / "model_input")
    ).load()["records"]
    assert actual_rows == _load_tracked_representative_rows()


def _load_tracked_representative_rows() -> list[dict[str, Any]]:
    row_root = TRAINING_FIXTURE_ROOT / "representative_rows"
    return [
        load_json(path)
        for path in sorted(row_root.rglob("*.row.json"))
        if path.is_file()
    ]
