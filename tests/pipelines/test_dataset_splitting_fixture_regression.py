"""Fixture-backed regression tests for split-planning artifacts."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import (
    load_json,
    run_session,
    write_test_conf,
)
from tests.pipelines.training_test_support import (
    TRAINING_FIXTURE_ROOT,
    materialize_training_fixture_corpus,
)


def test_dataset_splitting_outputs_match_tracked_training_fixtures(
    tmp_path: Path,
) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    run_session(conf_source, ["dataset_splitting"])

    assert load_json(output_root / "split_manifest.json") == load_json(
        TRAINING_FIXTURE_ROOT / "split_manifest.json"
    )
    assert load_json(output_root / "split_stats.json") == load_json(
        TRAINING_FIXTURE_ROOT / "split_stats.json"
    )
