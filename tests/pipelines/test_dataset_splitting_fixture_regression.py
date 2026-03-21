"""Fixture-backed regression tests for split-planning artifacts."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    load_json,
    run_session,
    write_test_conf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "training"


def test_dataset_splitting_outputs_match_tracked_training_fixtures(
    tmp_path: Path,
) -> None:
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    run_session(conf_source, ["dataset_splitting"])

    assert load_json(output_root / "split_manifest.json") == load_json(
        TRAINING_FIXTURE_ROOT / "split_manifest.json"
    )
    assert load_json(output_root / "split_stats.json") == load_json(
        TRAINING_FIXTURE_ROOT / "split_stats.json"
    )
