"""Fixture-backed regression tests for the normalized training smoke bundle."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import run_session, write_test_conf
from tests.pipelines.training_test_support import (
    assert_nested_close,
    baseline_evaluation_runtime_overrides,
    build_normalized_smoke_bundle,
    load_tracked_smoke_bundle,
    materialize_training_fixture_corpus,
)


def test_training_and_evaluation_outputs_match_tracked_smoke_bundle(
    tmp_path: Path,
) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)
    runtime_overrides = baseline_evaluation_runtime_overrides()

    run_session(
        conf_source,
        ["baseline_training"],
        runtime_params=runtime_overrides,
    )
    run_session(
        conf_source,
        ["evaluation"],
        runtime_params=runtime_overrides,
    )

    actual_bundle = build_normalized_smoke_bundle(
        output_root,
        runtime_overrides=runtime_overrides,
    )
    expected_bundle = load_tracked_smoke_bundle()

    for relative_path in (
        "frozen_config.json",
        "training_run_metadata.json",
        "evaluation_run_metadata.json",
        "evaluation/qualitative_samples.json",
    ):
        assert actual_bundle[relative_path] == expected_bundle[relative_path]

    for relative_path in ("training_history.json", "metrics.json"):
        assert_nested_close(
            actual_bundle[relative_path],
            expected_bundle[relative_path],
        )

    assert (
        actual_bundle["qualitative_report.md"]
        == expected_bundle["qualitative_report.md"]
    )
