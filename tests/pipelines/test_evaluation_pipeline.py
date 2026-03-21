"""Integration smoke tests for the baseline evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import (
    load_json,
    load_text,
    run_session,
    write_test_conf,
)
from tests.pipelines.training_test_support import (
    baseline_evaluation_runtime_overrides,
    materialize_training_fixture_corpus,
)


def test_evaluation_pipeline_persists_metrics_samples_and_report(
    tmp_path: Path,
) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)
    runtime_overrides = baseline_evaluation_runtime_overrides()

    run_session(conf_source, ["baseline_training"], runtime_params=runtime_overrides)
    run_session(conf_source, ["evaluation"], runtime_params=runtime_overrides)

    metrics = load_json(output_root / "metrics.json")
    samples = load_json(output_root / "evaluation" / "qualitative_samples.json")
    run_metadata = load_json(output_root / "evaluation_run_metadata.json")
    report = load_text(output_root / "qualitative_report.md")

    assert metrics["evaluation_run_id"]
    assert metrics["training_run_id"]
    assert set(metrics["splits"]) == {"validation"}
    validation_metrics = metrics["splits"]["validation"]
    assert validation_metrics["quantitative"]["token_count"] > 0
    assert "baseline_comparison" in validation_metrics
    assert "unknown_token_usage" in validation_metrics
    assert "generated_unknown_token_usage" in validation_metrics
    assert "structural" in validation_metrics
    assert validation_metrics["unknown_token_usage"]["unk_rate"] >= 0.0
    assert validation_metrics["generated_unknown_token_usage"]["unk_rate"] >= 0.0

    assert samples["evaluation_run_id"] == metrics["evaluation_run_id"]
    assert len(samples["samples"]["validation"]) == 1
    assert samples["samples"]["validation"][0]["generated_summary"]

    assert run_metadata["evaluation_run_id"] == metrics["evaluation_run_id"]
    assert run_metadata["evaluated_splits"] == ["validation"]

    assert "# Baseline Evaluation Report" in report
    assert "## Validation" in report
    assert "Generated Continuation" in report
