"""Integration smoke tests for the baseline evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    load_json,
    load_text,
    run_session,
    write_test_conf,
)


def test_evaluation_pipeline_persists_metrics_samples_and_report(
    tmp_path: Path,
) -> None:
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)
    runtime_overrides = _baseline_evaluation_runtime_overrides()

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
    assert "structural" in validation_metrics

    assert samples["evaluation_run_id"] == metrics["evaluation_run_id"]
    assert len(samples["samples"]["validation"]) == 1
    assert samples["samples"]["validation"][0]["generated_summary"]

    assert run_metadata["evaluation_run_id"] == metrics["evaluation_run_id"]
    assert run_metadata["evaluated_splits"] == ["validation"]

    assert "# Baseline Evaluation Report" in report
    assert "## Validation" in report
    assert "Generated Continuation" in report


def _baseline_evaluation_runtime_overrides() -> dict[str, object]:
    return {
        "data_split": {
            "ratios": {
                "train": 0.5,
                "validation": 0.25,
                "test": 0.25,
            },
            "hash_seed": 17,
            "grouping_strategy": "document_id",
            "grouping_key_fallback": "relative_path",
        },
        "model_input": {
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
            "context_length": 64,
            "stride": 32,
            "padding_strategy": "right",
            "special_token_policy": {
                "bos": "document",
                "eos": "document",
                "padding_interaction": "outside_boundaries",
                "unknown_token_mapping": "map_to_unk",
            },
            "storage": {
                "backend": "parquet",
                "schema_version": "parquet-v1",
            },
            "reporting": {
                "worst_document_limit": 10,
                "oversized_token_count_threshold": 8192,
            },
        },
        "model": {
            "architecture": "decoder_only_transformer",
            "embedding_dim": 32,
            "hidden_size": 64,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "positional_encoding": "learned",
        },
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "optimizer": "adamw",
            "lr_scheduler": {
                "name": "constant",
                "warmup_steps": 0,
            },
        },
        "evaluation": {
            "device": "cpu",
            "batch_size": 2,
            "top_k": 3,
            "decode_max_tokens": 4,
            "splits": ["validation"],
            "qualitative": {
                "samples_per_split": 1,
                "prompt_token_count": 4,
                "summary_token_limit": 3,
            },
        },
    }
