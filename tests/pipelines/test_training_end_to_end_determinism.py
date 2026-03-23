"""End-to-end determinism tests for training preparation and smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.pipelines.ir_test_support import (
    load_json,
    load_partitioned_record_set,
    load_tokenized_model_input,
    materialize_raw_fixture_subset,
    run_session,
    write_test_conf,
)
from tests.pipelines.training_test_support import (
    TRAINING_FIXTURE_ROOT,
    assert_nested_close,
    baseline_evaluation_runtime_overrides,
    baseline_training_runtime_overrides,
    build_normalized_smoke_bundle,
    training_fixture_raw_paths,
)


def test_training_preparation_outputs_are_stable_across_repeated_and_reordered_runs(
    tmp_path: Path,
) -> None:
    runtime_overrides = baseline_training_runtime_overrides()
    fixture_paths = list(training_fixture_raw_paths())

    forward_conf_source, forward_output_root = _write_training_fixture_conf(
        tmp_path / "forward",
        fixture_paths,
    )
    run_session(
        forward_conf_source,
        ["__default__"],
        runtime_params=runtime_overrides,
    )
    first_snapshot = _build_training_preparation_snapshot(forward_output_root)

    run_session(
        forward_conf_source,
        ["__default__"],
        runtime_params=runtime_overrides,
    )
    second_snapshot = _build_training_preparation_snapshot(forward_output_root)

    reverse_conf_source, reverse_output_root = _write_training_fixture_conf(
        tmp_path / "reverse",
        list(reversed(fixture_paths)),
    )
    run_session(
        reverse_conf_source,
        ["__default__"],
        runtime_params=runtime_overrides,
    )
    reversed_snapshot = _build_training_preparation_snapshot(reverse_output_root)

    assert second_snapshot == first_snapshot
    assert reversed_snapshot == first_snapshot

    version_keys = first_snapshot["version_keys"]
    tracked_split_manifest = load_json(TRAINING_FIXTURE_ROOT / "split_manifest.json")
    tracked_continuation_summary = load_json(
        TRAINING_FIXTURE_ROOT / "v1_continuation_summary.json"
    )
    tracked_vocabulary_version = load_json(
        TRAINING_FIXTURE_ROOT / "vocabulary_version.json"
    )
    tracked_model_input_version = load_json(
        TRAINING_FIXTURE_ROOT / "model_input" / "model_input_version.json"
    )

    assert version_keys["normalized_ir_version"]
    assert version_keys["feature_version"]
    assert (
        version_keys["sequence_schema_version"]
        == first_snapshot["feature_parameters"]["sequence_schema_version"]
    )
    assert version_keys["split_versions"] == sorted(
        {entry["split_version"] for entry in tracked_split_manifest}
    )
    assert (
        version_keys["continuation_dataset_version"]
        == tracked_continuation_summary["continuation_dataset_version"]
    )
    assert (
        version_keys["vocabulary_version"]
        == tracked_vocabulary_version["vocabulary_version"]
    )
    assert (
        version_keys["model_input_version"]
        == tracked_model_input_version["model_input_version"]
    )


def test_baseline_training_evaluation_smoke_bundle_is_deterministic(
    tmp_path: Path,
) -> None:
    runtime_overrides = baseline_evaluation_runtime_overrides()
    fixture_paths = list(training_fixture_raw_paths())

    forward_conf_source, forward_output_root = _write_training_fixture_conf(
        tmp_path / "forward_smoke",
        fixture_paths,
    )
    run_session(
        forward_conf_source,
        ["baseline_training_evaluation"],
        runtime_params=runtime_overrides,
    )
    first_bundle = build_normalized_smoke_bundle(
        forward_output_root,
        runtime_overrides=runtime_overrides,
    )
    _assert_smoke_artifact_families(forward_output_root)

    run_session(
        forward_conf_source,
        ["baseline_training_evaluation"],
        runtime_params=runtime_overrides,
    )
    second_bundle = build_normalized_smoke_bundle(
        forward_output_root,
        runtime_overrides=runtime_overrides,
    )

    reverse_conf_source, reverse_output_root = _write_training_fixture_conf(
        tmp_path / "reverse_smoke",
        list(reversed(fixture_paths)),
    )
    run_session(
        reverse_conf_source,
        ["baseline_training_evaluation"],
        runtime_params=runtime_overrides,
    )
    reversed_bundle = build_normalized_smoke_bundle(
        reverse_output_root,
        runtime_overrides=runtime_overrides,
    )
    _assert_smoke_artifact_families(reverse_output_root)

    assert_nested_close(second_bundle, first_bundle)
    assert_nested_close(reversed_bundle, first_bundle)


def _write_training_fixture_conf(
    tmp_path: Path,
    fixture_paths: list[str],
) -> tuple[Path, Path]:
    raw_root = materialize_raw_fixture_subset(tmp_path / "raw_training", fixture_paths)
    return write_test_conf(tmp_path, raw_root)


def _build_training_preparation_snapshot(output_root: Path) -> dict[str, Any]:
    split_manifest = load_json(output_root / "split_manifest.json")
    feature_parameters = load_partitioned_record_set(output_root / "ir_features")[
        "parameters"
    ]
    continuation_examples = load_partitioned_record_set(output_root / "v1_continuation")
    vocabulary_version = load_json(output_root / "vocabulary_version.json")
    model_input_version = load_json(
        output_root / "model_input" / "model_input_version.json"
    )
    model_input = load_tokenized_model_input(output_root / "model_input")

    return {
        "normalized_ir_version": load_json(output_root / "normalized_ir_version.json"),
        "split_manifest": split_manifest,
        "split_stats": load_json(output_root / "split_stats.json"),
        "continuation_examples": continuation_examples,
        "continuation_summary": load_json(output_root / "v1_continuation_summary.json"),
        "feature_parameters": feature_parameters,
        "vocabulary": load_json(output_root / "vocabulary.json"),
        "vocabulary_version": vocabulary_version,
        "vocab_stats": load_json(output_root / "vocab_stats.json"),
        "model_input_version": model_input_version,
        "model_input_stats": load_json(output_root / "model_input_stats.json"),
        "model_input_parameters": model_input["parameters"],
        "model_input_storage_schema": model_input["storage_schema"],
        "model_input_records": model_input["records"],
        "version_keys": {
            "normalized_ir_version": load_json(
                output_root / "normalized_ir_version.json"
            )["normalized_ir_version"],
            "feature_version": feature_parameters["feature_version"],
            "sequence_schema_version": feature_parameters["sequence_schema_version"],
            "split_versions": sorted(
                {entry["split_version"] for entry in split_manifest}
            ),
            "continuation_dataset_version": continuation_examples["parameters"][
                "continuation_dataset_version"
            ],
            "vocabulary_version": vocabulary_version["vocabulary_version"],
            "model_input_version": model_input_version["model_input_version"],
        },
    }


def _assert_smoke_artifact_families(output_root: Path) -> None:
    checkpoint_root = output_root / "training" / "baseline"

    assert (checkpoint_root / "checkpoint_manifest.json").exists()
    assert (checkpoint_root / "best_checkpoint.json").exists()
    assert list(sorted((checkpoint_root / "checkpoints").glob("*.pt")))
    assert (checkpoint_root / "model_config.json").exists()
    assert (checkpoint_root / "training_config.json").exists()
    assert (checkpoint_root / "run_metadata.json").exists()
    assert (output_root / "training_history.json").exists()
    assert (output_root / "training_run_metadata.json").exists()
    assert (output_root / "metrics.json").exists()
    assert (output_root / "evaluation_run_metadata.json").exists()
    assert (output_root / "qualitative_report.md").exists()
    assert (output_root / "evaluation" / "qualitative_samples.json").exists()
