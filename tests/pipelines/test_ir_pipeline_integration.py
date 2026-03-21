"""End-to-end Kedro pipeline integration tests for IR build and validation."""

from __future__ import annotations

from pathlib import Path

from motifml.pipeline_registry import register_pipelines
from motifml.sharding import shard_ids_from_entries
from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    fixture_entries,
    load_json,
    load_partition_index,
    load_partitioned_record_set,
    run_session,
    write_test_conf,
)

EXPECTED_FIXTURE_COUNT = len(fixture_entries())
EXPECTED_DEFAULT_STAGE_ORDER = [
    "build_raw_corpus_manifest",
    "build_raw_partition_index",
    "summarize_raw_corpus",
    "build_raw_shard_manifests",
    "stage_raw_corpus_for_ir_build",
    "validate_canonical_score_surface",
    "build_written_time_map",
    "emit_parts_and_staves",
    "emit_bars",
    "emit_voice_lanes",
    "emit_onset_groups",
    "emit_point_control_events",
    "emit_span_control_events",
    "emit_note_events",
    "emit_intrinsic_edges",
    "assemble_ir_document",
    "build_ir_manifest",
    "normalize_ir_corpus",
    "validate_ir_documents",
    "build_normalized_ir_version",
    "publish_ir_validation_report",
    "summarize_ir_corpus",
    "extract_features",
    "report_ir_scale_metrics",
    "tokenize_features",
]


def test_kedro_session_runs_ir_build_and_ir_validation_in_isolation(tmp_path: Path):
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)
    run_session(conf_source, ["ir_build"])

    ir_documents = sorted((output_root / "documents").rglob("*.ir.json"))
    manifest = load_json(output_root / "motif_ir_manifest.json")

    assert len(ir_documents) == EXPECTED_FIXTURE_COUNT
    assert len(manifest) == EXPECTED_FIXTURE_COUNT
    assert not (output_root / "motif_ir_validation_report.json").exists()
    assert not (output_root / "motif_ir_summary.json").exists()

    run_session(conf_source, ["ir_validation"])

    validation_report = load_json(output_root / "motif_ir_validation_report.json")
    summary = load_json(output_root / "motif_ir_summary.json")

    assert len(validation_report) == EXPECTED_FIXTURE_COUNT
    assert summary["document_count"] == EXPECTED_FIXTURE_COUNT


def test_kedro_session_runs_default_pipeline_in_stage_sequence(tmp_path: Path):
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)
    run_session(conf_source)

    ir_features = load_partitioned_record_set(output_root / "ir_features")

    assert load_json(output_root / "raw_motif_json_manifest.json")
    assert load_json(output_root / "raw_motif_json_summary.json")
    assert load_json(output_root / "raw_partition_index.json")
    assert load_json(output_root / "raw_shard_manifests.json")
    assert load_json(output_root / "motif_ir_manifest.json")
    assert load_json(output_root / "motif_ir_validation_report.json")
    assert load_json(output_root / "motif_ir_summary.json")
    assert len(ir_features["records"]) == EXPECTED_FIXTURE_COUNT
    assert ir_features["parameters"]["feature_version"]
    assert ir_features["parameters"]["sequence_schema_version"]
    assert ir_features["parameters"]["normalized_ir_version"]
    assert len(load_partitioned_record_set(output_root / "model_input")["records"]) == (
        EXPECTED_FIXTURE_COUNT
    )
    assert sorted((output_root / "normalized_documents").rglob("*.ir.json"))
    assert load_json(output_root / "normalized_ir_version.json")[
        "normalized_ir_version"
    ]
    assert _default_stage_order() == EXPECTED_DEFAULT_STAGE_ORDER


def test_kedro_session_runs_partitioned_pipeline_flow(tmp_path: Path):
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ingestion"])
    partition_index = load_partition_index(output_root / "raw_partition_index.json")
    shard_ids = shard_ids_from_entries(partition_index)

    assert shard_ids

    for shard_id in shard_ids:
        run_session(
            conf_source,
            ["shard_processing"],
            runtime_params={"execution": {"shard_id": shard_id}},
        )

    run_session(conf_source, ["partitioned_reduce"])

    manifest = load_json(output_root / "motif_ir_manifest.json")
    validation_report = load_json(output_root / "motif_ir_validation_report.json")
    summary = load_json(output_root / "motif_ir_summary.json")
    ir_features = load_partitioned_record_set(output_root / "ir_features")

    assert len(manifest) == EXPECTED_FIXTURE_COUNT
    assert len(validation_report) == EXPECTED_FIXTURE_COUNT
    assert summary["document_count"] == EXPECTED_FIXTURE_COUNT
    assert sorted((output_root / "documents").rglob("*.ir.json"))
    assert sorted((output_root / "normalized_documents").rglob("*.ir.json"))
    assert load_json(output_root / "normalized_ir_version.json")[
        "normalized_ir_version"
    ]
    assert len(sorted((output_root / "ir_manifests").glob("*.json"))) == len(shard_ids)
    assert len(sorted((output_root / "normalized_ir_versions").glob("*.json"))) == len(
        shard_ids
    )
    assert len(sorted((output_root / "validation_shards").glob("*.json"))) == len(
        shard_ids
    )
    assert len(sorted((output_root / "summary_shards").glob("*.json"))) == len(
        shard_ids
    )
    assert len(ir_features["records"]) == EXPECTED_FIXTURE_COUNT
    assert ir_features["parameters"]["feature_version"]
    assert ir_features["parameters"]["sequence_schema_version"]
    assert ir_features["parameters"]["normalized_ir_version"]
    assert len(load_partitioned_record_set(output_root / "model_input")["records"]) == (
        EXPECTED_FIXTURE_COUNT
    )


def test_kedro_session_emits_stable_normalized_ir_version_for_unchanged_inputs(
    tmp_path: Path,
):
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    first_version = load_json(output_root / "normalized_ir_version.json")

    run_session(conf_source, ["normalization"])
    second_version = load_json(output_root / "normalized_ir_version.json")

    assert first_version == second_version


def test_normalization_stage_persists_training_safe_documents(tmp_path: Path):
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])

    forbidden_model_fields = {
        "attention_mask",
        "input_ids",
        "model_input_version",
        "padding_strategy",
        "split",
        "split_version",
        "target_ids",
        "token_count",
        "token_ids",
        "training_run_id",
        "vocabulary_version",
        "window_start_offsets",
    }

    for document_path in sorted(
        (output_root / "normalized_documents").rglob("*.ir.json")
    ):
        document_text = document_path.read_text(encoding="utf-8")
        for field_name in forbidden_model_fields:
            assert f'"{field_name}"' not in document_text


def test_feature_extraction_persists_stable_feature_artifacts_for_unchanged_inputs(
    tmp_path: Path,
):
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    run_session(conf_source, ["feature_extraction"])
    first_bytes = _load_partitioned_json_bytes(output_root / "ir_features")

    run_session(conf_source, ["feature_extraction"])
    second_bytes = _load_partitioned_json_bytes(output_root / "ir_features")

    assert first_bytes == second_bytes


def _default_stage_order() -> list[str]:
    return [node.name for node in register_pipelines()["__default__"].nodes]


def _load_partitioned_json_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*.json"))
        if item.is_file()
    }
