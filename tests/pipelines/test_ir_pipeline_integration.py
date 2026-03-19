"""End-to-end Kedro pipeline integration tests for IR build and validation."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    fixture_entries,
    load_json,
    run_session,
    write_test_conf,
)

EXPECTED_FIXTURE_COUNT = len(fixture_entries())
EXPECTED_DEFAULT_STAGE_ORDER = [
    "build_raw_corpus_manifest",
    "summarize_raw_corpus",
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
    "validate_ir_documents",
    "publish_ir_validation_report",
    "summarize_ir_corpus",
    "report_ir_scale_metrics",
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

    assert load_json(output_root / "raw_motif_json_manifest.json")
    assert load_json(output_root / "raw_motif_json_summary.json")
    assert load_json(output_root / "motif_ir_manifest.json")
    assert load_json(output_root / "motif_ir_validation_report.json")
    assert load_json(output_root / "motif_ir_summary.json")
    assert _default_stage_order() == EXPECTED_DEFAULT_STAGE_ORDER


def _default_stage_order() -> list[str]:
    from motifml.pipeline_registry import register_pipelines

    return [node.name for node in register_pipelines()["__default__"].nodes]
