"""Sanity checks for Kedro pipeline registration."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from motifml.pipeline_registry import register_pipelines

EXPECTED_INGESTION_NODE_COUNT = 2
EXPECTED_IR_BUILD_NODE_COUNT = 12
EXPECTED_IR_VALIDATION_NODE_COUNT = 4
EXPECTED_NORMALIZATION_NODE_COUNT = 1
EXPECTED_FEATURE_EXTRACTION_NODE_COUNT = 1
EXPECTED_DEFAULT_NODE_ORDER = [
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
    "normalize_ir_corpus",
    "validate_ir_documents",
    "extract_features",
    "publish_ir_validation_report",
    "summarize_ir_corpus",
    "report_ir_scale_metrics",
]


def test_register_pipelines_exposes_project_pipelines():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert "ingestion" in pipelines
    assert "ir_build" in pipelines
    assert "ir_validation" in pipelines
    assert "normalization" in pipelines
    assert "feature_extraction" in pipelines
    assert "__default__" in pipelines
    assert len(pipelines["ingestion"].nodes) == EXPECTED_INGESTION_NODE_COUNT
    assert len(pipelines["ir_build"].nodes) == EXPECTED_IR_BUILD_NODE_COUNT
    assert len(pipelines["ir_validation"].nodes) == EXPECTED_IR_VALIDATION_NODE_COUNT
    assert len(pipelines["normalization"].nodes) == EXPECTED_NORMALIZATION_NODE_COUNT
    assert (
        len(pipelines["feature_extraction"].nodes)
        == EXPECTED_FEATURE_EXTRACTION_NODE_COUNT
    )


def test_register_pipelines_builds_the_default_pipeline_in_stage_order():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert [
        node.name for node in pipelines["__default__"].nodes
    ] == EXPECTED_DEFAULT_NODE_ORDER


def test_pipeline_inputs_and_outputs_match_the_registered_catalog_contract():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert pipelines["ir_build"].inputs() == {
        "params:ir_build_metadata",
        "raw_motif_json_corpus",
    }
    assert pipelines["ir_build"].all_outputs() >= {
        "motif_ir_corpus",
        "motif_ir_manifest",
    }

    assert pipelines["ir_validation"].inputs() == {
        "motif_ir_corpus",
        "motif_ir_manifest",
        "params:ir_validation",
    }
    assert pipelines["ir_validation"].outputs() == {
        "motif_ir_summary",
        "motif_ir_validation_report",
    }

    assert pipelines["normalization"].inputs() == {"motif_ir_corpus"}
    assert pipelines["normalization"].outputs() == {"normalized_ir_corpus"}

    assert pipelines["feature_extraction"].inputs() == {
        "normalized_ir_corpus",
        "params:feature_extraction",
    }
    assert pipelines["feature_extraction"].outputs() == {"ir_features"}
