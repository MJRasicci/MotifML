"""Sanity checks for Kedro pipeline registration."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from motifml.pipeline_registry import register_pipelines

EXPECTED_INGESTION_NODE_COUNT = 4
EXPECTED_IR_BUILD_NODE_COUNT = 12
EXPECTED_IR_VALIDATION_NODE_COUNT = 4
EXPECTED_NORMALIZATION_NODE_COUNT = 2
EXPECTED_DATASET_SPLITTING_NODE_COUNT = 1
EXPECTED_FEATURE_EXTRACTION_NODE_COUNT = 1
EXPECTED_TOKENIZATION_NODE_COUNT = 1
EXPECTED_DEFAULT_NODE_ORDER = [
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


def test_register_pipelines_exposes_project_pipelines():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert "ingestion" in pipelines
    assert "partition_ingestion" in pipelines
    assert "partitioned_ingestion" in pipelines
    assert "ir_build" in pipelines
    assert "ir_build_shard" in pipelines
    assert "ir_validation" in pipelines
    assert "ir_validation_shard" in pipelines
    assert "normalization" in pipelines
    assert "normalization_shard" in pipelines
    assert "dataset_splitting" in pipelines
    assert "feature_extraction" in pipelines
    assert "feature_extraction_shard" in pipelines
    assert "tokenization" in pipelines
    assert "tokenization_shard" in pipelines
    assert "partitioned_reduce" in pipelines
    assert "shard_reduce" in pipelines
    assert "shard_processing" in pipelines
    assert "__default__" in pipelines
    assert len(pipelines["ingestion"].nodes) == EXPECTED_INGESTION_NODE_COUNT
    assert len(pipelines["ir_build"].nodes) == EXPECTED_IR_BUILD_NODE_COUNT
    assert len(pipelines["ir_validation"].nodes) == EXPECTED_IR_VALIDATION_NODE_COUNT
    assert len(pipelines["normalization"].nodes) == EXPECTED_NORMALIZATION_NODE_COUNT
    assert (
        len(pipelines["dataset_splitting"].nodes)
        == EXPECTED_DATASET_SPLITTING_NODE_COUNT
    )
    assert (
        len(pipelines["feature_extraction"].nodes)
        == EXPECTED_FEATURE_EXTRACTION_NODE_COUNT
    )
    assert len(pipelines["tokenization"].nodes) == EXPECTED_TOKENIZATION_NODE_COUNT


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

    assert pipelines["normalization"].inputs() == {
        "motif_ir_corpus",
        "params:normalization",
    }
    assert pipelines["normalization"].all_outputs() >= {
        "normalized_ir_corpus",
        "normalized_ir_version",
    }

    assert pipelines["feature_extraction"].inputs() == {
        "normalized_ir_corpus",
        "normalized_ir_version",
        "params:feature_extraction",
        "params:sequence_schema",
    }
    assert pipelines["feature_extraction"].outputs() == {"ir_features"}

    assert pipelines["dataset_splitting"].inputs() == {
        "normalized_ir_corpus",
        "params:data_split",
    }
    assert pipelines["dataset_splitting"].outputs() == {"split_manifest"}

    assert pipelines["tokenization"].inputs() == {
        "ir_features",
        "params:tokenization",
    }
    assert pipelines["tokenization"].outputs() == {"model_input"}

    assert pipelines["ingestion"].all_outputs() >= {
        "raw_motif_json_manifest",
        "raw_motif_json_summary",
        "raw_partition_index",
        "raw_shard_manifests",
    }

    assert pipelines["ir_build_shard"].inputs() == {
        "params:ir_build_metadata",
        "raw_motif_json_corpus_shard",
    }
    assert pipelines["ir_build_shard"].all_outputs() >= {
        "motif_ir_corpus_shard",
        "motif_ir_manifest_shard",
    }

    assert pipelines["ir_validation_shard"].inputs() == {
        "motif_ir_corpus_shard",
        "motif_ir_manifest_shard",
        "params:ir_validation",
    }
    assert pipelines["partitioned_reduce"].inputs() == {
        "motif_ir_manifest_shard_collection",
        "normalized_ir_version_shard_collection",
        "motif_ir_summary_shard_collection",
        "motif_ir_validation_report_shard_collection",
    }

    assert pipelines["normalization_shard"].inputs() == {
        "motif_ir_corpus_shard",
        "params:normalization",
    }
    assert pipelines["normalization_shard"].all_outputs() >= {
        "normalized_ir_corpus_shard",
        "normalized_ir_version_shard",
    }

    assert pipelines["feature_extraction_shard"].inputs() == {
        "normalized_ir_corpus_shard",
        "normalized_ir_version_shard",
        "params:feature_extraction",
        "params:sequence_schema",
    }
    assert pipelines["feature_extraction_shard"].outputs() == {"ir_features_shard"}

    assert pipelines["tokenization_shard"].inputs() == {
        "ir_features_shard",
        "params:tokenization",
    }
    assert pipelines["tokenization_shard"].outputs() == {"model_input_shard"}
    assert pipelines["shard_processing"].outputs() >= {
        "motif_ir_summary_shard",
        "motif_ir_validation_report_shard",
        "model_input_shard",
    }
