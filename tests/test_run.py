"""Sanity checks for Kedro pipeline registration."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from motifml.pipeline_registry import register_pipelines

EXPECTED_INGESTION_NODE_COUNT = 4
EXPECTED_IR_BUILD_NODE_COUNT = 12
EXPECTED_IR_VALIDATION_NODE_COUNT = 4
EXPECTED_NORMALIZATION_NODE_COUNT = 2
EXPECTED_DATASET_SPLITTING_NODE_COUNT = 2
EXPECTED_FEATURE_EXTRACTION_NODE_COUNT = 1
EXPECTED_TOKENIZATION_NODE_COUNT = 3
EXPECTED_VOCABULARY_COUNTING_NODE_COUNT = 1
EXPECTED_MODEL_INPUT_REDUCE_NODE_COUNT = 2
EXPECTED_TRAINING_NODE_COUNT = 1
EXPECTED_EVALUATION_NODE_COUNT = 1
EXPECTED_BASELINE_TRAINING_NODE_COUNT = 31
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
    "assign_dataset_splits",
    "build_normalized_ir_version",
    "publish_ir_validation_report",
    "summarize_ir_corpus",
    "build_split_statistics",
    "extract_features",
    "report_ir_scale_metrics",
    "count_training_split_tokens_for_default_run",
    "reduce_vocabulary_for_default_run",
    "build_model_input_artifacts",
]
EXPECTED_BASELINE_TRAINING_NODE_ORDER = [
    *EXPECTED_DEFAULT_NODE_ORDER,
    "stage_model_input_runtime_for_training",
    "train_decoder_only_transformer",
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
    assert "vocabulary_counting_shard" in pipelines
    assert "training" in pipelines
    assert "evaluation" in pipelines
    assert "baseline_training" in pipelines
    assert "tokenization_shard" in pipelines
    assert "partitioned_reduce" in pipelines
    assert "shard_reduce" in pipelines
    assert "model_input_reduce" in pipelines
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
    assert (
        len(pipelines["vocabulary_counting_shard"].nodes)
        == EXPECTED_VOCABULARY_COUNTING_NODE_COUNT
    )
    assert (
        len(pipelines["model_input_reduce"].nodes)
        == EXPECTED_MODEL_INPUT_REDUCE_NODE_COUNT
    )
    assert len(pipelines["training"].nodes) == EXPECTED_TRAINING_NODE_COUNT
    assert len(pipelines["evaluation"].nodes) == EXPECTED_EVALUATION_NODE_COUNT
    assert (
        len(pipelines["baseline_training"].nodes)
        == EXPECTED_BASELINE_TRAINING_NODE_COUNT
    )


def test_register_pipelines_builds_the_default_pipeline_in_stage_order():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert [
        node.name for node in pipelines["__default__"].nodes
    ] == EXPECTED_DEFAULT_NODE_ORDER


def test_register_pipelines_builds_the_baseline_training_pipeline_in_stage_order():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert [
        node.name for node in pipelines["baseline_training"].nodes
    ] == EXPECTED_BASELINE_TRAINING_NODE_ORDER


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
    assert pipelines["dataset_splitting"].all_outputs() >= {
        "split_manifest",
        "split_stats",
    }
    assert pipelines["dataset_splitting"].outputs() == {"split_stats"}

    assert pipelines["tokenization"].inputs() == {
        "ir_features",
        "split_manifest",
        "params:sequence_schema",
        "params:vocabulary",
        "params:model_input",
        "params:data_split",
    }
    assert pipelines["tokenization"].outputs() == {
        "vocab_stats",
        "vocabulary_version",
        "model_input",
        "model_input_stats",
        "model_input_version",
    }
    assert pipelines["tokenization"].all_outputs() >= {"vocabulary"}

    assert pipelines["training"].inputs() == {
        "model_input_runtime",
        "vocabulary",
        "params:model",
        "params:training",
        "params:seed",
    }
    assert pipelines["training"].outputs() == {
        "training_artifacts",
        "training_history",
        "training_run_metadata",
    }
    assert pipelines["evaluation"].inputs() == {
        "training_artifacts",
        "model_input_runtime",
        "vocabulary",
        "params:evaluation",
        "params:seed",
    }
    assert pipelines["evaluation"].outputs() == {
        "evaluation_samples",
        "evaluation_metrics",
        "qualitative_report",
        "evaluation_run_metadata",
    }
    assert "model_input_runtime" in pipelines["baseline_training"].inputs()
    assert pipelines["baseline_training"].all_outputs() >= {
        "training_artifacts",
        "training_history",
        "training_run_metadata",
    }

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
        "token_count_shard_collection",
        "params:vocabulary",
        "params:data_split",
    }
    assert pipelines["model_input_reduce"].inputs() == {
        "model_input_version_shard_collection",
        "model_input_stats_shard_collection",
    }
    assert pipelines["model_input_reduce"].outputs() == {
        "model_input_stats",
        "model_input_version",
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

    assert pipelines["vocabulary_counting_shard"].inputs() == {
        "ir_features_shard",
        "split_manifest",
        "params:sequence_schema",
        "params:vocabulary",
        "params:model_input",
    }
    assert pipelines["vocabulary_counting_shard"].outputs() == {"token_count_shard"}

    assert pipelines["tokenization_shard"].inputs() == {
        "ir_features_shard",
        "split_manifest",
        "vocabulary",
        "params:sequence_schema",
        "params:model_input",
        "params:execution",
    }
    assert pipelines["tokenization_shard"].outputs() == {
        "model_input_shard",
        "model_input_stats_shard",
        "model_input_version_shard",
    }
    assert pipelines["shard_processing"].outputs() >= {
        "motif_ir_summary_shard",
        "motif_ir_validation_report_shard",
        "ir_features_shard",
    }
