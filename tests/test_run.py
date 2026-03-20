"""Sanity checks for Kedro pipeline registration."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from motifml.pipeline_registry import register_pipelines

EXPECTED_INGESTION_NODE_COUNT = 2
EXPECTED_IR_BUILD_NODE_COUNT = 12
EXPECTED_IR_VALIDATION_NODE_COUNT = 4
EXPECTED_NORMALIZATION_NODE_COUNT = 1


def test_register_pipelines_exposes_project_pipelines():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert "ingestion" in pipelines
    assert "ir_build" in pipelines
    assert "ir_validation" in pipelines
    assert "normalization" in pipelines
    assert "__default__" in pipelines
    assert len(pipelines["ingestion"].nodes) == EXPECTED_INGESTION_NODE_COUNT
    assert len(pipelines["ir_build"].nodes) == EXPECTED_IR_BUILD_NODE_COUNT
    assert len(pipelines["ir_validation"].nodes) == EXPECTED_IR_VALIDATION_NODE_COUNT
    assert len(pipelines["normalization"].nodes) == EXPECTED_NORMALIZATION_NODE_COUNT


def test_register_pipelines_builds_the_default_pipeline_in_stage_order():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    expected_node_order = (
        [node.name for node in pipelines["ingestion"].nodes]
        + ["stage_raw_corpus_for_ir_build"]
        + [node.name for node in pipelines["ir_build"].nodes]
        + [node.name for node in pipelines["normalization"].nodes]
        + [node.name for node in pipelines["ir_validation"].nodes]
    )

    assert [node.name for node in pipelines["__default__"].nodes] == expected_node_order


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
