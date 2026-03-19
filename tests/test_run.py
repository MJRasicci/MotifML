"""Sanity checks for Kedro pipeline registration."""

from pathlib import Path

from kedro.framework.startup import bootstrap_project

from motifml.pipeline_registry import register_pipelines

EXPECTED_INGESTION_NODE_COUNT = 2
EXPECTED_IR_BUILD_NODE_COUNT = 2


def test_register_pipelines_exposes_project_pipelines():
    bootstrap_project(Path(__file__).resolve().parents[1])
    pipelines = register_pipelines()

    assert "ingestion" in pipelines
    assert "ir_build" in pipelines
    assert "__default__" in pipelines
    assert len(pipelines["ingestion"].nodes) == EXPECTED_INGESTION_NODE_COUNT
    assert len(pipelines["ir_build"].nodes) == EXPECTED_IR_BUILD_NODE_COUNT
