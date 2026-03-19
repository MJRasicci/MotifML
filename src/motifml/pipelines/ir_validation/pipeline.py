"""Pipeline definition for IR validation."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ir_validation.nodes import validate_ir_documents


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the IR validation pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=validate_ir_documents,
                inputs=["motif_ir_corpus", "params:ir_validation"],
                outputs="motif_ir_validation_report",
                name="validate_ir_documents",
            )
        ],
        tags=["ir_validation"],
    )
