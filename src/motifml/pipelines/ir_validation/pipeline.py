"""Pipeline definition for IR validation."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ir_validation.nodes import (
    report_ir_scale_metrics,
    summarize_ir_corpus,
    validate_ir_documents,
)


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
            ),
            node(
                func=summarize_ir_corpus,
                inputs=[
                    "motif_ir_corpus",
                    "motif_ir_validation_report",
                    "motif_ir_manifest",
                ],
                outputs="ir_corpus_summary_model",
                name="summarize_ir_corpus",
            ),
            node(
                func=report_ir_scale_metrics,
                inputs="ir_corpus_summary_model",
                outputs="motif_ir_summary",
                name="report_ir_scale_metrics",
            ),
        ],
        tags=["ir_validation"],
    )
