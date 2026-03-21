"""Pipeline definition for baseline decoder-only Transformer evaluation."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.evaluation.nodes import evaluate_decoder_only_transformer


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the baseline evaluation pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=evaluate_decoder_only_transformer,
                inputs=[
                    "training_artifacts",
                    "model_input_runtime",
                    "vocabulary",
                    "params:evaluation",
                    "params:seed",
                ],
                outputs=[
                    "evaluation_samples",
                    "evaluation_metrics",
                    "qualitative_report",
                    "evaluation_run_metadata",
                ],
                name="evaluate_decoder_only_transformer",
            )
        ],
        tags=["evaluation", "modeling"],
    )


__all__ = ["create_pipeline"]
