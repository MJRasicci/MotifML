"""Pipeline definition for baseline decoder-only Transformer training."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.training.nodes import train_decoder_only_transformer


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the baseline training pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=train_decoder_only_transformer,
                inputs=[
                    "model_input_runtime",
                    "vocabulary",
                    "params:model",
                    "params:training",
                    "params:seed",
                ],
                outputs=[
                    "training_artifacts",
                    "training_history",
                    "training_run_metadata",
                ],
                name="train_decoder_only_transformer",
            )
        ],
        tags=["training", "modeling"],
    )


__all__ = ["create_pipeline"]
