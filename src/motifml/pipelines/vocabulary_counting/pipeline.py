"""Pipeline definition for shard-local vocabulary counting."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.tokenization.nodes import (
    count_training_split_tokens_from_model_input_parameters,
)


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the shard-local vocabulary counting pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=count_training_split_tokens_from_model_input_parameters,
                inputs=[
                    "ir_features_shard",
                    "split_manifest",
                    "params:sequence_schema",
                    "params:vocabulary",
                    "params:model_input",
                ],
                outputs="token_count_shard",
                name="count_training_split_tokens",
            )
        ],
        tags=["vocabulary", "tokenization"],
    )
