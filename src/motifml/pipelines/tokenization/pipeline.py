"""Pipeline definitions for vocabulary-backed tokenization and model-input build."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.tokenization.nodes import (
    build_model_input_artifacts,
    build_shard_model_input_artifacts,
    count_training_split_tokens_from_model_input_parameters,
    reduce_vocabulary_from_count_artifact,
)


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the default end-to-end tokenization pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=count_training_split_tokens_from_model_input_parameters,
                inputs=[
                    "ir_features",
                    "split_manifest",
                    "params:sequence_schema",
                    "params:vocabulary",
                    "params:model_input",
                ],
                outputs="token_count_for_model_input",
                name="count_training_split_tokens_for_default_run",
            ),
            node(
                func=reduce_vocabulary_from_count_artifact,
                inputs=[
                    "token_count_for_model_input",
                    "params:vocabulary",
                    "params:data_split",
                ],
                outputs=["vocabulary", "vocab_stats", "vocabulary_version"],
                name="reduce_vocabulary_for_default_run",
            ),
            node(
                func=build_model_input_artifacts,
                inputs=[
                    "ir_features",
                    "split_manifest",
                    "params:sequence_schema",
                    "vocabulary",
                    "params:model_input",
                ],
                outputs=["model_input", "model_input_stats", "model_input_version"],
                name="build_model_input_artifacts",
            ),
        ],
        tags=["tokenization", "training_prep"],
    )


def create_shard_pipeline(**kwargs: object) -> Pipeline:
    """Create the shard-local model-input build pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=build_shard_model_input_artifacts,
                inputs=[
                    "ir_features_shard",
                    "split_manifest",
                    "params:sequence_schema",
                    "vocabulary",
                    "params:model_input",
                    "params:execution",
                ],
                outputs=[
                    "model_input_shard",
                    "model_input_stats_shard",
                    "model_input_version_shard",
                ],
                name="build_shard_model_input_artifacts",
            )
        ],
        tags=["tokenization", "training_prep", "partitioned"],
    )


__all__ = [
    "create_pipeline",
    "create_shard_pipeline",
]
