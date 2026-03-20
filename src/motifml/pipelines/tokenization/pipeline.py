"""Pipeline definition for placeholder IR tokenization."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.tokenization.nodes import tokenize_features


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the tokenization pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=tokenize_features,
                inputs=["ir_features", "params:tokenization"],
                outputs="model_input",
                name="tokenize_features",
            )
        ],
        tags=["tokenization"],
    )
