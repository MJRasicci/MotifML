"""Pipeline definition for IR feature extraction."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.feature_extraction.nodes import extract_features


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the feature extraction pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=extract_features,
                inputs=["normalized_ir_corpus", "params:feature_extraction"],
                outputs="ir_features",
                name="extract_features",
            )
        ],
        tags=["feature_extraction"],
    )
