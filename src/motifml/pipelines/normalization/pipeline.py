"""Pipeline definition for IR normalization."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.normalization.nodes import (
    build_normalized_ir_version,
    normalize_ir_corpus,
)


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the normalization pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=normalize_ir_corpus,
                inputs="motif_ir_corpus",
                outputs="normalized_ir_corpus",
                name="normalize_ir_corpus",
            ),
            node(
                func=build_normalized_ir_version,
                inputs=["normalized_ir_corpus", "params:normalization"],
                outputs="normalized_ir_version",
                name="build_normalized_ir_version",
            ),
        ],
        tags=["normalization"],
    )
