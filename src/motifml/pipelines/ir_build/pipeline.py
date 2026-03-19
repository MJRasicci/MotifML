"""Pipeline definition for the IR build stage."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ir_build.nodes import validate_canonical_score_surface


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the IR build pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=validate_canonical_score_surface,
                inputs="raw_motif_json_corpus",
                outputs="canonical_score_validation_results",
                name="validate_canonical_score_surface",
            ),
        ],
        tags=["ir_build"],
    )
