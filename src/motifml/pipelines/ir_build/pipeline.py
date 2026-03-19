"""Pipeline definition for the IR build stage."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ir_build.nodes import (
    build_written_time_map,
    emit_bars,
    emit_parts_and_staves,
    validate_canonical_score_surface,
)


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
            node(
                func=build_written_time_map,
                inputs=["raw_motif_json_corpus", "canonical_score_validation_results"],
                outputs="written_time_maps",
                name="build_written_time_map",
            ),
            node(
                func=emit_parts_and_staves,
                inputs=["raw_motif_json_corpus", "canonical_score_validation_results"],
                outputs="part_staff_emissions",
                name="emit_parts_and_staves",
            ),
            node(
                func=emit_bars,
                inputs=["raw_motif_json_corpus", "written_time_maps"],
                outputs="bar_emissions",
                name="emit_bars",
            ),
        ],
        tags=["ir_build"],
    )
