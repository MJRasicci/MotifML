"""Pipeline definition for the IR build stage."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ir_build.nodes import (
    assemble_ir_document,
    build_written_time_map,
    emit_bars,
    emit_intrinsic_edges,
    emit_note_events,
    emit_onset_groups,
    emit_parts_and_staves,
    emit_point_control_events,
    emit_span_control_events,
    emit_voice_lanes,
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
            node(
                func=emit_voice_lanes,
                inputs=[
                    "raw_motif_json_corpus",
                    "part_staff_emissions",
                    "bar_emissions",
                ],
                outputs="voice_lane_emissions",
                name="emit_voice_lanes",
            ),
            node(
                func=emit_onset_groups,
                inputs=[
                    "raw_motif_json_corpus",
                    "written_time_maps",
                    "voice_lane_emissions",
                ],
                outputs="onset_group_emissions",
                name="emit_onset_groups",
            ),
            node(
                func=emit_note_events,
                inputs=[
                    "raw_motif_json_corpus",
                    "written_time_maps",
                    "voice_lane_emissions",
                    "onset_group_emissions",
                ],
                outputs="note_event_emissions",
                name="emit_note_events",
            ),
            node(
                func=emit_intrinsic_edges,
                inputs=[
                    "raw_motif_json_corpus",
                    "part_staff_emissions",
                    "bar_emissions",
                    "voice_lane_emissions",
                    "onset_group_emissions",
                    "note_event_emissions",
                ],
                outputs="intrinsic_edge_emissions",
                name="emit_intrinsic_edges",
            ),
            node(
                func=emit_point_control_events,
                inputs=[
                    "raw_motif_json_corpus",
                    "written_time_maps",
                    "part_staff_emissions",
                    "voice_lane_emissions",
                ],
                outputs="point_control_emissions",
                name="emit_point_control_events",
            ),
            node(
                func=emit_span_control_events,
                inputs=[
                    "raw_motif_json_corpus",
                    "written_time_maps",
                    "part_staff_emissions",
                    "voice_lane_emissions",
                ],
                outputs="span_control_emissions",
                name="emit_span_control_events",
            ),
            node(
                func=assemble_ir_document,
                inputs=[
                    "raw_motif_json_corpus",
                    "part_staff_emissions",
                    "bar_emissions",
                    "voice_lane_emissions",
                    "onset_group_emissions",
                    "note_event_emissions",
                    "point_control_emissions",
                    "span_control_emissions",
                    "intrinsic_edge_emissions",
                    "params:ir_build_metadata",
                ],
                outputs="motif_ir_corpus",
                name="assemble_ir_document",
            ),
        ],
        tags=["ir_build"],
    )
