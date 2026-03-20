"""Public IR build node exports."""

from motifml.pipelines.ir_build.assembly_nodes import (
    assemble_ir_document,
    build_ir_manifest,
    merge_ir_manifest_fragments,
)
from motifml.pipelines.ir_build.control_nodes import (
    emit_point_control_events,
    emit_span_control_events,
)
from motifml.pipelines.ir_build.edge_nodes import emit_intrinsic_edges
from motifml.pipelines.ir_build.note_nodes import emit_note_events
from motifml.pipelines.ir_build.onset_nodes import emit_onset_groups
from motifml.pipelines.ir_build.structure_nodes import (
    emit_bars,
    emit_parts_and_staves,
    emit_voice_lanes,
)
from motifml.pipelines.ir_build.validation_nodes import (
    build_written_time_map,
    validate_canonical_score_surface,
)

__all__ = [
    "assemble_ir_document",
    "build_ir_manifest",
    "merge_ir_manifest_fragments",
    "build_written_time_map",
    "emit_bars",
    "emit_intrinsic_edges",
    "emit_note_events",
    "emit_onset_groups",
    "emit_parts_and_staves",
    "emit_point_control_events",
    "emit_span_control_events",
    "emit_voice_lanes",
    "validate_canonical_score_surface",
]
