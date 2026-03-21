"""Nodes for the feature extraction pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.projections.graph import GraphProjectionParameters, project_graph
from motifml.ir.projections.hierarchical import project_hierarchical
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    SequenceProjectionConfig,
    SequenceProjectionMode,
    SpanControlSequenceEvent,
    StructureMarkerSequenceEvent,
    project_sequence,
)
from motifml.pipelines.feature_extraction.models import (
    BASELINE_SEQUENCE_MODE,
    FeatureExtractionParameters,
    IrFeatureRecord,
    IrFeatureSet,
    ProjectionType,
    coerce_feature_extraction_parameters,
)
from motifml.training.sequence_schema import (
    SequenceSchemaContract,
    coerce_sequence_schema_contract,
)


def extract_features(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    parameters: FeatureExtractionParameters | Mapping[str, Any],
    sequence_schema: SequenceSchemaContract | Mapping[str, Any],
) -> IrFeatureSet:
    """Project each normalized IR document into the configured feature surface."""
    typed_parameters = coerce_feature_extraction_parameters(parameters)
    typed_sequence_schema = coerce_sequence_schema_contract(sequence_schema)
    records = tuple(
        IrFeatureRecord(
            relative_path=record.relative_path,
            projection_type=typed_parameters.projection_type,
            projection=_project_document(
                record,
                typed_parameters,
                typed_sequence_schema,
            ),
        )
        for record in normalized_ir_corpus
    )
    return IrFeatureSet(parameters=typed_parameters, records=records)


def merge_feature_shards(
    feature_shards: list[IrFeatureSet] | list[Mapping[str, Any]],
) -> IrFeatureSet:
    """Merge shard-local feature sets into one global feature set."""
    typed_shards = [_coerce_feature_set(shard) for shard in feature_shards]
    if not typed_shards:
        return IrFeatureSet(parameters=FeatureExtractionParameters())

    parameters = typed_shards[0].parameters
    for shard in typed_shards[1:]:
        if shard.parameters != parameters:
            raise ValueError("All feature shards must use identical parameters.")

    return IrFeatureSet(
        parameters=parameters,
        records=tuple(record for shard in typed_shards for record in shard.records),
    )


def _project_document(
    record: MotifIrDocumentRecord,
    parameters: FeatureExtractionParameters,
    sequence_schema: SequenceSchemaContract,
):
    if parameters.projection_type is ProjectionType.SEQUENCE:
        projection = project_sequence(
            record.document,
            SequenceProjectionConfig(
                mode=_sequence_projection_mode(parameters, sequence_schema)
            ),
        )
        return _apply_sequence_schema(projection, sequence_schema)

    if parameters.projection_type is ProjectionType.GRAPH:
        return project_graph(
            record.document,
            GraphProjectionParameters(
                derived_edge_types=parameters.derived_edge_families_included
            ),
        )

    return project_hierarchical(record.document)


def _coerce_feature_set(value: IrFeatureSet | Mapping[str, Any]) -> IrFeatureSet:
    if isinstance(value, IrFeatureSet):
        return value

    return IrFeatureSet(
        parameters=coerce_feature_extraction_parameters(value.get("parameters", {})),
        records=tuple(
            IrFeatureRecord(
                relative_path=str(record["relative_path"]),
                projection_type=ProjectionType(record["projection_type"]),
                projection=record.get("projection", {}),
            )
            for record in value.get("records", ())
        ),
    )


def _sequence_mode_for_event_types(
    event_types_included: tuple[str, ...],
) -> SequenceProjectionMode:
    normalized = {
        value.strip().casefold().replace(" ", "_").replace("-", "_")
        for value in event_types_included
    }
    if {
        "structure_markers",
        "notes+controls+structure_markers",
        "notes+controls+structuremarkers",
    } & normalized:
        return SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS

    if {"controls", "notes+controls"} & normalized:
        return SequenceProjectionMode.NOTES_AND_CONTROLS

    return SequenceProjectionMode.NOTES_ONLY


def _sequence_projection_mode(
    parameters: FeatureExtractionParameters,
    sequence_schema: SequenceSchemaContract,
) -> SequenceProjectionMode:
    if parameters.sequence_mode == BASELINE_SEQUENCE_MODE:
        return sequence_schema.projection_mode

    if parameters.event_types_included:
        return _sequence_mode_for_event_types(parameters.event_types_included)

    return SequenceProjectionMode(parameters.sequence_mode)


def _apply_sequence_schema(
    projection,
    sequence_schema: SequenceSchemaContract,
):
    filtered_events = tuple(
        event
        for event in projection.events
        if _sequence_event_is_enabled(event, sequence_schema)
    )
    return type(projection)(mode=projection.mode, events=filtered_events)


def _sequence_event_is_enabled(
    event: (
        NoteSequenceEvent
        | PointControlSequenceEvent
        | SpanControlSequenceEvent
        | StructureMarkerSequenceEvent
    ),
    sequence_schema: SequenceSchemaContract,
) -> bool:
    if isinstance(event, NoteSequenceEvent):
        return True

    if isinstance(event, StructureMarkerSequenceEvent):
        return (
            sequence_schema.structure_markers.enabled
            and event.marker_kind in sequence_schema.structure_markers.marker_kinds
        )

    if isinstance(event, PointControlSequenceEvent):
        return (
            sequence_schema.controls.include_point_controls
            and event.control.kind in sequence_schema.controls.point_control_kinds
        )

    return (
        sequence_schema.controls.include_span_controls
        and event.control.kind in sequence_schema.controls.span_control_kinds
    )
