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
    SequenceEvent,
    SequenceProjection,
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
from motifml.pipelines.normalization.models import (
    NormalizedIrVersionMetadata,
    coerce_normalized_ir_version_metadata,
)
from motifml.training.sequence_schema import (
    SequenceSchemaContract,
    coerce_sequence_schema_contract,
)
from motifml.training.versioning import build_feature_version

SEQUENCE_EVENT_ORDERING_VERSION = "time_then_family_then_entity_v1"
_SEQUENCE_SCHEMA_NOT_APPLICABLE = "not_applicable"


def extract_features(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    normalized_ir_version: NormalizedIrVersionMetadata | Mapping[str, Any],
    parameters: FeatureExtractionParameters | Mapping[str, Any],
    sequence_schema: SequenceSchemaContract | Mapping[str, Any],
) -> IrFeatureSet:
    """Project each normalized IR document into the configured feature surface."""
    typed_normalized_ir_version = coerce_normalized_ir_version_metadata(
        normalized_ir_version
    )
    typed_parameters = coerce_feature_extraction_parameters(parameters)
    typed_sequence_schema = coerce_sequence_schema_contract(sequence_schema)
    output_parameters = _build_output_parameters(
        typed_parameters,
        typed_normalized_ir_version,
        typed_sequence_schema,
    )
    records = tuple(
        IrFeatureRecord(
            relative_path=record.relative_path,
            projection_type=output_parameters.projection_type,
            projection=_project_document(
                record,
                output_parameters,
                typed_sequence_schema,
            ),
        )
        for record in normalized_ir_corpus
    )
    return IrFeatureSet(parameters=output_parameters, records=records)


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
        return _validate_sequence_projection_contract(
            _apply_sequence_schema(projection, sequence_schema)
        )

    if parameters.projection_type is ProjectionType.GRAPH:
        return project_graph(
            record.document,
            GraphProjectionParameters(
                derived_edge_types=parameters.derived_edge_families_included
            ),
        )

    return project_hierarchical(record.document)


def _build_output_parameters(
    parameters: FeatureExtractionParameters,
    normalized_ir_version: NormalizedIrVersionMetadata,
    sequence_schema: SequenceSchemaContract,
) -> FeatureExtractionParameters:
    sequence_schema_version = (
        sequence_schema.sequence_schema_version
        if parameters.projection_type is ProjectionType.SEQUENCE
        else _SEQUENCE_SCHEMA_NOT_APPLICABLE
    )
    return FeatureExtractionParameters(
        projection_type=parameters.projection_type,
        sequence_mode=parameters.sequence_mode,
        event_types_included=parameters.event_types_included,
        derived_edge_families_included=parameters.derived_edge_families_included,
        normalized_ir_version=normalized_ir_version.normalized_ir_version,
        sequence_schema_version=sequence_schema_version,
        feature_version=build_feature_version(
            normalized_ir_version=normalized_ir_version.normalized_ir_version,
            projection_config=_feature_projection_contract_payload(
                parameters,
                sequence_schema,
            ),
            sequence_schema_version=sequence_schema_version,
        ),
    )


def _feature_projection_contract_payload(
    parameters: FeatureExtractionParameters,
    sequence_schema: SequenceSchemaContract,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "projection_type": parameters.projection_type.value,
        "sequence_mode": parameters.sequence_mode,
        "event_types_included": list(parameters.event_types_included),
        "derived_edge_families_included": list(
            parameters.derived_edge_families_included
        ),
    }
    if parameters.projection_type is ProjectionType.SEQUENCE:
        payload["event_ordering_version"] = SEQUENCE_EVENT_ORDERING_VERSION
        payload["projection_mode"] = _sequence_projection_mode(
            parameters,
            sequence_schema,
        ).value
    else:
        payload["event_ordering_version"] = _SEQUENCE_SCHEMA_NOT_APPLICABLE
        payload["projection_mode"] = _SEQUENCE_SCHEMA_NOT_APPLICABLE
    return payload


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
    projection: SequenceProjection,
    sequence_schema: SequenceSchemaContract,
) -> SequenceProjection:
    filtered_events = tuple(
        event
        for event in projection.events
        if _sequence_event_is_enabled(event, sequence_schema)
    )
    return type(projection)(mode=projection.mode, events=filtered_events)


def _validate_sequence_projection_contract(
    projection: SequenceProjection,
) -> SequenceProjection:
    _validate_sequence_event_order(projection.events)
    return projection


def _validate_sequence_event_order(events: tuple[SequenceEvent, ...]) -> None:
    expected_events = tuple(sorted(events, key=_sequence_event_sort_key))
    if events == expected_events:
        return

    for index, (actual, expected) in enumerate(
        zip(events, expected_events, strict=True)
    ):
        if actual != expected:
            raise ValueError(
                "Sequence projection contract violation at event index "
                f"{index}: expected {type(expected).__name__} before "
                f"{type(actual).__name__} according to the canonical ordering."
            )


def _sequence_event_sort_key(event: SequenceEvent) -> tuple[object, ...]:
    return event.sort_key()


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
