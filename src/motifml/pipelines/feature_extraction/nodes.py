"""Nodes for the feature extraction pipeline skeleton."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.projections.graph import GraphProjectionParameters, project_graph
from motifml.ir.projections.hierarchical import project_hierarchical
from motifml.ir.projections.sequence import (
    SequenceProjectionConfig,
    SequenceProjectionMode,
    project_sequence,
)
from motifml.pipelines.feature_extraction.models import (
    FeatureExtractionParameters,
    IrFeatureRecord,
    IrFeatureSet,
    ProjectionType,
    coerce_feature_extraction_parameters,
)


def extract_features(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    parameters: FeatureExtractionParameters | Mapping[str, Any],
) -> IrFeatureSet:
    """Project each normalized IR document into the configured feature surface."""
    typed_parameters = coerce_feature_extraction_parameters(parameters)
    records = tuple(
        IrFeatureRecord(
            relative_path=record.relative_path,
            projection_type=typed_parameters.projection_type,
            projection=_project_document(record, typed_parameters),
        )
        for record in sorted(normalized_ir_corpus, key=lambda item: item.relative_path)
    )
    return IrFeatureSet(parameters=typed_parameters, records=records)


def _project_document(
    record: MotifIrDocumentRecord,
    parameters: FeatureExtractionParameters,
):
    if parameters.projection_type is ProjectionType.SEQUENCE:
        return project_sequence(
            record.document,
            SequenceProjectionConfig(
                mode=_sequence_mode_for_event_types(parameters.event_types_included)
            ),
        )

    if parameters.projection_type is ProjectionType.GRAPH:
        return project_graph(
            record.document,
            GraphProjectionParameters(
                derived_edge_types=parameters.derived_edge_families_included
            ),
        )

    return project_hierarchical(record.document)


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
