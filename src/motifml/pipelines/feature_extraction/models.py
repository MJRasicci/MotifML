"""Typed configuration and outputs for the feature extraction skeleton."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from motifml.ir.projections.graph import GraphProjection
from motifml.ir.projections.hierarchical import HierarchicalProjection
from motifml.ir.projections.sequence import SequenceProjection


class ProjectionType(StrEnum):
    """Supported projection families for skeleton feature extraction."""

    SEQUENCE = "sequence"
    GRAPH = "graph"
    HIERARCHICAL = "hierarchical"


ProjectionPayload: TypeAlias = (
    SequenceProjection | GraphProjection | HierarchicalProjection
)


@dataclass(frozen=True)
class FeatureExtractionParameters:
    """Configuration surface for the feature extraction pipeline skeleton."""

    projection_type: ProjectionType = ProjectionType.SEQUENCE
    event_types_included: tuple[str, ...] = ()
    derived_edge_families_included: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projection_type",
            ProjectionType(self.projection_type),
        )
        object.__setattr__(
            self,
            "event_types_included",
            _normalize_text_sequence(self.event_types_included, "event_types_included"),
        )
        object.__setattr__(
            self,
            "derived_edge_families_included",
            _normalize_text_sequence(
                self.derived_edge_families_included,
                "derived_edge_families_included",
            ),
        )


@dataclass(frozen=True)
class IrFeatureRecord:
    """One projected feature artifact for a source-relative IR document."""

    relative_path: str
    projection_type: ProjectionType
    projection: ProjectionPayload

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "projection_type",
            ProjectionType(self.projection_type),
        )


@dataclass(frozen=True)
class IrFeatureSet:
    """Collection of projected features emitted by the skeleton extractor."""

    parameters: FeatureExtractionParameters
    records: tuple[IrFeatureRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda item: item.relative_path.casefold())),
        )


def coerce_feature_extraction_parameters(
    value: FeatureExtractionParameters | Mapping[str, Any],
) -> FeatureExtractionParameters:
    """Coerce Kedro parameter mappings into the typed feature extraction config."""
    if isinstance(value, FeatureExtractionParameters):
        return value

    return FeatureExtractionParameters(
        projection_type=value.get("projection_type", ProjectionType.SEQUENCE),
        event_types_included=tuple(value.get("event_types_included", ())),
        derived_edge_families_included=tuple(
            value.get("derived_edge_families_included", ())
        ),
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


def _normalize_text_sequence(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    return tuple(_normalize_text(str(value), field_name) for value in values)
