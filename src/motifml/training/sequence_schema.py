"""Typed baseline sequence-schema contract for training-facing feature extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from motifml.ir.models import PointControlKind, SpanControlKind
from motifml.ir.projections.sequence import SequenceProjectionMode, StructureMarkerKind
from motifml.training.versioning import build_contract_version


class SequenceEventFamily(StrEnum):
    """Supported projected event families for the baseline sequence contract."""

    NOTE = "note"
    STRUCTURE_MARKER = "structure_marker"
    POINT_CONTROL = "point_control"
    SPAN_CONTROL = "span_control"


class NotePayloadField(StrEnum):
    """Optional note-local payload fields available to the baseline contract."""

    PITCH = "pitch"
    DURATION = "duration"
    STRING_NUMBER = "string_number"
    VELOCITY = "velocity"


@dataclass(frozen=True, slots=True)
class StructureMarkerPolicy:
    """Structure-marker inclusion policy for the baseline sequence schema."""

    enabled: bool = True
    marker_kinds: tuple[StructureMarkerKind, ...] = field(
        default_factory=lambda: tuple(StructureMarkerKind)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        marker_kinds = tuple(StructureMarkerKind(kind) for kind in self.marker_kinds)
        if self.enabled and not marker_kinds:
            raise ValueError(
                "marker_kinds must contain at least one marker when structure markers are enabled."
            )
        if not self.enabled:
            marker_kinds = ()
        object.__setattr__(self, "marker_kinds", marker_kinds)


@dataclass(frozen=True, slots=True)
class ControlFamilyPolicy:
    """Control-family inclusion policy for the baseline sequence schema."""

    include_point_controls: bool = True
    point_control_kinds: tuple[PointControlKind, ...] = field(
        default_factory=lambda: tuple(PointControlKind)
    )
    include_span_controls: bool = True
    span_control_kinds: tuple[SpanControlKind, ...] = field(
        default_factory=lambda: tuple(SpanControlKind)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "include_point_controls", bool(self.include_point_controls)
        )
        object.__setattr__(
            self, "include_span_controls", bool(self.include_span_controls)
        )
        point_control_kinds = tuple(
            PointControlKind(kind) for kind in self.point_control_kinds
        )
        span_control_kinds = tuple(
            SpanControlKind(kind) for kind in self.span_control_kinds
        )
        if self.include_point_controls and not point_control_kinds:
            raise ValueError(
                "point_control_kinds must contain at least one kind when point controls are enabled."
            )
        if self.include_span_controls and not span_control_kinds:
            raise ValueError(
                "span_control_kinds must contain at least one kind when span controls are enabled."
            )
        if not self.include_point_controls:
            point_control_kinds = ()
        if not self.include_span_controls:
            span_control_kinds = ()
        object.__setattr__(self, "point_control_kinds", point_control_kinds)
        object.__setattr__(self, "span_control_kinds", span_control_kinds)


@dataclass(frozen=True, slots=True)
class SequenceSchemaContract:
    """Frozen configuration surface for the baseline sequence projection contract."""

    schema_name: str = "baseline_sequence"
    schema_mode: str = "baseline_v1"
    note_payload_fields: tuple[NotePayloadField, ...] = (
        NotePayloadField.PITCH,
        NotePayloadField.DURATION,
    )
    structure_markers: StructureMarkerPolicy = field(
        default_factory=StructureMarkerPolicy
    )
    controls: ControlFamilyPolicy = field(default_factory=ControlFamilyPolicy)
    sequence_schema_version: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_name", _normalize_text(self.schema_name, "schema_name")
        )
        object.__setattr__(
            self, "schema_mode", _normalize_text(self.schema_mode, "schema_mode")
        )
        object.__setattr__(
            self,
            "note_payload_fields",
            tuple(
                NotePayloadField(field_name) for field_name in self.note_payload_fields
            ),
        )
        if not self.note_payload_fields:
            raise ValueError("note_payload_fields must contain at least one field.")
        structure_policy = self.structure_markers
        if not isinstance(structure_policy, StructureMarkerPolicy):
            structure_policy = StructureMarkerPolicy(**dict(structure_policy))
        object.__setattr__(self, "structure_markers", structure_policy)
        control_policy = self.controls
        if not isinstance(control_policy, ControlFamilyPolicy):
            control_policy = ControlFamilyPolicy(**dict(control_policy))
        object.__setattr__(self, "controls", control_policy)
        object.__setattr__(
            self,
            "sequence_schema_version",
            build_contract_version(
                namespace="sequence_schema",
                payload=self.to_version_payload(),
            ),
        )

    @property
    def enabled_event_families(self) -> tuple[SequenceEventFamily, ...]:
        """Return the event families enabled by the current schema policy."""
        families = [SequenceEventFamily.NOTE]
        if self.structure_markers.enabled:
            families.append(SequenceEventFamily.STRUCTURE_MARKER)
        if self.controls.include_point_controls:
            families.append(SequenceEventFamily.POINT_CONTROL)
        if self.controls.include_span_controls:
            families.append(SequenceEventFamily.SPAN_CONTROL)
        return tuple(families)

    @property
    def projection_mode(self) -> SequenceProjectionMode:
        """Return the coarse sequence-projection mode required by this schema."""
        if self.structure_markers.enabled:
            return SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS
        if self.controls.include_point_controls or self.controls.include_span_controls:
            return SequenceProjectionMode.NOTES_AND_CONTROLS
        return SequenceProjectionMode.NOTES_ONLY

    def to_version_payload(self) -> dict[str, Any]:
        """Return the frozen payload used to derive the sequence schema version."""
        return {
            "schema_name": self.schema_name,
            "schema_mode": self.schema_mode,
            "enabled_event_families": [
                family.value for family in self.enabled_event_families
            ],
            "note_payload_fields": [
                field_name.value for field_name in self.note_payload_fields
            ],
            "structure_markers": {
                "enabled": self.structure_markers.enabled,
                "marker_kinds": [
                    marker_kind.value
                    for marker_kind in self.structure_markers.marker_kinds
                ],
            },
            "controls": {
                "include_point_controls": self.controls.include_point_controls,
                "point_control_kinds": [
                    kind.value for kind in self.controls.point_control_kinds
                ],
                "include_span_controls": self.controls.include_span_controls,
                "span_control_kinds": [
                    kind.value for kind in self.controls.span_control_kinds
                ],
            },
        }


def coerce_sequence_schema_contract(
    value: SequenceSchemaContract | Mapping[str, Any],
) -> SequenceSchemaContract:
    """Coerce a Kedro-loaded mapping into the typed sequence schema contract."""
    if isinstance(value, SequenceSchemaContract):
        return value

    return SequenceSchemaContract(
        schema_name=str(value.get("schema_name", "baseline_sequence")),
        schema_mode=str(value.get("schema_mode", "baseline_v1")),
        note_payload_fields=tuple(
            value.get(
                "note_payload_fields",
                (
                    NotePayloadField.PITCH,
                    NotePayloadField.DURATION,
                ),
            )
        ),
        structure_markers=_coerce_structure_marker_policy(
            value.get("structure_markers", {})
        ),
        controls=_coerce_control_family_policy(value.get("controls", {})),
    )


def _coerce_structure_marker_policy(value: object) -> StructureMarkerPolicy:
    if isinstance(value, StructureMarkerPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "structure_markers must be a mapping or StructureMarkerPolicy."
        )
    return StructureMarkerPolicy(
        enabled=bool(value.get("enabled", True)),
        marker_kinds=tuple(value.get("marker_kinds", tuple(StructureMarkerKind))),
    )


def _coerce_control_family_policy(value: object) -> ControlFamilyPolicy:
    if isinstance(value, ControlFamilyPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("controls must be a mapping or ControlFamilyPolicy.")
    return ControlFamilyPolicy(
        include_point_controls=bool(value.get("include_point_controls", True)),
        point_control_kinds=tuple(
            value.get("point_control_kinds", tuple(PointControlKind))
        ),
        include_span_controls=bool(value.get("include_span_controls", True)),
        span_control_kinds=tuple(
            value.get("span_control_kinds", tuple(SpanControlKind))
        ),
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


__all__ = [
    "ControlFamilyPolicy",
    "NotePayloadField",
    "SequenceEventFamily",
    "SequenceSchemaContract",
    "StructureMarkerPolicy",
    "coerce_sequence_schema_contract",
]
