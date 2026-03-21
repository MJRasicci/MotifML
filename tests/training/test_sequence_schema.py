"""Tests for the frozen baseline sequence-schema contract."""

from __future__ import annotations

from motifml.training.sequence_schema import (
    ControlFamilyPolicy,
    NotePayloadField,
    SequenceEventFamily,
    SequenceSchemaContract,
    StructureMarkerPolicy,
    coerce_sequence_schema_contract,
)
from motifml.training.versioning import build_feature_version


def test_sequence_schema_defaults_to_the_baseline_rich_event_surface() -> None:
    schema = SequenceSchemaContract()

    assert schema.enabled_event_families == (
        SequenceEventFamily.NOTE,
        SequenceEventFamily.STRUCTURE_MARKER,
        SequenceEventFamily.POINT_CONTROL,
        SequenceEventFamily.SPAN_CONTROL,
    )
    assert schema.note_payload_fields == (
        NotePayloadField.PITCH,
        NotePayloadField.DURATION,
    )


def test_sequence_schema_can_be_coerced_from_kedro_loaded_mappings() -> None:
    schema = coerce_sequence_schema_contract(
        {
            "schema_name": "baseline_sequence",
            "schema_mode": "baseline_v1",
            "note_payload_fields": ["pitch", "duration", "string_number"],
            "structure_markers": {"enabled": False},
            "controls": {
                "include_point_controls": True,
                "include_span_controls": False,
            },
        }
    )

    assert schema.note_payload_fields == (
        NotePayloadField.PITCH,
        NotePayloadField.DURATION,
        NotePayloadField.STRING_NUMBER,
    )
    assert schema.structure_markers.enabled is False
    assert schema.controls.include_span_controls is False
    assert schema.enabled_event_families == (
        SequenceEventFamily.NOTE,
        SequenceEventFamily.POINT_CONTROL,
    )


def test_sequence_schema_version_changes_when_note_payload_fields_change() -> None:
    baseline = SequenceSchemaContract()
    updated = SequenceSchemaContract(
        note_payload_fields=(
            NotePayloadField.PITCH,
            NotePayloadField.DURATION,
            NotePayloadField.STRING_NUMBER,
        )
    )

    assert updated.sequence_schema_version != baseline.sequence_schema_version


def test_sequence_schema_version_changes_when_inclusion_policies_change() -> None:
    baseline = SequenceSchemaContract()
    updated = SequenceSchemaContract(
        structure_markers=StructureMarkerPolicy(enabled=False),
        controls=ControlFamilyPolicy(include_span_controls=False),
    )

    assert updated.sequence_schema_version != baseline.sequence_schema_version


def test_feature_version_changes_when_sequence_schema_changes() -> None:
    baseline_schema = SequenceSchemaContract()
    updated_schema = SequenceSchemaContract(
        note_payload_fields=(
            NotePayloadField.PITCH,
            NotePayloadField.DURATION,
            NotePayloadField.STRING_NUMBER,
        )
    )

    baseline_version = build_feature_version(
        normalized_ir_version="normalized-v1",
        projection_config={
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
        },
        sequence_schema_version=baseline_schema.sequence_schema_version,
    )
    updated_version = build_feature_version(
        normalized_ir_version="normalized-v1",
        projection_config={
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
        },
        sequence_schema_version=updated_schema.sequence_schema_version,
    )

    assert updated_version != baseline_version
