"""Tests for IR structural validation rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from motifml.ir.models import Edge, EdgeType, OptionalOverlays
from motifml.ir.serialization import deserialize_document
from motifml.ir.time import ScoreTime
from motifml.ir.validation import (
    IrValidationRule,
    build_document_validation_report,
    validate_document,
)

GOLDEN_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ir" / "golden"
PAIR_SIZE = 2


def test_build_document_validation_report_accepts_approved_golden_documents():
    report = build_document_validation_report(
        relative_path="single_track_monophonic_pickup.json",
        source_hash="single_track_monophonic_pickup",
        document=_load_golden_document("single_track_monophonic_pickup.ir.json"),
    )

    assert report.passed is True
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.rule_reports == ()


def test_validate_document_reports_structural_invariant_failures():
    document = deepcopy(_load_golden_document("single_track_monophonic_pickup.ir.json"))
    voice_lane_onset_indexes = _voice_lane_onset_indexes(document)

    broken_onset = document.onset_groups[voice_lane_onset_indexes[1]]
    object.__setattr__(broken_onset, "bar_id", "bar:999")
    object.__setattr__(
        broken_onset, "time", document.onset_groups[voice_lane_onset_indexes[0]].time
    )
    object.__setattr__(broken_onset, "attack_order_in_voice", 3)

    broken_note = document.note_events[-1]
    object.__setattr__(broken_note, "onset_id", "onset:missing")

    misaligned_note = document.note_events[-2]
    object.__setattr__(misaligned_note, "time", ScoreTime(7, 8))

    negative_duration_note = document.note_events[-3]
    object.__setattr__(negative_duration_note, "sounding_duration", ScoreTime(-1, 4))

    broken_voice_lane = document.voice_lanes[0]
    object.__setattr__(broken_voice_lane, "voice_lane_chain_id", "voice-chain:broken")

    issues_by_rule = validate_document(document)

    assert set(issues_by_rule) >= {
        IrValidationRule.ONSET_OWNERSHIP,
        IrValidationRule.NOTE_OWNERSHIP,
        IrValidationRule.NOTE_TIME_ALIGNMENT,
        IrValidationRule.VOICE_LANE_ONSET_TIMING,
        IrValidationRule.ATTACK_ORDER_CONTIGUITY,
        IrValidationRule.SOUNDING_DURATION_POSITIVE,
        IrValidationRule.VOICE_LANE_CHAIN_STABILITY,
    }


def test_validate_document_reports_non_canonical_note_order():
    document = deepcopy(_load_golden_document("ensemble_polyphony_controls.ir.json"))
    first_note_index, second_note_index = _chord_note_indexes(document)
    swapped_notes = list(document.note_events)
    swapped_notes[first_note_index], swapped_notes[second_note_index] = (
        swapped_notes[second_note_index],
        swapped_notes[first_note_index],
    )
    object.__setattr__(document, "note_events", tuple(swapped_notes))

    issues_by_rule = validate_document(document)

    assert IrValidationRule.NOTE_ORDER_CANONICAL in issues_by_rule


def test_validate_document_reports_tie_chain_and_edge_endpoint_failures():
    document = deepcopy(_load_golden_document("single_track_monophonic_pickup.ir.json"))
    note_a = document.note_events[0].note_id
    note_b = document.note_events[1].note_id
    broken_edge = Edge(
        source_id=document.onset_groups[0].onset_id,
        target_id=document.onset_groups[1].onset_id,
        edge_type=EdgeType.NEXT_IN_VOICE,
    )
    object.__setattr__(broken_edge, "target_id", "onset:missing")
    object.__setattr__(
        document,
        "edges",
        document.edges
        + (
            Edge(source_id=note_a, target_id=note_b, edge_type=EdgeType.TIE_TO),
            Edge(source_id=note_b, target_id=note_a, edge_type=EdgeType.TIE_TO),
            broken_edge,
        ),
    )

    issues_by_rule = validate_document(document)

    assert IrValidationRule.TIE_CHAIN_LINEAR in issues_by_rule
    assert IrValidationRule.EDGE_ENDPOINT_REFERENCE_INTEGRITY in issues_by_rule


def test_validate_document_reports_phrase_span_and_forbidden_metadata_failures():
    document = deepcopy(_load_golden_document("single_track_monophonic_pickup.ir.json"))
    object.__setattr__(
        document,
        "optional_overlays",
        OptionalOverlays(
            phrase_spans=(
                {
                    "phrase_id": "phrase:demo:0",
                    "scope_ref": "voice-chain:missing",
                    "start_time": {"numerator": 1, "denominator": 2},
                    "end_time": {"numerator": 1, "denominator": 4},
                    "title": "Forbidden",
                },
            )
        ),
    )

    issues_by_rule = validate_document(document)

    assert IrValidationRule.PHRASE_SPAN_VALIDITY in issues_by_rule
    assert IrValidationRule.FORBIDDEN_METADATA_ABSENT in issues_by_rule


def test_validate_document_reports_fretted_string_collisions():
    document = deepcopy(_load_golden_document("single_track_monophonic_pickup.ir.json"))
    original_note = document.note_events[0]
    duplicated_note = deepcopy(original_note)
    object.__setattr__(
        duplicated_note,
        "note_id",
        f"{original_note.note_id}:dup",
    )
    object.__setattr__(
        document,
        "note_events",
        document.note_events + (duplicated_note,),
    )
    object.__setattr__(
        document,
        "edges",
        document.edges
        + (
            Edge(
                source_id=original_note.onset_id,
                target_id=duplicated_note.note_id,
                edge_type=EdgeType.CONTAINS,
            ),
        ),
    )

    issues_by_rule = validate_document(document)

    assert IrValidationRule.FRETTED_STRING_COLLISION in issues_by_rule


def _load_golden_document(filename: str):
    path = GOLDEN_FIXTURE_ROOT / filename
    return deserialize_document(path.read_text(encoding="utf-8"))


def _voice_lane_onset_indexes(document) -> tuple[int, int]:
    onset_indexes_by_voice_lane: dict[str, list[int]] = {}
    for index, onset in enumerate(document.onset_groups):
        onset_indexes_by_voice_lane.setdefault(onset.voice_lane_id, []).append(index)

    for indexes in onset_indexes_by_voice_lane.values():
        if len(indexes) >= PAIR_SIZE:
            return indexes[0], indexes[1]

    raise AssertionError("Expected a voice lane with at least two onsets.")


def _chord_note_indexes(document) -> tuple[int, int]:
    note_indexes_by_onset: dict[str, list[int]] = {}
    for index, note in enumerate(document.note_events):
        note_indexes_by_onset.setdefault(note.onset_id, []).append(index)

    for indexes in note_indexes_by_onset.values():
        if len(indexes) >= PAIR_SIZE:
            return indexes[0], indexes[1]

    raise AssertionError("Expected an onset with at least two notes.")
