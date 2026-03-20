"""Tests for canonical IR JSON serialization."""

from __future__ import annotations

import json
from pathlib import Path

from motifml.ir.models import (
    Bar,
    ControlScope,
    DynamicChangeValue,
    Edge,
    EdgeType,
    HairpinDirection,
    HairpinValue,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    OptionalOverlays,
    OptionalViews,
    Part,
    PhraseKind,
    PhraseSource,
    PhraseSpan,
    Pitch,
    PointControlEvent,
    PointControlKind,
    SpanControlEvent,
    SpanControlKind,
    Staff,
    TempoChangeValue,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.serialization import deserialize_document, serialize_document
from motifml.ir.time import ScoreTime

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ir"
SERIALIZED_DOCUMENT_FIXTURE = FIXTURE_ROOT / "representative_document.ir.json"


def test_serialize_document_enforces_canonical_collection_order_and_is_byte_stable():
    serialized_once = serialize_document(_build_unsorted_document())
    serialized_twice = serialize_document(_build_unsorted_document())
    payload = json.loads(serialized_once)

    assert serialized_once == serialized_twice
    assert [part["part_id"] for part in payload["parts"]] == [
        "part:track-a",
        "part:track-b",
    ]
    assert [bar["bar_id"] for bar in payload["bars"]] == ["bar:0", "bar:1"]
    assert [note["note_id"] for note in payload["note_events"]] == [
        "note:onset:voice:staff:part:track-a:0:0:0:0:0",
        "note:onset:voice:staff:part:track-a:0:0:0:0:1",
    ]


def test_deserialize_document_round_trips_a_canonical_document():
    document = _build_canonical_document()

    serialized = serialize_document(document)
    deserialized = deserialize_document(serialized)

    assert deserialized == document
    assert serialize_document(deserialized) == serialized


def test_serialize_document_matches_the_checked_in_golden_fixture():
    document = _build_canonical_document()

    assert serialize_document(document) == SERIALIZED_DOCUMENT_FIXTURE.read_text(
        encoding="utf-8"
    )


def test_serialize_document_round_trips_phrase_spans_canonically():
    document = _build_unsorted_document()
    document = MotifMlIrDocument(
        metadata=document.metadata,
        parts=document.parts,
        staves=document.staves,
        bars=document.bars,
        voice_lanes=document.voice_lanes,
        point_control_events=document.point_control_events,
        span_control_events=document.span_control_events,
        onset_groups=document.onset_groups,
        note_events=document.note_events,
        edges=document.edges,
        optional_overlays=OptionalOverlays(
            phrase_spans=(
                PhraseSpan(
                    phrase_id="phrase:part:track-a:1",
                    scope_ref="part:track-a",
                    start_time=ScoreTime(1, 4),
                    end_time=ScoreTime(1, 1),
                    phrase_kind=PhraseKind.GESTURE,
                    source=PhraseSource.DERIVED_RULE_BASED,
                    confidence=0.75,
                ),
                PhraseSpan(
                    phrase_id="phrase:part:track-a:0",
                    scope_ref="part:track-a",
                    start_time=ScoreTime(0, 1),
                    end_time=ScoreTime(1, 2),
                    phrase_kind=PhraseKind.MELODIC,
                    source=PhraseSource.MANUAL_ANNOTATION,
                    confidence="high",
                ),
            )
        ),
        optional_views=document.optional_views,
    )

    serialized = serialize_document(document)
    payload = json.loads(serialized)
    round_tripped = deserialize_document(serialized)

    assert [
        span["phrase_id"] for span in payload["optional_overlays"]["phrase_spans"]
    ] == [
        "phrase:part:track-a:0",
        "phrase:part:track-a:1",
    ]
    assert serialize_document(round_tripped) == serialized
    assert (
        round_tripped.optional_overlays.phrase_spans[0].phrase_kind
        is PhraseKind.MELODIC
    )


def _build_unsorted_document() -> MotifMlIrDocument:
    return MotifMlIrDocument(
        metadata=IrDocumentMetadata(
            ir_schema_version="1.0.0",
            corpus_build_version="build-1",
            generator_version="0.1.0",
            source_document_hash="abc123",
        ),
        parts=(
            Part(
                part_id="part:track-b",
                instrument_family=2,
                instrument_kind=3,
                role=4,
                transposition=Transposition(),
                staff_ids=("staff:part:track-b:0",),
            ),
            Part(
                part_id="part:track-a",
                instrument_family=1,
                instrument_kind=2,
                role=3,
                transposition=Transposition(chromatic=-2, octave=1),
                staff_ids=("staff:part:track-a:0",),
            ),
        ),
        staves=(
            Staff(
                staff_id="staff:part:track-b:0", part_id="part:track-b", staff_index=0
            ),
            Staff(
                staff_id="staff:part:track-a:0", part_id="part:track-a", staff_index=0
            ),
        ),
        bars=(
            Bar(
                bar_id="bar:1",
                bar_index=1,
                start=ScoreTime(1, 1),
                duration=ScoreTime(4, 4),
                time_signature=TimeSignature(4, 4),
            ),
            Bar(
                bar_id="bar:0",
                bar_index=0,
                start=ScoreTime(0, 1),
                duration=ScoreTime(4, 4),
                time_signature=TimeSignature(4, 4),
            ),
        ),
        voice_lanes=(
            VoiceLane(
                voice_lane_id="voice:staff:part:track-a:0:1:0",
                voice_lane_chain_id="voice-chain:part:track-a:staff:part:track-a:0:0",
                part_id="part:track-a",
                staff_id="staff:part:track-a:0",
                bar_id="bar:1",
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id="voice:staff:part:track-a:0:0:0",
                voice_lane_chain_id="voice-chain:part:track-a:staff:part:track-a:0:0",
                part_id="part:track-a",
                staff_id="staff:part:track-a:0",
                bar_id="bar:0",
                voice_index=0,
            ),
        ),
        point_control_events=(
            PointControlEvent(
                control_id="ctrlp:score:1",
                kind=PointControlKind.DYNAMIC_CHANGE,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(1, 4),
                value=DynamicChangeValue(marking="mf"),
            ),
            PointControlEvent(
                control_id="ctrlp:score:0",
                kind=PointControlKind.TEMPO_CHANGE,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(0, 1),
                value=TempoChangeValue(beats_per_minute=120.0),
            ),
        ),
        span_control_events=(
            SpanControlEvent(
                control_id="ctrls:staff:0",
                kind=SpanControlKind.HAIRPIN,
                scope=ControlScope.STAFF,
                target_ref="staff:part:track-a:0",
                start_time=ScoreTime(0, 1),
                end_time=ScoreTime(1, 2),
                value=HairpinValue(direction=HairpinDirection.CRESCENDO),
            ),
        ),
        onset_groups=(
            OnsetGroup(
                onset_id="onset:voice:staff:part:track-a:0:0:0:1",
                voice_lane_id="voice:staff:part:track-a:0:0:0",
                bar_id="bar:0",
                time=ScoreTime(1, 4),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=1,
            ),
            OnsetGroup(
                onset_id="onset:voice:staff:part:track-a:0:0:0:0",
                voice_lane_id="voice:staff:part:track-a:0:0:0",
                bar_id="bar:0",
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=0,
            ),
        ),
        note_events=(
            NoteEvent(
                note_id="note:onset:voice:staff:part:track-a:0:0:0:0:1",
                onset_id="onset:voice:staff:part:track-a:0:0:0:0",
                part_id="part:track-a",
                staff_id="staff:part:track-a:0",
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step="E", octave=4),
            ),
            NoteEvent(
                note_id="note:onset:voice:staff:part:track-a:0:0:0:0:0",
                onset_id="onset:voice:staff:part:track-a:0:0:0:0",
                part_id="part:track-a",
                staff_id="staff:part:track-a:0",
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 2),
                pitch=Pitch(step="C", accidental="#", octave=4),
                string_number=1,
                show_string_number=True,
            ),
        ),
        edges=(
            Edge(
                source_id="onset:voice:staff:part:track-a:0:0:0:0",
                target_id="note:onset:voice:staff:part:track-a:0:0:0:0:0",
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id="part:track-a",
                target_id="staff:part:track-a:0",
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id="bar:0",
                target_id="voice:staff:part:track-a:0:0:0",
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id="voice:staff:part:track-a:0:0:0",
                target_id="onset:voice:staff:part:track-a:0:0:0:0",
                edge_type=EdgeType.CONTAINS,
            ),
        ),
        optional_overlays=OptionalOverlays(),
        optional_views=OptionalViews(),
    )


def _build_canonical_document() -> MotifMlIrDocument:
    return deserialize_document(serialize_document(_build_unsorted_document()))
