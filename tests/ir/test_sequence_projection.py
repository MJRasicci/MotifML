"""Tests for the typed MotifML IR sequence projection."""

from __future__ import annotations

from motifml.ir.ids import (
    bar_id,
    note_id,
    onset_id,
    part_id,
    point_control_id,
    span_control_id,
    staff_id,
    voice_lane_chain_id,
    voice_lane_id,
)
from motifml.ir.models import (
    Bar,
    ControlScope,
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
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    SequenceProjectionConfig,
    SequenceProjectionMode,
    SpanControlSequenceEvent,
    StructureMarkerSequenceEvent,
    project_sequence,
)
from motifml.ir.time import ScoreTime

SINGLE_VOICE_NOTE_COUNT = 3
MULTI_VOICE_NOTE_COUNT = 3
MULTI_VOICE_CONTROL_COUNT = 2
FULL_SEQUENCE_COUNT = 13


def test_project_sequence_orders_single_voice_events_by_time_and_note_key():
    document = _build_single_voice_document()

    projection = project_sequence(document)

    assert projection.mode is SequenceProjectionMode.NOTES_ONLY
    assert len(projection.events) == SINGLE_VOICE_NOTE_COUNT
    assert all(isinstance(event, NoteSequenceEvent) for event in projection.events)
    assert [event.time for event in projection.events] == [
        ScoreTime(0, 1),
        ScoreTime(0, 1),
        ScoreTime(1, 4),
    ]
    assert [event.note.note_id for event in projection.events] == [
        "note:onset:voice:staff:part:track-a:0:0:0:0:1",
        "note:onset:voice:staff:part:track-a:0:0:0:0:0",
        "note:onset:voice:staff:part:track-a:0:0:0:1:0",
    ]


def test_project_sequence_interleaves_multiple_voices_by_time():
    document = _build_multi_voice_document()

    projection = project_sequence(document)

    assert [event.time for event in projection.events] == [
        ScoreTime(0, 1),
        ScoreTime(1, 8),
        ScoreTime(1, 4),
    ]
    assert [event.voice_lane_id for event in projection.events] == [
        "voice:staff:part:track-b:0:0:0",
        "voice:staff:part:track-b:0:0:1",
        "voice:staff:part:track-b:0:0:0",
    ]


def test_project_sequence_filters_controls_and_structure_markers_by_mode():
    document = _build_multi_voice_document()

    notes_only = project_sequence(document, SequenceProjectionConfig(mode="notes_only"))
    notes_and_controls = project_sequence(
        document,
        SequenceProjectionConfig(mode=SequenceProjectionMode.NOTES_AND_CONTROLS),
    )
    full_sequence = project_sequence(
        document,
        SequenceProjectionConfig(
            mode=SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS
        ),
    )

    assert len(notes_only.events) == MULTI_VOICE_NOTE_COUNT
    assert all(isinstance(event, NoteSequenceEvent) for event in notes_only.events)

    assert len(notes_and_controls.events) == (
        MULTI_VOICE_NOTE_COUNT + MULTI_VOICE_CONTROL_COUNT
    )
    assert any(
        isinstance(event, PointControlSequenceEvent)
        for event in notes_and_controls.events
    )
    assert any(
        isinstance(event, SpanControlSequenceEvent)
        for event in notes_and_controls.events
    )
    assert all(
        not isinstance(event, StructureMarkerSequenceEvent)
        for event in notes_and_controls.events
    )

    assert len(full_sequence.events) == FULL_SEQUENCE_COUNT
    assert isinstance(full_sequence.events[0], StructureMarkerSequenceEvent)
    assert full_sequence.events[0].part_id == "part:track-b"
    assert full_sequence.events[0].voice_lane_id is None
    assert any(
        isinstance(event, StructureMarkerSequenceEvent)
        and event.voice_lane_id == "voice:staff:part:track-b:0:0:0"
        for event in full_sequence.events
    )


def _build_single_voice_document() -> MotifMlIrDocument:
    part = part_id("track-a")
    staff = staff_id(part, 0)
    bar = bar_id(0)
    voice_lane = voice_lane_id(staff, 0, 0)
    chain = voice_lane_chain_id(part, staff, 0)
    onset_0 = onset_id(voice_lane, 0)
    onset_1 = onset_id(voice_lane, 1)

    return MotifMlIrDocument(
        metadata=_metadata(),
        parts=(
            Part(
                part_id=part,
                instrument_family=1,
                instrument_kind=2,
                role=3,
                transposition=Transposition(),
                staff_ids=(staff,),
            ),
        ),
        staves=(
            Staff(
                staff_id=staff,
                part_id=part,
                staff_index=0,
            ),
        ),
        bars=(
            Bar(
                bar_id=bar,
                bar_index=0,
                start=ScoreTime(0, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            ),
        ),
        voice_lanes=(
            VoiceLane(
                voice_lane_id=voice_lane,
                voice_lane_chain_id=chain,
                part_id=part,
                staff_id=staff,
                bar_id=bar,
                voice_index=0,
            ),
        ),
        onset_groups=(
            OnsetGroup(
                onset_id=onset_1,
                voice_lane_id=voice_lane,
                bar_id=bar,
                time=ScoreTime(1, 4),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=1,
            ),
            OnsetGroup(
                onset_id=onset_0,
                voice_lane_id=voice_lane,
                bar_id=bar,
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=0,
            ),
        ),
        note_events=(
            NoteEvent(
                note_id=note_id(onset_0, 1),
                onset_id=onset_0,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step="C", octave=4),
                string_number=1,
            ),
            NoteEvent(
                note_id=note_id(onset_0, 0),
                onset_id=onset_0,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step="E", octave=4),
                string_number=2,
            ),
            NoteEvent(
                note_id=note_id(onset_1, 0),
                onset_id=onset_1,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(1, 4),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step="G", octave=4),
            ),
        ),
        edges=(
            Edge(
                source_id=onset_0,
                target_id=note_id(onset_0, 0),
                edge_type=EdgeType.CONTAINS,
            ),
        ),
        optional_overlays=OptionalOverlays(),
        optional_views=OptionalViews(),
    )


def _build_multi_voice_document() -> MotifMlIrDocument:
    part = part_id("track-b")
    staff = staff_id(part, 0)
    bar = bar_id(0)
    voice_lane_0 = voice_lane_id(staff, 0, 0)
    voice_lane_1 = voice_lane_id(staff, 0, 1)
    chain_0 = voice_lane_chain_id(part, staff, 0)
    chain_1 = voice_lane_chain_id(part, staff, 1)
    onset_0 = onset_id(voice_lane_0, 0)
    onset_1 = onset_id(voice_lane_1, 0)
    onset_2 = onset_id(voice_lane_0, 1)

    return MotifMlIrDocument(
        metadata=_metadata(),
        parts=(
            Part(
                part_id=part,
                instrument_family=1,
                instrument_kind=2,
                role=3,
                transposition=Transposition(),
                staff_ids=(staff,),
            ),
        ),
        staves=(
            Staff(
                staff_id=staff,
                part_id=part,
                staff_index=0,
            ),
        ),
        bars=(
            Bar(
                bar_id=bar,
                bar_index=0,
                start=ScoreTime(0, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            ),
        ),
        voice_lanes=(
            VoiceLane(
                voice_lane_id=voice_lane_0,
                voice_lane_chain_id=chain_0,
                part_id=part,
                staff_id=staff,
                bar_id=bar,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=voice_lane_1,
                voice_lane_chain_id=chain_1,
                part_id=part,
                staff_id=staff,
                bar_id=bar,
                voice_index=1,
            ),
        ),
        point_control_events=(
            PointControlEvent(
                control_id=point_control_id("score", 0),
                kind=PointControlKind.TEMPO_CHANGE,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(1, 8),
                value=TempoChangeValue(beats_per_minute=120.0),
            ),
        ),
        span_control_events=(
            SpanControlEvent(
                control_id=span_control_id(staff, 0),
                kind=SpanControlKind.HAIRPIN,
                scope=ControlScope.STAFF,
                target_ref=staff,
                start_time=ScoreTime(1, 8),
                end_time=ScoreTime(1, 2),
                value=HairpinValue(direction=HairpinDirection.CRESCENDO),
            ),
        ),
        onset_groups=(
            OnsetGroup(
                onset_id=onset_0,
                voice_lane_id=voice_lane_0,
                bar_id=bar,
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 8),
                is_rest=False,
                attack_order_in_voice=0,
            ),
            OnsetGroup(
                onset_id=onset_1,
                voice_lane_id=voice_lane_1,
                bar_id=bar,
                time=ScoreTime(1, 8),
                duration_notated=ScoreTime(1, 8),
                is_rest=False,
                attack_order_in_voice=0,
            ),
            OnsetGroup(
                onset_id=onset_2,
                voice_lane_id=voice_lane_0,
                bar_id=bar,
                time=ScoreTime(1, 4),
                duration_notated=ScoreTime(1, 8),
                is_rest=False,
                attack_order_in_voice=1,
            ),
        ),
        note_events=(
            NoteEvent(
                note_id=note_id(onset_0, 0),
                onset_id=onset_0,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 8),
                sounding_duration=ScoreTime(1, 8),
                pitch=Pitch(step="C", octave=4),
            ),
            NoteEvent(
                note_id=note_id(onset_1, 0),
                onset_id=onset_1,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(1, 8),
                attack_duration=ScoreTime(1, 8),
                sounding_duration=ScoreTime(1, 8),
                pitch=Pitch(step="D", octave=4),
            ),
            NoteEvent(
                note_id=note_id(onset_2, 0),
                onset_id=onset_2,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(1, 4),
                attack_duration=ScoreTime(1, 8),
                sounding_duration=ScoreTime(1, 8),
                pitch=Pitch(step="E", octave=4),
            ),
        ),
        edges=(
            Edge(
                source_id=onset_0,
                target_id=note_id(onset_0, 0),
                edge_type=EdgeType.CONTAINS,
            ),
        ),
        optional_overlays=OptionalOverlays(),
        optional_views=OptionalViews(),
    )


def _metadata() -> IrDocumentMetadata:
    return IrDocumentMetadata(
        ir_schema_version="1.0.0",
        corpus_build_version="build-1",
        generator_version="0.1.0",
        source_document_hash="abc123",
    )
