"""Tests for sequence-event token ordering helpers."""

from __future__ import annotations

import pytest

from motifml.ir.models import (
    ControlScope,
    DynamicChangeValue,
    HairpinValue,
    NoteEvent,
    Pitch,
    PointControlEvent,
    PointControlKind,
    SpanControlEvent,
    SpanControlKind,
)
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    SpanControlSequenceEvent,
    StructureMarkerKind,
    StructureMarkerSequenceEvent,
)
from motifml.ir.time import ScoreTime
from motifml.training.sequence_schema import NotePayloadField
from motifml.training.token_ordering import (
    expand_sequence_event_spans,
    flatten_token_spans,
)


def test_expand_sequence_event_spans_uses_canonical_note_field_order() -> None:
    spans = expand_sequence_event_spans(
        (_build_note_event(time=ScoreTime(0, 1), velocity=80, string_number=2),),
        time_resolution=96,
        note_payload_fields=(
            NotePayloadField.VELOCITY,
            NotePayloadField.STRING_NUMBER,
            NotePayloadField.DURATION,
            NotePayloadField.PITCH,
        ),
    )

    assert len(spans) == 1
    assert spans[0].time_shift_ticks is None
    assert spans[0].tokens == (
        "NOTE_PITCH:C4",
        "NOTE_DURATION:96",
        "NOTE_STRING:2",
        "NOTE_VELOCITY:80",
    )


def test_expand_sequence_event_spans_emits_time_shift_only_when_time_advances() -> None:
    spans = expand_sequence_event_spans(
        (
            _build_structure_event(ScoreTime(0, 1), "bar", "bar:1"),
            _build_point_control_event(ScoreTime(1, 4)),
            _build_note_event(time=ScoreTime(1, 4)),
            _build_span_control_event(ScoreTime(1, 2), ScoreTime(3, 4)),
        ),
        time_resolution=96,
    )

    assert spans[0].tokens == ("STRUCTURE:BAR",)
    assert spans[1].tokens == ("TIME_SHIFT:96", "CONTROL_POINT:SCORE:DYNAMIC_CHANGE:MF")
    assert spans[2].tokens == ("NOTE_PITCH:C4", "NOTE_DURATION:96")
    assert spans[3].tokens == (
        "TIME_SHIFT:96",
        "CONTROL_SPAN:VOICE:HAIRPIN:CRESCENDO:96",
    )


def test_flatten_token_spans_places_explicit_bos_and_eos_at_sequence_boundaries() -> (
    None
):
    spans = expand_sequence_event_spans(
        (_build_structure_event(ScoreTime(0, 1), "part", "part:1"),),
        time_resolution=96,
    )

    assert flatten_token_spans(spans, bos_token="<bos>", eos_token="<eos>") == (
        "<bos>",
        "STRUCTURE:PART",
        "<eos>",
    )


def test_expand_sequence_event_spans_rejects_out_of_order_events() -> None:
    with pytest.raises(ValueError, match="canonical order"):
        expand_sequence_event_spans(
            (
                _build_note_event(time=ScoreTime(0, 1)),
                _build_structure_event(ScoreTime(0, 1), "part", "part:1"),
            ),
            time_resolution=96,
        )


def test_expand_sequence_event_spans_rejects_non_quantized_time_deltas() -> None:
    with pytest.raises(ValueError, match="quantize exactly"):
        expand_sequence_event_spans(
            (_build_structure_event(ScoreTime(1, 10), "part", "part:1"),),
            time_resolution=96,
        )


def _build_structure_event(
    time: ScoreTime,
    marker_kind: str,
    entity_id: str,
) -> StructureMarkerSequenceEvent:
    return StructureMarkerSequenceEvent(
        time=time,
        marker_kind=StructureMarkerKind(marker_kind),
        entity_id=entity_id,
    )


def _build_note_event(
    *,
    time: ScoreTime,
    velocity: int | None = None,
    string_number: int | None = None,
) -> NoteSequenceEvent:
    return NoteSequenceEvent(
        time=time,
        note=NoteEvent(
            note_id="note:onset:1:1",
            onset_id="onset:1",
            part_id="part:1",
            staff_id="staff:part:1:1",
            time=time,
            attack_duration=ScoreTime(1, 4),
            sounding_duration=ScoreTime(1, 4),
            pitch=Pitch(step="C", octave=4),
            velocity=velocity,
            string_number=string_number,
        ),
        part_id="part:1",
        staff_id="staff:part:1:1",
        bar_id="bar:1",
        voice_lane_id="voice:1",
        onset_id="onset:1",
    )


def _build_point_control_event(time: ScoreTime) -> PointControlSequenceEvent:
    return PointControlSequenceEvent(
        time=time,
        control=PointControlEvent(
            control_id="ctrlp:score:1",
            kind=PointControlKind.DYNAMIC_CHANGE,
            scope=ControlScope.SCORE,
            target_ref="score",
            time=time,
            value=DynamicChangeValue(marking="mf"),
        ),
    )


def _build_span_control_event(
    start_time: ScoreTime,
    end_time: ScoreTime,
) -> SpanControlSequenceEvent:
    return SpanControlSequenceEvent(
        time=start_time,
        control=SpanControlEvent(
            control_id="ctrls:voice:1",
            kind=SpanControlKind.HAIRPIN,
            scope=ControlScope.VOICE,
            target_ref="voice:1",
            start_time=start_time,
            end_time=end_time,
            value=HairpinValue(direction="crescendo"),
        ),
        voice_lane_id="voice:1",
    )
