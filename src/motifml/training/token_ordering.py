"""Sequence-event token expansion helpers with explicit ordering guarantees."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from motifml.ir.models import (
    DynamicChangeValue,
    FermataValue,
    HairpinValue,
    OttavaValue,
    Pitch,
    TempoChangeValue,
)
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    SequenceEvent,
    SpanControlSequenceEvent,
    StructureMarkerSequenceEvent,
)
from motifml.ir.time import ScoreTime
from motifml.training.sequence_schema import NotePayloadField
from motifml.training.token_families import (
    build_control_point_token,
    build_control_span_token,
    build_note_duration_token,
    build_note_pitch_token,
    build_note_string_token,
    build_note_velocity_token,
    build_structure_token,
    build_time_shift_token,
)

_ZERO_TIME = ScoreTime(0, 1)
_WHOLE_NOTES_PER_QUARTER = 4
_NOTE_FIELD_ORDER = (
    NotePayloadField.PITCH,
    NotePayloadField.DURATION,
    NotePayloadField.STRING_NUMBER,
    NotePayloadField.VELOCITY,
)


@dataclass(frozen=True, slots=True)
class EventTokenSpan:
    """One contiguous token emission block for a projected sequence event."""

    event_index: int
    event_type: str
    start_time: ScoreTime
    time_shift_ticks: int | None
    tokens: tuple[str, ...]


def expand_sequence_event_spans(
    events: Sequence[SequenceEvent],
    *,
    time_resolution: int,
    ordering_context: str | None = None,
    note_payload_fields: Sequence[NotePayloadField | str] = (
        NotePayloadField.PITCH,
        NotePayloadField.DURATION,
    ),
) -> tuple[EventTokenSpan, ...]:
    """Expand ordered sequence events into contiguous token spans."""
    if time_resolution <= 0:
        raise ValueError("time_resolution must be positive.")

    normalized_fields = _canonical_note_payload_fields(note_payload_fields)
    validate_sequence_event_order(events, context=ordering_context)

    spans: list[EventTokenSpan] = []
    previous_time = _ZERO_TIME
    for index, event in enumerate(events):
        time_shift_ticks = _time_shift_ticks(previous_time, event.time, time_resolution)
        event_tokens = _event_tokens(
            event,
            time_shift_ticks=time_shift_ticks,
            time_resolution=time_resolution,
            note_payload_fields=normalized_fields,
        )
        spans.append(
            EventTokenSpan(
                event_index=index,
                event_type=type(event).__name__,
                start_time=event.time,
                time_shift_ticks=time_shift_ticks,
                tokens=event_tokens,
            )
        )
        previous_time = event.time

    return tuple(spans)


def flatten_token_spans(
    spans: Sequence[EventTokenSpan],
    *,
    bos_token: str | None = None,
    eos_token: str | None = None,
) -> tuple[str, ...]:
    """Flatten token spans into one token sequence with explicit BOS/EOS placement."""
    tokens: list[str] = []
    if bos_token is not None:
        tokens.append(_normalize_boundary_token(bos_token, "bos_token"))
    for span in spans:
        tokens.extend(span.tokens)
    if eos_token is not None:
        tokens.append(_normalize_boundary_token(eos_token, "eos_token"))
    return tuple(tokens)


def _event_tokens(
    event: SequenceEvent,
    *,
    time_shift_ticks: int | None,
    time_resolution: int,
    note_payload_fields: tuple[NotePayloadField, ...],
) -> tuple[str, ...]:
    tokens: list[str] = []
    if time_shift_ticks is not None:
        tokens.append(build_time_shift_token(time_shift_ticks))

    if isinstance(event, StructureMarkerSequenceEvent):
        tokens.append(build_structure_token(event.marker_kind.value))
        return tuple(tokens)

    if isinstance(event, PointControlSequenceEvent):
        tokens.append(
            build_control_point_token(
                event.control.scope.value,
                event.control.kind.value,
                _encode_point_control_value(event.control.value),
            )
        )
        return tuple(tokens)

    if isinstance(event, SpanControlSequenceEvent):
        tokens.append(
            build_control_span_token(
                event.control.scope.value,
                event.control.kind.value,
                _encode_span_control_value(event.control.value),
                _score_time_to_ticks(
                    event.control.end_time - event.control.start_time,
                    time_resolution,
                ),
            )
        )
        return tuple(tokens)

    if isinstance(event, NoteSequenceEvent):
        for field_name in note_payload_fields:
            if field_name is NotePayloadField.PITCH:
                tokens.append(build_note_pitch_token(_encode_pitch(event.note.pitch)))
                continue
            if field_name is NotePayloadField.DURATION:
                tokens.append(
                    build_note_duration_token(
                        _score_time_to_ticks(
                            event.note.attack_duration, time_resolution
                        )
                    )
                )
                continue
            if field_name is NotePayloadField.STRING_NUMBER:
                if event.note.string_number is not None:
                    tokens.append(build_note_string_token(event.note.string_number))
                continue
            if field_name is NotePayloadField.VELOCITY:
                if event.note.velocity is not None:
                    tokens.append(build_note_velocity_token(event.note.velocity))
                continue

        return tuple(tokens)

    raise TypeError(f"Unsupported sequence event type: {type(event).__name__}")


def _canonical_note_payload_fields(
    note_payload_fields: Sequence[NotePayloadField | str],
) -> tuple[NotePayloadField, ...]:
    requested = {NotePayloadField(field_name) for field_name in note_payload_fields}
    return tuple(
        field_name for field_name in _NOTE_FIELD_ORDER if field_name in requested
    )


def validate_sequence_event_order(
    events: Sequence[SequenceEvent],
    *,
    context: str | None = None,
) -> None:
    expected = tuple(sorted(events, key=lambda event: event.sort_key()))
    if tuple(events) == expected:
        return

    context_prefix = ""
    if context is not None:
        normalized_context = context.strip()
        if not normalized_context:
            raise ValueError("context must be non-empty when provided.")
        context_prefix = f"{normalized_context}: "

    for index, (actual, ordered) in enumerate(zip(events, expected, strict=True)):
        if actual != ordered:
            raise ValueError(
                f"{context_prefix}Sequence events must already be in canonical order "
                "before token expansion. "
                f"event_index={index}, "
                f"actual={type(actual).__name__}@{actual.time} "
                f"sort_key={actual.sort_key()}, "
                f"expected={type(ordered).__name__}@{ordered.time} "
                f"sort_key={ordered.sort_key()}."
            )


def _time_shift_ticks(
    previous_time: ScoreTime,
    current_time: ScoreTime,
    time_resolution: int,
) -> int | None:
    delta = current_time - previous_time
    if delta.numerator < 0:
        raise ValueError("Sequence event times must not move backward.")
    if delta == _ZERO_TIME:
        return None
    return _score_time_to_ticks(delta, time_resolution)


def _score_time_to_ticks(score_time: ScoreTime, time_resolution: int) -> int:
    numerator = score_time.numerator * time_resolution * _WHOLE_NOTES_PER_QUARTER
    if numerator % score_time.denominator != 0:
        raise ValueError(
            "ScoreTime values must quantize exactly at the configured time_resolution."
        )
    ticks = numerator // score_time.denominator
    if ticks <= 0:
        raise ValueError("Quantized tick values must be positive.")
    return ticks


def _encode_pitch(pitch: Pitch | None) -> str:
    if pitch is None:
        return "UNPITCHED"

    accidental = "" if pitch.accidental is None else pitch.accidental
    return f"{pitch.step.value}{accidental}{pitch.octave}"


def _encode_point_control_value(
    value: TempoChangeValue | DynamicChangeValue | FermataValue,
) -> str:
    if isinstance(value, TempoChangeValue):
        return f"BPM={format(value.beats_per_minute, 'g')}"
    if isinstance(value, DynamicChangeValue):
        return value.marking
    fermata_type = "GENERIC" if value.fermata_type is None else value.fermata_type
    if value.length_scale is None:
        return fermata_type
    return f"{fermata_type}_{format(value.length_scale, 'g')}"


def _encode_span_control_value(value: HairpinValue | OttavaValue) -> str:
    if isinstance(value, HairpinValue):
        return (
            f"{value.direction.value}_NIENTE" if value.niente else value.direction.value
        )
    return f"OCTAVE_SHIFT_{value.octave_shift}"


def _normalize_boundary_token(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when provided.")
    return normalized


__all__ = [
    "EventTokenSpan",
    "expand_sequence_event_spans",
    "flatten_token_spans",
    "validate_sequence_event_order",
]
