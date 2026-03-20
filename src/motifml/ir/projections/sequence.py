"""Typed sequence projection for MotifML IR documents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from motifml.ir.ids import note_sort_key
from motifml.ir.models import (
    Bar,
    ControlScope,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    Part,
    PointControlEvent,
    SpanControlEvent,
    Staff,
    VoiceLane,
)
from motifml.ir.time import ScoreTime

_ZERO_TIME = ScoreTime(0, 1)


class SequenceProjectionMode(StrEnum):
    """Supported inclusion modes for the sequence projection."""

    NOTES_ONLY = "notes_only"
    NOTES_AND_CONTROLS = "notes_and_controls"
    NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS = (
        "notes_and_controls_and_structure_markers"
    )


class SequenceEventKind(StrEnum):
    """Typed event families emitted by the sequence projection."""

    STRUCTURE_MARKER = "structure_marker"
    POINT_CONTROL = "point_control"
    SPAN_CONTROL = "span_control"
    NOTE = "note"


class StructureMarkerKind(StrEnum):
    """Canonical structure markers emitted in the optional rich sequence mode."""

    PART = "part"
    STAFF = "staff"
    BAR = "bar"
    VOICE_LANE = "voice_lane"
    ONSET_GROUP = "onset_group"


_STRUCTURE_MARKER_PRIORITY = {
    StructureMarkerKind.PART: 0,
    StructureMarkerKind.STAFF: 1,
    StructureMarkerKind.BAR: 2,
    StructureMarkerKind.VOICE_LANE: 3,
    StructureMarkerKind.ONSET_GROUP: 4,
}


@dataclass(frozen=True)
class SequenceProjectionConfig:
    """Configurable inclusion options for the sequence projector."""

    mode: SequenceProjectionMode = SequenceProjectionMode.NOTES_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SequenceProjectionMode(self.mode))

    @property
    def include_controls(self) -> bool:
        """Return whether control events should be emitted."""
        return self.mode is not SequenceProjectionMode.NOTES_ONLY

    @property
    def include_structure_markers(self) -> bool:
        """Return whether structure markers should be emitted."""
        return (
            self.mode is SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS
        )


@dataclass(frozen=True)
class SequenceProjection:
    """Typed, time-ordered event sequence derived from a canonical IR document."""

    mode: SequenceProjectionMode
    events: tuple[SequenceEvent, ...]


@dataclass(frozen=True)
class StructureMarkerSequenceEvent:
    """Marker event for canonical score structure."""

    time: ScoreTime
    marker_kind: StructureMarkerKind
    entity_id: str
    part_id: str | None = None
    staff_id: str | None = None
    bar_id: str | None = None
    voice_lane_id: str | None = None

    def sort_key(self) -> tuple[ScoreTime, int, int, str]:
        """Return a canonical sort key."""
        return (
            self.time,
            0,
            _STRUCTURE_MARKER_PRIORITY[self.marker_kind],
            self.entity_id,
        )


@dataclass(frozen=True)
class PointControlSequenceEvent:
    """Point control event projected into sequential form."""

    time: ScoreTime
    control: PointControlEvent
    part_id: str | None = None
    staff_id: str | None = None
    bar_id: str | None = None
    voice_lane_id: str | None = None

    def sort_key(self) -> tuple[ScoreTime, int, str, str, str]:
        """Return a canonical sort key."""
        return (
            self.time,
            1,
            self.control.scope.value,
            self.control.target_ref,
            self.control.control_id,
        )


@dataclass(frozen=True)
class SpanControlSequenceEvent:
    """Span control event projected into sequential form."""

    time: ScoreTime
    control: SpanControlEvent
    part_id: str | None = None
    staff_id: str | None = None
    bar_id: str | None = None
    voice_lane_id: str | None = None

    def sort_key(self) -> tuple[ScoreTime, int, str, str, ScoreTime, str]:
        """Return a canonical sort key."""
        return (
            self.time,
            2,
            self.control.scope.value,
            self.control.target_ref,
            self.control.end_time,
            self.control.control_id,
        )


@dataclass(frozen=True)
class NoteSequenceEvent:
    """Note event projected into sequential form."""

    time: ScoreTime
    note: NoteEvent
    part_id: str
    staff_id: str
    bar_id: str
    voice_lane_id: str
    onset_id: str

    def sort_key(self) -> tuple[ScoreTime, int, str, str, tuple[Any, ...], str]:
        """Return a canonical sort key."""
        return (
            self.time,
            3,
            self.part_id,
            self.voice_lane_id,
            note_sort_key(self.note.string_number, self.note.pitch, self.note.note_id),
            self.note.note_id,
        )


SequenceEvent: TypeAlias = (
    StructureMarkerSequenceEvent
    | PointControlSequenceEvent
    | SpanControlSequenceEvent
    | NoteSequenceEvent
)


def project_sequence(
    document: MotifMlIrDocument,
    config: SequenceProjectionConfig | None = None,
) -> SequenceProjection:
    """Project a canonical IR document into a typed time-ordered sequence."""
    effective_config = config or SequenceProjectionConfig()
    # Canonical IR documents already preserve deterministic family ordering.
    parts = document.parts
    staves = document.staves
    bars = document.bars
    bars_by_id = {bar.bar_id: bar for bar in bars}
    del bars_by_id
    voice_lanes = document.voice_lanes
    onsets = document.onset_groups
    notes = document.note_events

    events: list[SequenceEvent] = []
    if effective_config.include_structure_markers:
        events.extend(
            _build_structure_markers(parts, staves, bars, voice_lanes, onsets)
        )

    if effective_config.include_controls:
        events.extend(_build_control_events(document))

    events.extend(_build_note_events(notes, onsets))

    return SequenceProjection(
        mode=effective_config.mode,
        events=tuple(sorted(events, key=lambda event: event.sort_key())),
    )


def _build_structure_markers(
    parts: tuple[Part, ...],
    staves: tuple[Staff, ...],
    bars: tuple[Bar, ...],
    voice_lanes: tuple[VoiceLane, ...],
    onsets: tuple[OnsetGroup, ...],
) -> list[StructureMarkerSequenceEvent]:
    bar_start_by_id = {bar.bar_id: bar.start for bar in bars}
    part_time_by_id = _build_part_time_index(
        parts, staves, voice_lanes, bar_start_by_id
    )
    staff_time_by_id = _build_staff_time_index(staves, voice_lanes, bar_start_by_id)

    events: list[StructureMarkerSequenceEvent] = []
    for part in parts:
        events.append(
            StructureMarkerSequenceEvent(
                time=part_time_by_id.get(part.part_id, _ZERO_TIME),
                marker_kind=StructureMarkerKind.PART,
                entity_id=part.part_id,
                part_id=part.part_id,
            )
        )

    for staff in staves:
        events.append(
            StructureMarkerSequenceEvent(
                time=staff_time_by_id.get(staff.staff_id, _ZERO_TIME),
                marker_kind=StructureMarkerKind.STAFF,
                entity_id=staff.staff_id,
                part_id=staff.part_id,
                staff_id=staff.staff_id,
            )
        )

    for bar in bars:
        events.append(
            StructureMarkerSequenceEvent(
                time=bar.start,
                marker_kind=StructureMarkerKind.BAR,
                entity_id=bar.bar_id,
                bar_id=bar.bar_id,
            )
        )

    for voice_lane in voice_lanes:
        events.append(
            StructureMarkerSequenceEvent(
                time=bar_start_by_id.get(voice_lane.bar_id, _ZERO_TIME),
                marker_kind=StructureMarkerKind.VOICE_LANE,
                entity_id=voice_lane.voice_lane_id,
                part_id=voice_lane.part_id,
                staff_id=voice_lane.staff_id,
                bar_id=voice_lane.bar_id,
                voice_lane_id=voice_lane.voice_lane_id,
            )
        )

    onset_time_by_id = {onset.onset_id: onset.time for onset in onsets}
    onset_voice_lane_by_id = {onset.onset_id: onset.voice_lane_id for onset in onsets}
    onset_bar_by_id = {onset.onset_id: onset.bar_id for onset in onsets}
    onset_part_by_id = _build_onset_part_index(onsets, voice_lanes)
    onset_staff_by_id = _build_onset_staff_index(onsets, voice_lanes)
    for onset in onsets:
        events.append(
            StructureMarkerSequenceEvent(
                time=onset_time_by_id[onset.onset_id],
                marker_kind=StructureMarkerKind.ONSET_GROUP,
                entity_id=onset.onset_id,
                part_id=onset_part_by_id.get(onset.onset_id),
                staff_id=onset_staff_by_id.get(onset.onset_id),
                bar_id=onset_bar_by_id.get(onset.onset_id),
                voice_lane_id=onset_voice_lane_by_id.get(onset.onset_id),
            )
        )

    return events


def _build_control_events(
    document: MotifMlIrDocument,
) -> list[PointControlSequenceEvent | SpanControlSequenceEvent]:
    voice_lane_by_id = {
        voice_lane.voice_lane_id: voice_lane for voice_lane in document.voice_lanes
    }
    staff_by_id = {staff.staff_id: staff for staff in document.staves}

    events: list[PointControlSequenceEvent | SpanControlSequenceEvent] = []
    for control in document.point_control_events:
        part_id, staff_id, voice_lane_id = _control_attribution(
            control.scope,
            control.target_ref,
            staff_by_id,
            voice_lane_by_id,
        )
        events.append(
            PointControlSequenceEvent(
                time=control.time,
                control=control,
                part_id=part_id,
                staff_id=staff_id,
                voice_lane_id=voice_lane_id,
            )
        )

    for control in document.span_control_events:
        part_id, staff_id, voice_lane_id = _control_attribution(
            control.scope,
            control.target_ref,
            staff_by_id,
            voice_lane_by_id,
        )
        events.append(
            SpanControlSequenceEvent(
                time=control.start_time,
                control=control,
                part_id=part_id,
                staff_id=staff_id,
                voice_lane_id=voice_lane_id,
            )
        )

    return events


def _build_note_events(
    notes: tuple[NoteEvent, ...],
    onsets: tuple[OnsetGroup, ...],
) -> list[NoteSequenceEvent]:
    onset_by_id = {onset.onset_id: onset for onset in onsets}
    events: list[NoteSequenceEvent] = []
    for note in notes:
        onset = onset_by_id[note.onset_id]
        events.append(
            NoteSequenceEvent(
                time=note.time,
                note=note,
                part_id=note.part_id,
                staff_id=note.staff_id,
                bar_id=onset.bar_id,
                voice_lane_id=onset.voice_lane_id,
                onset_id=note.onset_id,
            )
        )

    return events


def _build_part_time_index(
    parts: tuple[Part, ...],
    staves: tuple[Staff, ...],
    voice_lanes: tuple[VoiceLane, ...],
    bar_start_by_id: dict[str, ScoreTime],
) -> dict[str, ScoreTime]:
    bar_ids_by_part: dict[str, set[str]] = {}
    for voice_lane in voice_lanes:
        bar_ids_by_part.setdefault(voice_lane.part_id, set()).add(voice_lane.bar_id)

    part_time_by_id: dict[str, ScoreTime] = {}
    for part in parts:
        bar_times = [
            bar_start_by_id[bar_id]
            for bar_id in bar_ids_by_part.get(part.part_id, set())
            if bar_id in bar_start_by_id
        ]
        if bar_times:
            part_time_by_id[part.part_id] = min(bar_times)
            continue

        staff_bars = [
            bar_start_by_id[voice_lane.bar_id]
            for voice_lane in voice_lanes
            if voice_lane.part_id == part.part_id
            and voice_lane.bar_id in bar_start_by_id
        ]
        part_time_by_id[part.part_id] = min(staff_bars) if staff_bars else _ZERO_TIME

    return part_time_by_id


def _build_staff_time_index(
    staves: tuple[Staff, ...],
    voice_lanes: tuple[VoiceLane, ...],
    bar_start_by_id: dict[str, ScoreTime],
) -> dict[str, ScoreTime]:
    staff_time_by_id: dict[str, ScoreTime] = {}
    for staff in staves:
        staff_bars = [
            bar_start_by_id[voice_lane.bar_id]
            for voice_lane in voice_lanes
            if voice_lane.staff_id == staff.staff_id
            and voice_lane.bar_id in bar_start_by_id
        ]
        staff_time_by_id[staff.staff_id] = min(staff_bars) if staff_bars else _ZERO_TIME

    return staff_time_by_id


def _build_onset_part_index(
    onsets: tuple[OnsetGroup, ...],
    voice_lanes: tuple[VoiceLane, ...],
) -> dict[str, str]:
    voice_lane_by_id = {
        voice_lane.voice_lane_id: voice_lane for voice_lane in voice_lanes
    }
    return {
        onset.onset_id: voice_lane_by_id[onset.voice_lane_id].part_id
        for onset in onsets
        if onset.voice_lane_id in voice_lane_by_id
    }


def _build_onset_staff_index(
    onsets: tuple[OnsetGroup, ...],
    voice_lanes: tuple[VoiceLane, ...],
) -> dict[str, str]:
    voice_lane_by_id = {
        voice_lane.voice_lane_id: voice_lane for voice_lane in voice_lanes
    }
    return {
        onset.onset_id: voice_lane_by_id[onset.voice_lane_id].staff_id
        for onset in onsets
        if onset.voice_lane_id in voice_lane_by_id
    }


def _control_attribution(
    scope: ControlScope,
    target_ref: str,
    staff_by_id: dict[str, Staff],
    voice_lane_by_id: dict[str, VoiceLane],
) -> tuple[str | None, str | None, str | None]:
    if scope is ControlScope.PART:
        return target_ref, None, None
    if scope is ControlScope.STAFF:
        staff = staff_by_id.get(target_ref)
        if staff is not None:
            return staff.part_id, staff.staff_id, None
        return None, target_ref, None
    if scope is ControlScope.VOICE:
        voice_lane = voice_lane_by_id.get(target_ref)
        if voice_lane is not None:
            return voice_lane.part_id, voice_lane.staff_id, voice_lane.voice_lane_id
        return None, None, target_ref

    return None, None, None


__all__ = [
    "NoteSequenceEvent",
    "PointControlSequenceEvent",
    "SequenceEvent",
    "SequenceEventKind",
    "SequenceProjection",
    "SequenceProjectionConfig",
    "SequenceProjectionMode",
    "SpanControlSequenceEvent",
    "StructureMarkerKind",
    "StructureMarkerSequenceEvent",
    "project_sequence",
]
