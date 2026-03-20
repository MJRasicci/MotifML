"""Typed hierarchical projections for canonical MotifML IR documents."""

from __future__ import annotations

from dataclasses import dataclass, field

from motifml.ir.ids import (
    bar_sort_key,
    note_sort_key,
    onset_sort_key,
    part_sort_key,
    point_control_sort_key,
    span_control_sort_key,
    staff_sort_key,
    voice_lane_sort_key,
)
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


@dataclass(frozen=True)
class ControlAttachment:
    """Scope-local control events attached to one projection node."""

    point_controls: tuple[PointControlEvent, ...] = ()
    span_controls: tuple[SpanControlEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_controls",
            tuple(
                sorted(
                    self.point_controls,
                    key=lambda control: point_control_sort_key(
                        control.scope.value,
                        control.target_ref,
                        control.time,
                        control.control_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "span_controls",
            tuple(
                sorted(
                    self.span_controls,
                    key=lambda control: span_control_sort_key(
                        control.scope.value,
                        control.target_ref,
                        control.start_time,
                        control.end_time,
                        control.control_id,
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class NoteProjection:
    """Leaf node wrapping one canonical note event."""

    note_event: NoteEvent


@dataclass(frozen=True)
class OnsetProjection:
    """One onset group and its note children."""

    onset_group: OnsetGroup
    children: tuple[NoteProjection, ...] = ()
    controls: ControlAttachment = field(default_factory=ControlAttachment)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "children",
            tuple(
                sorted(
                    self.children,
                    key=lambda child: note_sort_key(
                        child.note_event.string_number,
                        child.note_event.pitch,
                        child.note_event.note_id,
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class VoiceLaneProjection:
    """One voice lane and its onset children."""

    voice_lane: VoiceLane
    children: tuple[OnsetProjection, ...] = ()
    controls: ControlAttachment = field(default_factory=ControlAttachment)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "children",
            tuple(
                sorted(
                    self.children,
                    key=lambda child: onset_sort_key(
                        child.onset_group.voice_lane_id,
                        child.onset_group.time,
                        child.onset_group.attack_order_in_voice,
                        child.onset_group.onset_id,
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class StaffProjection:
    """One staff and its scoped controls."""

    staff: Staff
    controls: ControlAttachment = field(default_factory=ControlAttachment)
    children: tuple[object, ...] = ()


@dataclass(frozen=True)
class PartProjection:
    """One part and its staff children."""

    part: Part
    children: tuple[StaffProjection, ...] = ()
    controls: ControlAttachment = field(default_factory=ControlAttachment)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "children",
            tuple(
                sorted(
                    self.children,
                    key=lambda child: staff_sort_key(
                        child.staff.part_id,
                        child.staff.staff_index,
                        child.staff.staff_id,
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class BarProjection:
    """One bar and its voice-lane children."""

    bar: Bar
    children: tuple[VoiceLaneProjection, ...] = ()
    controls: ControlAttachment = field(default_factory=ControlAttachment)

    def __post_init__(self) -> None:
        bar_index = self.bar.bar_index
        object.__setattr__(
            self,
            "children",
            tuple(
                sorted(
                    self.children,
                    key=lambda child: voice_lane_sort_key(
                        bar_index,
                        child.voice_lane.staff_id,
                        child.voice_lane.voice_index,
                        child.voice_lane.voice_lane_id,
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class HierarchicalProjection:
    """Typed hierarchical view over one canonical IR document."""

    score_controls: ControlAttachment = field(default_factory=ControlAttachment)
    parts: tuple[PartProjection, ...] = ()
    bars: tuple[BarProjection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parts",
            tuple(
                sorted(
                    self.parts,
                    key=lambda child: part_sort_key(child.part.part_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "bars",
            tuple(
                sorted(
                    self.bars,
                    key=lambda child: bar_sort_key(
                        child.bar.bar_index, child.bar.bar_id
                    ),
                )
            ),
        )

    @property
    def children(self) -> tuple[PartProjection | BarProjection, ...]:
        """Return all top-level branches in canonical order."""
        return self.parts + self.bars


def project_hierarchical(document: MotifMlIrDocument) -> HierarchicalProjection:
    """Project one canonical IR document into a typed containment tree."""
    staves_by_part_id: dict[str, list[Staff]] = {
        part.part_id: [] for part in document.parts
    }
    for staff in document.staves:
        staves_by_part_id.setdefault(staff.part_id, []).append(staff)

    voice_lanes_by_bar_id: dict[str, list[VoiceLane]] = {
        bar.bar_id: [] for bar in document.bars
    }
    for voice_lane in document.voice_lanes:
        voice_lanes_by_bar_id.setdefault(voice_lane.bar_id, []).append(voice_lane)

    onsets_by_voice_lane_id: dict[str, list[OnsetGroup]] = {
        voice_lane.voice_lane_id: [] for voice_lane in document.voice_lanes
    }
    for onset in document.onset_groups:
        onsets_by_voice_lane_id.setdefault(onset.voice_lane_id, []).append(onset)

    onset_order = {
        onset.onset_id: index
        for index, onset in enumerate(
            sorted(
                document.onset_groups,
                key=lambda item: onset_sort_key(
                    item.voice_lane_id,
                    item.time,
                    item.attack_order_in_voice,
                    item.onset_id,
                ),
            )
        )
    }
    notes_by_onset_id: dict[str, list[NoteEvent]] = {
        onset.onset_id: [] for onset in document.onset_groups
    }
    for note in document.note_events:
        notes_by_onset_id.setdefault(note.onset_id, []).append(note)

    score_controls = ControlAttachment(
        point_controls=tuple(
            control
            for control in document.point_control_events
            if control.scope is ControlScope.SCORE
        ),
        span_controls=tuple(
            control
            for control in document.span_control_events
            if control.scope is ControlScope.SCORE
        ),
    )
    point_controls_by_part = _group_point_controls(
        document.point_control_events, ControlScope.PART
    )
    span_controls_by_part = _group_span_controls(
        document.span_control_events, ControlScope.PART
    )
    point_controls_by_staff = _group_point_controls(
        document.point_control_events, ControlScope.STAFF
    )
    span_controls_by_staff = _group_span_controls(
        document.span_control_events, ControlScope.STAFF
    )
    point_controls_by_voice = _group_point_controls(
        document.point_control_events, ControlScope.VOICE
    )
    span_controls_by_voice = _group_span_controls(
        document.span_control_events, ControlScope.VOICE
    )

    part_nodes = []
    for part in sorted(document.parts, key=lambda item: part_sort_key(item.part_id)):
        staff_nodes = []
        for staff in sorted(
            staves_by_part_id.get(part.part_id, ()),
            key=lambda item: staff_sort_key(
                item.part_id, item.staff_index, item.staff_id
            ),
        ):
            staff_nodes.append(
                StaffProjection(
                    staff=staff,
                    controls=ControlAttachment(
                        point_controls=point_controls_by_staff.get(staff.staff_id, ()),
                        span_controls=span_controls_by_staff.get(staff.staff_id, ()),
                    ),
                )
            )

        part_nodes.append(
            PartProjection(
                part=part,
                controls=ControlAttachment(
                    point_controls=point_controls_by_part.get(part.part_id, ()),
                    span_controls=span_controls_by_part.get(part.part_id, ()),
                ),
                children=tuple(staff_nodes),
            )
        )

    bar_nodes = []
    for bar in sorted(
        document.bars, key=lambda item: bar_sort_key(item.bar_index, item.bar_id)
    ):
        voice_lane_nodes = []
        for voice_lane in sorted(
            voice_lanes_by_bar_id.get(bar.bar_id, ()),
            key=lambda item: voice_lane_sort_key(
                bar.bar_index,
                item.staff_id,
                item.voice_index,
                item.voice_lane_id,
            ),
        ):
            onset_nodes = []
            for onset in sorted(
                onsets_by_voice_lane_id.get(voice_lane.voice_lane_id, ()),
                key=lambda item: onset_sort_key(
                    item.voice_lane_id,
                    item.time,
                    item.attack_order_in_voice,
                    item.onset_id,
                ),
            ):
                note_nodes = []
                for note in sorted(
                    notes_by_onset_id.get(onset.onset_id, ()),
                    key=lambda item: (
                        onset_order[onset.onset_id],
                        note_sort_key(item.string_number, item.pitch, item.note_id),
                    ),
                ):
                    note_nodes.append(NoteProjection(note_event=note))

                onset_nodes.append(
                    OnsetProjection(
                        onset_group=onset,
                        children=tuple(note_nodes),
                    )
                )

            voice_lane_nodes.append(
                VoiceLaneProjection(
                    voice_lane=voice_lane,
                    children=tuple(onset_nodes),
                    controls=ControlAttachment(
                        point_controls=point_controls_by_voice.get(
                            voice_lane.voice_lane_id, ()
                        ),
                        span_controls=span_controls_by_voice.get(
                            voice_lane.voice_lane_id, ()
                        ),
                    ),
                )
            )

        bar_nodes.append(
            BarProjection(
                bar=bar,
                children=tuple(voice_lane_nodes),
            )
        )

    return HierarchicalProjection(
        score_controls=score_controls,
        parts=tuple(part_nodes),
        bars=tuple(bar_nodes),
    )


def _group_point_controls(
    controls: tuple[PointControlEvent, ...], scope: ControlScope
) -> dict[str, tuple[PointControlEvent, ...]]:
    grouped: dict[str, list[PointControlEvent]] = {}
    for control in controls:
        if control.scope is scope:
            grouped.setdefault(control.target_ref, []).append(control)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_span_controls(
    controls: tuple[SpanControlEvent, ...], scope: ControlScope
) -> dict[str, tuple[SpanControlEvent, ...]]:
    grouped: dict[str, list[SpanControlEvent]] = {}
    for control in controls:
        if control.scope is scope:
            grouped.setdefault(control.target_ref, []).append(control)
    return {key: tuple(value) for key, value in grouped.items()}


__all__ = [
    "BarProjection",
    "ControlAttachment",
    "HierarchicalProjection",
    "NoteProjection",
    "OnsetProjection",
    "PartProjection",
    "StaffProjection",
    "VoiceLaneProjection",
    "project_hierarchical",
]
