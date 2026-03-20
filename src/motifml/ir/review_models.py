"""Typed review-only models for inspecting MotifML IR documents."""

from __future__ import annotations

from dataclasses import dataclass

from motifml.ir.time import ScoreTime


@dataclass(frozen=True)
class ReviewNamedCount:
    """A deterministic name/count pair used in review summaries."""

    name: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_text(self.name, "name"))
        _require_non_negative(self.count, "count")


@dataclass(frozen=True)
class BarReviewRollup:
    """Review counts for one bar."""

    bar_id: str
    bar_index: int
    start: ScoreTime
    duration: ScoreTime
    voice_lane_count: int
    onset_count: int
    note_count: int
    point_control_count: int
    span_control_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar_id", _normalize_text(self.bar_id, "bar_id"))
        _require_non_negative(self.bar_index, "bar_index")
        _require_non_negative(self.voice_lane_count, "voice_lane_count")
        _require_non_negative(self.onset_count, "onset_count")
        _require_non_negative(self.note_count, "note_count")
        _require_non_negative(self.point_control_count, "point_control_count")
        _require_non_negative(self.span_control_count, "span_control_count")


@dataclass(frozen=True)
class VoiceLaneReviewRollup:
    """Review counts for one authored voice lane."""

    part_id: str
    staff_id: str
    bar_id: str
    bar_index: int
    voice_lane_id: str
    voice_lane_chain_id: str
    voice_index: int
    onset_count: int
    note_count: int
    rest_onset_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "part_id",
            "staff_id",
            "bar_id",
            "voice_lane_id",
            "voice_lane_chain_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        _require_non_negative(self.bar_index, "bar_index")
        _require_non_negative(self.voice_index, "voice_index")
        _require_non_negative(self.onset_count, "onset_count")
        _require_non_negative(self.note_count, "note_count")
        _require_non_negative(self.rest_onset_count, "rest_onset_count")


@dataclass(frozen=True)
class IrStructureSummary:
    """Deterministic structure rollup for one IR document."""

    part_count: int
    staff_count: int
    bar_count: int
    voice_lane_count: int
    onset_count: int
    note_count: int
    point_control_count: int
    span_control_count: int
    edge_count: int
    edge_counts_by_type: tuple[ReviewNamedCount, ...] = ()
    bar_rollups: tuple[BarReviewRollup, ...] = ()
    voice_lane_rollups: tuple[VoiceLaneReviewRollup, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "part_count",
            "staff_count",
            "bar_count",
            "voice_lane_count",
            "onset_count",
            "note_count",
            "point_control_count",
            "span_control_count",
            "edge_count",
        ):
            _require_non_negative(getattr(self, field_name), field_name)

        object.__setattr__(
            self,
            "edge_counts_by_type",
            tuple(
                sorted(self.edge_counts_by_type, key=lambda item: item.name.casefold())
            ),
        )
        object.__setattr__(
            self,
            "bar_rollups",
            tuple(
                sorted(
                    self.bar_rollups,
                    key=lambda item: (item.bar_index, item.bar_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "voice_lane_rollups",
            tuple(
                sorted(
                    self.voice_lane_rollups,
                    key=lambda item: (
                        item.bar_index,
                        item.staff_id.casefold(),
                        item.voice_index,
                        item.voice_lane_id.casefold(),
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class VoiceLaneOnsetRow:
    """One onset row for a per-bar / per-voice review table."""

    part_id: str
    staff_id: str
    bar_id: str
    bar_index: int
    voice_lane_id: str
    voice_lane_chain_id: str
    voice_index: int
    onset_id: str
    time: ScoreTime
    bar_offset: ScoreTime
    duration_notated: ScoreTime
    duration_sounding_max: ScoreTime | None
    is_rest: bool
    attack_order_in_voice: int
    note_count: int
    grace_type: str | None = None
    dynamic_local: str | None = None
    technique_summary: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "part_id",
            "staff_id",
            "bar_id",
            "voice_lane_id",
            "voice_lane_chain_id",
            "onset_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        _require_non_negative(self.bar_index, "bar_index")
        _require_non_negative(self.voice_index, "voice_index")
        _require_non_negative(self.attack_order_in_voice, "attack_order_in_voice")
        _require_non_negative(self.note_count, "note_count")

        if self.grace_type is not None:
            object.__setattr__(
                self,
                "grace_type",
                _normalize_text(self.grace_type, "grace_type"),
            )

        if self.dynamic_local is not None:
            object.__setattr__(
                self,
                "dynamic_local",
                _normalize_text(self.dynamic_local, "dynamic_local"),
            )

        if self.technique_summary is not None:
            object.__setattr__(
                self,
                "technique_summary",
                _normalize_text(self.technique_summary, "technique_summary"),
            )


@dataclass(frozen=True)
class VoiceLaneOnsetTable:
    """A grouped onset table for one authored voice lane."""

    part_id: str
    staff_id: str
    bar_id: str
    bar_index: int
    voice_lane_id: str
    voice_lane_chain_id: str
    voice_index: int
    rows: tuple[VoiceLaneOnsetRow, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "part_id",
            "staff_id",
            "bar_id",
            "voice_lane_id",
            "voice_lane_chain_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        _require_non_negative(self.bar_index, "bar_index")
        _require_non_negative(self.voice_index, "voice_index")

        normalized_rows = tuple(
            sorted(
                self.rows,
                key=lambda item: (
                    item.time,
                    item.attack_order_in_voice,
                    item.onset_id.casefold(),
                ),
            )
        )
        if not normalized_rows:
            raise ValueError("rows must contain at least one onset row.")

        object.__setattr__(self, "rows", normalized_rows)


@dataclass(frozen=True)
class OnsetNoteRow:
    """One note row grouped under an onset for inspection."""

    part_id: str
    staff_id: str
    bar_id: str
    bar_index: int
    voice_lane_id: str
    voice_lane_chain_id: str
    voice_index: int
    onset_id: str
    note_id: str
    time: ScoreTime
    bar_offset: ScoreTime
    pitch_text: str
    attack_duration: ScoreTime
    sounding_duration: ScoreTime
    string_number: int | None = None
    velocity: int | None = None
    technique_summary: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "part_id",
            "staff_id",
            "bar_id",
            "voice_lane_id",
            "voice_lane_chain_id",
            "onset_id",
            "note_id",
            "pitch_text",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        _require_non_negative(self.bar_index, "bar_index")
        _require_non_negative(self.voice_index, "voice_index")

        if self.string_number is not None:
            _require_non_negative(self.string_number, "string_number")

        if self.velocity is not None:
            _require_non_negative(self.velocity, "velocity")

        if self.technique_summary is not None:
            object.__setattr__(
                self,
                "technique_summary",
                _normalize_text(self.technique_summary, "technique_summary"),
            )


@dataclass(frozen=True)
class OnsetNoteTable:
    """A grouped note table for one onset."""

    part_id: str
    staff_id: str
    bar_id: str
    bar_index: int
    voice_lane_id: str
    voice_lane_chain_id: str
    voice_index: int
    onset_id: str
    onset_time: ScoreTime
    onset_bar_offset: ScoreTime
    rows: tuple[OnsetNoteRow, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "part_id",
            "staff_id",
            "bar_id",
            "voice_lane_id",
            "voice_lane_chain_id",
            "onset_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        _require_non_negative(self.bar_index, "bar_index")
        _require_non_negative(self.voice_index, "voice_index")

        normalized_rows = tuple(
            sorted(
                self.rows,
                key=lambda item: (
                    item.string_number is None,
                    item.string_number if item.string_number is not None else 0,
                    item.pitch_text.casefold(),
                    item.note_id.casefold(),
                ),
            )
        )
        if not normalized_rows:
            raise ValueError("rows must contain at least one note row.")

        object.__setattr__(self, "rows", normalized_rows)


@dataclass(frozen=True)
class ControlEventRow:
    """A normalized point- or span-control row for inspection."""

    control_id: str
    family: str
    kind: str
    scope: str
    target_ref: str
    start_time: ScoreTime
    end_time: ScoreTime | None
    start_bar_index: int | None
    end_bar_index: int | None
    value_summary: str
    start_anchor_ref: str | None = None
    end_anchor_ref: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "control_id",
            "family",
            "kind",
            "scope",
            "target_ref",
            "value_summary",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        if self.start_bar_index is not None:
            _require_non_negative(self.start_bar_index, "start_bar_index")

        if self.end_bar_index is not None:
            _require_non_negative(self.end_bar_index, "end_bar_index")

        if self.start_anchor_ref is not None:
            object.__setattr__(
                self,
                "start_anchor_ref",
                _normalize_text(self.start_anchor_ref, "start_anchor_ref"),
            )

        if self.end_anchor_ref is not None:
            object.__setattr__(
                self,
                "end_anchor_ref",
                _normalize_text(self.end_anchor_ref, "end_anchor_ref"),
            )


@dataclass(frozen=True)
class IrReviewBundleManifest:
    """Deterministic metadata describing one generated review bundle."""

    bundle_version: str
    fixture_id: str
    description: str
    source_path: str
    source_hash: str
    ir_document_path: str
    schema_validation_passed: bool
    validation_error_count: int
    validation_warning_count: int
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "bundle_version",
            "fixture_id",
            "description",
            "source_path",
            "source_hash",
            "ir_document_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )

        _require_non_negative(self.validation_error_count, "validation_error_count")
        _require_non_negative(self.validation_warning_count, "validation_warning_count")
        object.__setattr__(
            self,
            "artifacts",
            tuple(
                sorted(_normalize_text(item, "artifacts") for item in self.artifacts)
            ),
        )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


def _require_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


__all__ = [
    "BarReviewRollup",
    "ControlEventRow",
    "IrReviewBundleManifest",
    "IrStructureSummary",
    "OnsetNoteRow",
    "OnsetNoteTable",
    "ReviewNamedCount",
    "VoiceLaneOnsetRow",
    "VoiceLaneOnsetTable",
    "VoiceLaneReviewRollup",
]
