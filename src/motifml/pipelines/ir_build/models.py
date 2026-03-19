"""Typed models used by the IR build pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from motifml.ir.ids import (
    bar_sort_key,
    part_sort_key,
    point_control_sort_key,
    span_control_sort_key,
    staff_sort_key,
    voice_lane_sort_key,
)
from motifml.ir.models import (
    Bar,
    Part,
    PointControlEvent,
    SpanControlEvent,
    Staff,
    VoiceLane,
)
from motifml.ir.time import ScoreTime


class DiagnosticSeverity(StrEnum):
    """Supported severities for IR build diagnostics."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class IrBuildDiagnostic:
    """One deterministic diagnostic emitted during IR build validation."""

    severity: DiagnosticSeverity
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            DiagnosticSeverity(self.severity),
        )
        object.__setattr__(self, "code", _normalize_text(self.code, "code"))
        object.__setattr__(self, "path", _normalize_text(self.path, "path"))
        object.__setattr__(self, "message", _normalize_text(self.message, "message"))

    def sort_key(self) -> tuple[int, str, str, str]:
        """Return a stable diagnostic sort key."""
        severity_rank = 0 if self.severity is DiagnosticSeverity.ERROR else 1
        return (severity_rank, self.path, self.code, self.message)


@dataclass(frozen=True)
class CanonicalScoreValidationResult:
    """Typed validation result for one raw Motif JSON score surface."""

    relative_path: str
    source_hash: str
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


@dataclass(frozen=True)
class WrittenTimeMapEntry:
    """One bar-level written-time anchor emitted from `timelineBars`."""

    bar_index: int
    start: ScoreTime
    duration: ScoreTime
    is_anacrusis: bool = False

    def __post_init__(self) -> None:
        if self.bar_index < 0:
            raise ValueError("bar_index must be non-negative.")

        self.start.require_non_negative("WrittenTimeMapEntry start")
        if self.duration.numerator <= 0:
            raise ValueError("duration must be positive.")

    def sort_key(self) -> tuple[int, ScoreTime, ScoreTime]:
        """Return a stable written-time-map entry ordering key."""
        return (self.bar_index, self.start, self.duration)


@dataclass(frozen=True)
class WrittenTimeMapResult:
    """Typed written-time-map result for one validated raw score."""

    relative_path: str
    source_hash: str
    bars: tuple[WrittenTimeMapEntry, ...] = ()
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "bars",
            tuple(sorted(self.bars, key=lambda item: item.sort_key())),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)

    @property
    def bar_times(self) -> dict[int, tuple[ScoreTime, ScoreTime]]:
        """Return the result as a deterministic bar-indexed time map."""
        return {
            bar.bar_index: (bar.start, bar.duration)
            for bar in sorted(self.bars, key=lambda item: item.sort_key())
        }


@dataclass(frozen=True)
class PartStaffEmissionResult:
    """Typed part/staff emission result for one validated raw score."""

    relative_path: str
    source_hash: str
    parts: tuple[Part, ...] = ()
    staves: tuple[Staff, ...] = ()
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "parts",
            tuple(sorted(self.parts, key=lambda item: part_sort_key(item.part_id))),
        )
        object.__setattr__(
            self,
            "staves",
            tuple(
                sorted(
                    self.staves,
                    key=lambda item: staff_sort_key(
                        item.part_id,
                        item.staff_index,
                        item.staff_id,
                    ),
                )
            ),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


@dataclass(frozen=True)
class BarEmissionResult:
    """Typed bar emission result for one validated raw score."""

    relative_path: str
    source_hash: str
    bars: tuple[Bar, ...] = ()
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "bars",
            tuple(
                sorted(
                    self.bars,
                    key=lambda item: bar_sort_key(item.bar_index, item.bar_id),
                )
            ),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


@dataclass(frozen=True)
class VoiceLaneEmissionResult:
    """Typed voice-lane emission result for one validated raw score."""

    relative_path: str
    source_hash: str
    voice_lanes: tuple[VoiceLane, ...] = ()
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "voice_lanes",
            tuple(
                sorted(
                    self.voice_lanes,
                    key=lambda item: voice_lane_sort_key(
                        int(item.bar_id.split(":")[-1]),
                        item.staff_id,
                        item.voice_index,
                        item.voice_lane_id,
                    ),
                )
            ),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


@dataclass(frozen=True)
class PointControlEmissionResult:
    """Typed point-control emission result for one validated raw score."""

    relative_path: str
    source_hash: str
    point_control_events: tuple[PointControlEvent, ...] = ()
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "point_control_events",
            tuple(
                sorted(
                    self.point_control_events,
                    key=lambda item: point_control_sort_key(
                        item.scope,
                        item.target_ref,
                        item.time,
                        item.control_id,
                    ),
                )
            ),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


@dataclass(frozen=True)
class SpanControlEmissionResult:
    """Typed span-control emission result for one validated raw score."""

    relative_path: str
    source_hash: str
    span_control_events: tuple[SpanControlEvent, ...] = ()
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "span_control_events",
            tuple(
                sorted(
                    self.span_control_events,
                    key=lambda item: span_control_sort_key(
                        item.scope,
                        item.target_ref,
                        item.start_time,
                        item.end_time,
                        item.control_id,
                    ),
                )
            ),
        )

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


__all__ = [
    "CanonicalScoreValidationResult",
    "DiagnosticSeverity",
    "IrBuildDiagnostic",
    "BarEmissionResult",
    "PartStaffEmissionResult",
    "PointControlEmissionResult",
    "SpanControlEmissionResult",
    "VoiceLaneEmissionResult",
    "WrittenTimeMapEntry",
    "WrittenTimeMapResult",
]
