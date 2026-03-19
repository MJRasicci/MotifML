"""Nodes for early IR build validation and time-foundation construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.ids import (
    bar_id as build_bar_id,
)
from motifml.ir.ids import (
    note_id as build_note_id,
)
from motifml.ir.ids import (
    note_sort_key,
)
from motifml.ir.ids import (
    onset_id as build_onset_id,
)
from motifml.ir.ids import (
    part_id as build_part_id,
)
from motifml.ir.ids import (
    point_control_id as build_point_control_id,
)
from motifml.ir.ids import (
    span_control_id as build_span_control_id,
)
from motifml.ir.ids import (
    staff_id as build_staff_id,
)
from motifml.ir.ids import (
    voice_lane_chain_id as build_voice_lane_chain_id,
)
from motifml.ir.ids import (
    voice_lane_id as build_voice_lane_id,
)
from motifml.ir.models import (
    Bar,
    ControlScope,
    DynamicChangeValue,
    FermataValue,
    GenericTechniqueFlags,
    HairpinDirection,
    HairpinValue,
    NoteEvent,
    OnsetGroup,
    OttavaValue,
    Part,
    Pitch,
    PointControlEvent,
    PointControlKind,
    RhythmBaseValue,
    RhythmShape,
    SpanControlEvent,
    SpanControlKind,
    Staff,
    StringFrettedTechniquePayload,
    TechniquePayload,
    TempoChangeValue,
    TimeSignature,
    Transposition,
    TupletRatio,
    VoiceLane,
)
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.models import (
    BarEmissionResult,
    CanonicalScoreValidationResult,
    DiagnosticSeverity,
    IrBuildDiagnostic,
    NoteEventEmissionResult,
    OnsetGroupEmissionResult,
    PartStaffEmissionResult,
    PointControlEmissionResult,
    SpanControlEmissionResult,
    VoiceLaneEmissionResult,
    WrittenTimeMapEntry,
    WrittenTimeMapResult,
)

REQUIRED_TOP_LEVEL_LIST_FIELDS = (
    "pointControls",
    "spanControls",
    "timelineBars",
    "tracks",
)
RELATION_HINT_FIELDS = (
    "hopoDestination",
    "hopoOrigin",
    "hopoType",
    "slides",
    "tieDestination",
    "tieOrigin",
)
POINT_CONTROL_KIND_MAP = {
    "Dynamic": PointControlKind.DYNAMIC_CHANGE,
    "Fermata": PointControlKind.FERMATA,
    "Tempo": PointControlKind.TEMPO_CHANGE,
}
SPAN_CONTROL_KIND_MAP = {
    "Hairpin": SpanControlKind.HAIRPIN,
    "Ottava": SpanControlKind.OTTAVA,
}
SOURCE_SCOPE_MAP = {
    "Score": ControlScope.SCORE,
    "Track": ControlScope.PART,
    "Staff": ControlScope.STAFF,
    "Voice": ControlScope.VOICE,
}
UNSUPPORTED_ONSET_TECHNIQUE_FIELDS = (
    "arpeggio",
    "brush",
    "brushIsUp",
    "deadSlapped",
    "popped",
    "rasgueado",
    "slapped",
    "slashed",
    "tremolo",
)


@dataclass(frozen=True)
class _VoiceLaneBuildContext:
    part_identifier: str
    staff_identifier: str
    bar_index: int
    bar_id: str
    voice_path: str


@dataclass(frozen=True)
class _VoiceLaneStaffBuildContext:
    part_identifier: str
    staff_path: str
    known_staff_ids: frozenset[str]
    bar_ids_by_index: dict[int, str]


@dataclass(frozen=True)
class _OnsetStaffBuildContext:
    part_identifier: str
    staff_path: str
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True)
class _OnsetVoiceBuildContext:
    staff_identifier: str
    bar_index: int
    bar_id: str
    bar_start: ScoreTime
    bar_duration: ScoreTime
    voice_path: str
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True)
class _OnsetBeatBuildContext:
    voice_lane_id: str
    bar_index: int
    bar_id: str
    bar_start: ScoreTime
    bar_duration: ScoreTime
    beat_path: str
    attack_order_in_voice: int


@dataclass(frozen=True)
class _NoteTrackBuildContext:
    track_path: str
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    voice_lanes_by_id: dict[str, VoiceLane]
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]]


@dataclass(frozen=True)
class _NoteStaffBuildContext:
    part_identifier: str
    staff_path: str
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    voice_lanes_by_id: dict[str, VoiceLane]
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]]


@dataclass(frozen=True)
class _NoteVoiceTraversalContext:
    staff_identifier: str
    bar_index: int
    bar_duration: ScoreTime
    voice_path: str
    voice_lanes_by_id: dict[str, VoiceLane]
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]]


@dataclass(frozen=True)
class _NoteBeatBuildContext:
    voice_lane: VoiceLane


@dataclass(frozen=True)
class _NoteSeed:
    path: str
    onset_id: str
    part_id: str
    staff_id: str
    time: ScoreTime
    attack_duration: ScoreTime
    sounding_duration: ScoreTime
    pitch: Pitch | None
    velocity: int | None
    string_number: int | None
    show_string_number: bool | None
    techniques: TechniquePayload | None

    def sort_key(self) -> tuple[object, ...]:
        return note_sort_key(
            self.string_number,
            self.pitch,
            self.path,
        )


@dataclass(frozen=True)
class _ControlResolutionContext:
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    known_part_ids: frozenset[str]
    known_staff_ids: frozenset[str]
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True)
class _ResolvedControlPosition:
    bar_index: int
    time: ScoreTime


@dataclass(frozen=True)
class _VoiceLaneTargetResolutionContext:
    control_path: str
    staff_identifier: str
    bar_index: int
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True)
class _PointControlSeed:
    path: str
    scope: ControlScope
    target_ref: str
    time: ScoreTime
    kind: PointControlKind
    value: TempoChangeValue | DynamicChangeValue | FermataValue

    def sort_key(self) -> tuple[str, str, ScoreTime, str, tuple[object, ...], str]:
        return (
            self.scope.value.casefold(),
            self.target_ref,
            self.time,
            self.kind.value,
            _point_control_value_sort_key(self.value),
            self.path,
        )


@dataclass(frozen=True)
class _SpanControlSeed:
    path: str
    scope: ControlScope
    target_ref: str
    start_time: ScoreTime
    end_time: ScoreTime
    kind: SpanControlKind
    value: HairpinValue | OttavaValue
    start_anchor_ref: str | None = None
    end_anchor_ref: str | None = None

    def sort_key(
        self,
    ) -> tuple[str, str, ScoreTime, ScoreTime, str, tuple[object, ...], str]:
        return (
            self.scope.value.casefold(),
            self.target_ref,
            self.start_time,
            self.end_time,
            self.kind.value,
            _span_control_value_sort_key(self.value),
            self.path,
        )


def validate_canonical_score_surface(
    documents: list[MotifJsonDocument],
) -> list[CanonicalScoreValidationResult]:
    """Validate that raw Motif JSON documents expose the canonical IR input surface.

    Args:
        documents: Raw Motif JSON corpus documents loaded from the `01_raw` stage.

    Returns:
        Deterministic validation results with fatal errors and recoverable warnings.
    """
    return [
        CanonicalScoreValidationResult(
            relative_path=document.relative_path,
            source_hash=document.sha256,
            diagnostics=tuple(_validate_document_surface(document.score)),
        )
        for document in sorted(
            documents, key=lambda item: item.relative_path.casefold()
        )
    ]


def build_written_time_map(
    documents: list[MotifJsonDocument],
    validation_results: list[CanonicalScoreValidationResult],
) -> list[WrittenTimeMapResult]:
    """Build deterministic bar-level written time maps from canonical timeline bars.

    Args:
        documents: Raw Motif JSON corpus documents.
        validation_results: Surface-validation results from
            `validate_canonical_score_surface`.

    Returns:
        One typed written-time-map result per input document.
    """
    validation_results_by_path = {
        result.relative_path: result for result in validation_results
    }
    written_time_maps: list[WrittenTimeMapResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        validation_result = validation_results_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        bars: tuple[WrittenTimeMapEntry, ...] = ()

        if validation_result is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_validation_result",
                    message=(
                        "canonical score surface validation must run before written "
                        "time-map construction."
                    ),
                )
            )
        elif not validation_result.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="canonical_surface_validation_failed",
                    message=(
                        "written time map cannot be built because the canonical "
                        "score surface validation failed."
                    ),
                )
            )
        else:
            bars, built_diagnostics = _build_written_time_map_entries(document.score)
            diagnostics.extend(built_diagnostics)

        written_time_maps.append(
            WrittenTimeMapResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                bars=bars,
                diagnostics=tuple(diagnostics),
            )
        )

    return written_time_maps


def emit_parts_and_staves(
    documents: list[MotifJsonDocument],
    validation_results: list[CanonicalScoreValidationResult],
) -> list[PartStaffEmissionResult]:
    """Emit IR parts and staves from validated raw Motif track structures."""
    validation_results_by_path = {
        result.relative_path: result for result in validation_results
    }
    emissions: list[PartStaffEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        validation_result = validation_results_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        parts: tuple[Part, ...] = ()
        staves: tuple[Staff, ...] = ()

        if validation_result is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_validation_result",
                    message=(
                        "canonical score surface validation must run before part "
                        "and staff emission."
                    ),
                )
            )
        elif not validation_result.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="canonical_surface_validation_failed",
                    message=(
                        "parts and staves cannot be emitted because the canonical "
                        "score surface validation failed."
                    ),
                )
            )
        else:
            parts, staves, emitted_diagnostics = _emit_part_staff_entities(
                document.score
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            PartStaffEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                parts=parts,
                staves=staves,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def emit_bars(
    documents: list[MotifJsonDocument],
    written_time_maps: list[WrittenTimeMapResult],
) -> list[BarEmissionResult]:
    """Emit IR bars from raw timeline metadata and written time maps."""
    written_time_maps_by_path = {
        result.relative_path: result for result in written_time_maps
    }
    emissions: list[BarEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        written_time_map = written_time_maps_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        bars: tuple[Bar, ...] = ()

        if written_time_map is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_written_time_map",
                    message="written time map emission must run before bar emission.",
                )
            )
        elif not written_time_map.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="written_time_map_failed",
                    message=(
                        "bars cannot be emitted because the written time map "
                        "contains fatal diagnostics."
                    ),
                )
            )
        else:
            bars, emitted_diagnostics = _emit_bar_models(
                score=document.score,
                written_time_map=written_time_map,
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            BarEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                bars=bars,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def emit_voice_lanes(
    documents: list[MotifJsonDocument],
    part_staff_emissions: list[PartStaffEmissionResult],
    bar_emissions: list[BarEmissionResult],
) -> list[VoiceLaneEmissionResult]:
    """Emit bar-scoped voice lanes from validated raw voice structures."""
    part_staff_by_path = {
        result.relative_path: result for result in part_staff_emissions
    }
    bars_by_path = {result.relative_path: result for result in bar_emissions}
    emissions: list[VoiceLaneEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        part_staff_emission = part_staff_by_path.get(document.relative_path)
        bar_emission = bars_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        voice_lanes: tuple[VoiceLane, ...] = ()

        if part_staff_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_part_staff_emission",
                    message="part/staff emission must run before voice lane emission.",
                )
            )
        elif bar_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_bar_emission",
                    message="bar emission must run before voice lane emission.",
                )
            )
        elif not part_staff_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="part_staff_emission_failed",
                    message=(
                        "voice lanes cannot be emitted because part/staff emission "
                        "contains fatal diagnostics."
                    ),
                )
            )
        elif not bar_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="bar_emission_failed",
                    message=(
                        "voice lanes cannot be emitted because bar emission "
                        "contains fatal diagnostics."
                    ),
                )
            )
        else:
            voice_lanes, emitted_diagnostics = _emit_voice_lane_models(
                score=document.score,
                part_staff_emission=part_staff_emission,
                bar_emission=bar_emission,
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            VoiceLaneEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                voice_lanes=voice_lanes,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def emit_onset_groups(
    documents: list[MotifJsonDocument],
    written_time_maps: list[WrittenTimeMapResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
) -> list[OnsetGroupEmissionResult]:
    """Emit onset groups from voice-scoped beats."""
    written_time_maps_by_path = {
        result.relative_path: result for result in written_time_maps
    }
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    emissions: list[OnsetGroupEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        written_time_map = written_time_maps_by_path.get(document.relative_path)
        voice_lane_emission = voice_lane_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        onset_groups: tuple[OnsetGroup, ...] = ()

        if written_time_map is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_written_time_map",
                    message="written time map must run before onset group emission.",
                )
            )
        elif voice_lane_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_voice_lane_emission",
                    message="voice lane emission must run before onset group emission.",
                )
            )
        elif not written_time_map.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="written_time_map_failed",
                    message=(
                        "onset groups cannot be emitted because the written time map "
                        "contains fatal diagnostics."
                    ),
                )
            )
        elif not voice_lane_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="voice_lane_emission_failed",
                    message=(
                        "onset groups cannot be emitted because voice lane emission "
                        "contains fatal diagnostics."
                    ),
                )
            )
        else:
            onset_groups, emitted_diagnostics = _emit_onset_group_models(
                score=document.score,
                written_time_map=written_time_map,
                voice_lane_emission=voice_lane_emission,
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            OnsetGroupEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                onset_groups=onset_groups,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def emit_note_events(
    documents: list[MotifJsonDocument],
    written_time_maps: list[WrittenTimeMapResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
    onset_group_emissions: list[OnsetGroupEmissionResult],
) -> list[NoteEventEmissionResult]:
    """Emit note events from onset-group-aligned beat notes."""
    written_time_maps_by_path = {
        result.relative_path: result for result in written_time_maps
    }
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    onset_group_by_path = {
        result.relative_path: result for result in onset_group_emissions
    }
    emissions: list[NoteEventEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        written_time_map = written_time_maps_by_path.get(document.relative_path)
        voice_lane_emission = voice_lane_by_path.get(document.relative_path)
        onset_group_emission = onset_group_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        note_events: tuple[NoteEvent, ...] = ()

        if written_time_map is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_written_time_map",
                    message="written time map must run before note event emission.",
                )
            )
        elif voice_lane_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_voice_lane_emission",
                    message="voice lane emission must run before note event emission.",
                )
            )
        elif onset_group_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_onset_group_emission",
                    message="onset group emission must run before note event emission.",
                )
            )
        elif not written_time_map.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="written_time_map_failed",
                    message=(
                        "note events cannot be emitted because the written time map "
                        "contains fatal diagnostics."
                    ),
                )
            )
        elif not voice_lane_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="voice_lane_emission_failed",
                    message=(
                        "note events cannot be emitted because voice lane emission "
                        "contains fatal diagnostics."
                    ),
                )
            )
        elif not onset_group_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="onset_group_emission_failed",
                    message=(
                        "note events cannot be emitted because onset group emission "
                        "contains fatal diagnostics."
                    ),
                )
            )
        else:
            note_events, emitted_diagnostics = _emit_note_event_models(
                score=document.score,
                written_time_map=written_time_map,
                voice_lane_emission=voice_lane_emission,
                onset_group_emission=onset_group_emission,
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            NoteEventEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                note_events=note_events,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def emit_point_control_events(
    documents: list[MotifJsonDocument],
    written_time_maps: list[WrittenTimeMapResult],
    part_staff_emissions: list[PartStaffEmissionResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
) -> list[PointControlEmissionResult]:
    """Emit canonical point control events from score-level point controls."""
    written_time_maps_by_path = {
        result.relative_path: result for result in written_time_maps
    }
    part_staff_by_path = {
        result.relative_path: result for result in part_staff_emissions
    }
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    emissions: list[PointControlEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        written_time_map = written_time_maps_by_path.get(document.relative_path)
        part_staff_emission = part_staff_by_path.get(document.relative_path)
        voice_lane_emission = voice_lane_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        point_control_events: tuple[PointControlEvent, ...] = ()

        if written_time_map is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_written_time_map",
                    message="written time map must run before point control emission.",
                )
            )
        elif part_staff_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_part_staff_emission",
                    message="part/staff emission must run before point control emission.",
                )
            )
        elif voice_lane_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_voice_lane_emission",
                    message="voice lane emission must run before point control emission.",
                )
            )
        else:
            point_control_events, emitted_diagnostics = _emit_point_control_models(
                score=document.score,
                resolution_context=_ControlResolutionContext(
                    bar_times=written_time_map.bar_times,
                    known_part_ids=frozenset(
                        part.part_id for part in part_staff_emission.parts
                    ),
                    known_staff_ids=frozenset(
                        staff.staff_id for staff in part_staff_emission.staves
                    ),
                    known_voice_lane_ids=frozenset(
                        voice_lane.voice_lane_id
                        for voice_lane in voice_lane_emission.voice_lanes
                    ),
                ),
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            PointControlEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                point_control_events=point_control_events,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def emit_span_control_events(
    documents: list[MotifJsonDocument],
    written_time_maps: list[WrittenTimeMapResult],
    part_staff_emissions: list[PartStaffEmissionResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
) -> list[SpanControlEmissionResult]:
    """Emit canonical span control events from score-level span controls."""
    written_time_maps_by_path = {
        result.relative_path: result for result in written_time_maps
    }
    part_staff_by_path = {
        result.relative_path: result for result in part_staff_emissions
    }
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    emissions: list[SpanControlEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        written_time_map = written_time_maps_by_path.get(document.relative_path)
        part_staff_emission = part_staff_by_path.get(document.relative_path)
        voice_lane_emission = voice_lane_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        span_control_events: tuple[SpanControlEvent, ...] = ()

        if written_time_map is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_written_time_map",
                    message="written time map must run before span control emission.",
                )
            )
        elif part_staff_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_part_staff_emission",
                    message="part/staff emission must run before span control emission.",
                )
            )
        elif voice_lane_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_voice_lane_emission",
                    message="voice lane emission must run before span control emission.",
                )
            )
        else:
            span_control_events, emitted_diagnostics = _emit_span_control_models(
                score=document.score,
                resolution_context=_ControlResolutionContext(
                    bar_times=written_time_map.bar_times,
                    known_part_ids=frozenset(
                        part.part_id for part in part_staff_emission.parts
                    ),
                    known_staff_ids=frozenset(
                        staff.staff_id for staff in part_staff_emission.staves
                    ),
                    known_voice_lane_ids=frozenset(
                        voice_lane.voice_lane_id
                        for voice_lane in voice_lane_emission.voice_lanes
                    ),
                ),
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            SpanControlEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                span_control_events=span_control_events,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def _validate_document_surface(score: dict[str, Any]) -> list[IrBuildDiagnostic]:
    diagnostics: list[IrBuildDiagnostic] = []

    top_level_lists = {
        field_name: _require_list_field(
            score,
            field_name,
            path=field_name,
            diagnostics=diagnostics,
        )
        for field_name in REQUIRED_TOP_LEVEL_LIST_FIELDS
    }

    timeline_bars = top_level_lists["timelineBars"]
    if timeline_bars is not None:
        _validate_timeline_bars(timeline_bars, diagnostics)

    tracks = top_level_lists["tracks"]
    if tracks is not None:
        _validate_track_surfaces(tracks, diagnostics)

    return diagnostics


def _validate_timeline_bars(
    timeline_bars: list[object],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for bar_index, timeline_bar in enumerate(timeline_bars):
        path = f"timelineBars[{bar_index}]"
        if not isinstance(timeline_bar, Mapping):
            diagnostics.append(
                _error(
                    path=path,
                    code="invalid_canonical_field",
                    message="timelineBars entries must be objects.",
                )
            )
            continue

        _require_score_time_field(
            timeline_bar,
            field_name="start",
            path=f"{path}.start",
            diagnostics=diagnostics,
            require_positive=False,
        )
        _require_score_time_field(
            timeline_bar,
            field_name="duration",
            path=f"{path}.duration",
            diagnostics=diagnostics,
            require_positive=True,
        )


def _build_written_time_map_entries(
    score: dict[str, Any],
) -> tuple[tuple[WrittenTimeMapEntry, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    timeline_bars = score.get("timelineBars")
    if not isinstance(timeline_bars, list):
        diagnostics.append(
            _error(
                path="timelineBars",
                code="missing_canonical_field",
                message="canonical field 'timelineBars' is required.",
            )
        )
        return (), diagnostics

    raw_entries: list[tuple[int, int, ScoreTime, ScoreTime]] = []
    seen_bar_indexes: set[int] = set()
    anacrusis_flag = _coerce_optional_bool(score.get("anacrusis"), diagnostics)

    for ordinal, timeline_bar in enumerate(timeline_bars):
        timeline_bar_path = f"timelineBars[{ordinal}]"
        if not isinstance(timeline_bar, Mapping):
            diagnostics.append(
                _error(
                    path=timeline_bar_path,
                    code="invalid_canonical_field",
                    message="timelineBars entries must be objects.",
                )
            )
            continue

        bar_index = timeline_bar.get("index")
        if not isinstance(bar_index, int):
            diagnostics.append(
                _error(
                    path=f"{timeline_bar_path}.index",
                    code="invalid_canonical_field",
                    message="timelineBars entries must include an integer index.",
                )
            )
            continue

        if bar_index in seen_bar_indexes:
            diagnostics.append(
                _error(
                    path=f"{timeline_bar_path}.index",
                    code="duplicate_bar_index",
                    message=f"bar index {bar_index} appears more than once.",
                )
            )
            continue

        start = _coerce_score_time(
            timeline_bar.get("start"),
            path=f"{timeline_bar_path}.start",
            diagnostics=diagnostics,
            require_positive=False,
        )
        duration = _coerce_score_time(
            timeline_bar.get("duration"),
            path=f"{timeline_bar_path}.duration",
            diagnostics=diagnostics,
            require_positive=True,
        )
        if start is None or duration is None:
            continue

        seen_bar_indexes.add(bar_index)
        raw_entries.append((bar_index, ordinal, start, duration))

    raw_entries.sort(key=lambda item: item[0])

    bars: list[WrittenTimeMapEntry] = []
    previous_entry: WrittenTimeMapEntry | None = None
    for position, (bar_index, ordinal, start, duration) in enumerate(raw_entries):
        bar = WrittenTimeMapEntry(
            bar_index=bar_index,
            start=start,
            duration=duration,
            is_anacrusis=anacrusis_flag and position == 0,
        )
        if previous_entry is not None:
            expected_start = previous_entry.start + previous_entry.duration
            if bar.start < expected_start:
                diagnostics.append(
                    _error(
                        path=f"timelineBars[{ordinal}].start",
                        code="overlapping_bar_geometry",
                        message=(
                            "timeline bars must be contiguous and non-overlapping; "
                            "this bar starts before the previous bar ends."
                        ),
                    )
                )
            elif bar.start > expected_start:
                diagnostics.append(
                    _error(
                        path=f"timelineBars[{ordinal}].start",
                        code="non_contiguous_bar_geometry",
                        message=(
                            "timeline bars must be contiguous and non-overlapping; "
                            "this bar starts after a gap."
                        ),
                    )
                )

        bars.append(bar)
        previous_entry = bar

    diagnostics.extend(_build_bar_geometry_warnings(score, raw_entries, tuple(bars)))
    return tuple(bars), diagnostics


def _emit_part_staff_entities(
    score: dict[str, Any],
) -> tuple[tuple[Part, ...], tuple[Staff, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    tracks = score.get("tracks")
    if not isinstance(tracks, list):
        diagnostics.append(
            _error(
                path="tracks",
                code="missing_canonical_field",
                message="canonical field 'tracks' is required.",
            )
        )
        return (), (), diagnostics

    parts: list[Part] = []
    staves: list[Staff] = []
    seen_part_ids: set[str] = set()

    for track_index, track in enumerate(tracks):
        track_path = f"tracks[{track_index}]"
        if not isinstance(track, Mapping):
            diagnostics.append(
                _error(
                    path=track_path,
                    code="invalid_canonical_field",
                    message="tracks entries must be objects.",
                )
            )
            continue

        track_identity = track.get("id")
        if not isinstance(track_identity, int | str):
            diagnostics.append(
                _error(
                    path=f"{track_path}.id",
                    code="invalid_canonical_field",
                    message="tracks entries must include an integer or string id.",
                )
            )
            continue

        part_identifier = build_part_id(track_identity)
        if part_identifier in seen_part_ids:
            diagnostics.append(
                _error(
                    path=f"{track_path}.id",
                    code="duplicate_part_id",
                    message=f"part id '{part_identifier}' appears more than once.",
                )
            )
            continue

        instrument = _coerce_required_mapping(
            track.get("instrument"),
            path=f"{track_path}.instrument",
            diagnostics=diagnostics,
        )
        transposition = _coerce_transposition(
            track.get("transposition"),
            path=f"{track_path}.transposition",
            diagnostics=diagnostics,
        )
        raw_staves = _require_list_field(
            track,
            field_name="staves",
            path=f"{track_path}.staves",
            diagnostics=diagnostics,
        )
        if instrument is None or transposition is None or raw_staves is None:
            continue

        instrument_family = _coerce_required_int(
            instrument.get("family"),
            path=f"{track_path}.instrument.family",
            diagnostics=diagnostics,
        )
        instrument_kind = _coerce_required_int(
            instrument.get("kind"),
            path=f"{track_path}.instrument.kind",
            diagnostics=diagnostics,
        )
        role = _coerce_required_int(
            instrument.get("role"),
            path=f"{track_path}.instrument.role",
            diagnostics=diagnostics,
        )
        if instrument_family is None or instrument_kind is None or role is None:
            continue

        emitted_staves = _emit_staff_models(
            part_identifier=part_identifier,
            raw_staves=raw_staves,
            track_path=track_path,
            diagnostics=diagnostics,
        )
        if not emitted_staves:
            diagnostics.append(
                _error(
                    path=f"{track_path}.staves",
                    code="invalid_canonical_field",
                    message="tracks must emit at least one valid staff.",
                )
            )
            continue

        try:
            part = Part(
                part_id=part_identifier,
                instrument_family=instrument_family,
                instrument_kind=instrument_kind,
                role=role,
                transposition=transposition,
                staff_ids=tuple(staff.staff_id for staff in emitted_staves),
            )
        except ValueError as exc:
            diagnostics.append(
                _error(
                    path=track_path,
                    code="invalid_canonical_field",
                    message=str(exc),
                )
            )
            continue

        seen_part_ids.add(part_identifier)
        parts.append(part)
        staves.extend(emitted_staves)

    return tuple(parts), tuple(staves), diagnostics


def _emit_staff_models(
    part_identifier: str,
    raw_staves: list[object],
    track_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> list[Staff]:
    emitted_staves: list[Staff] = []
    seen_staff_indexes: set[int] = set()
    sortable_staffs: list[tuple[int, int, Staff]] = []

    for ordinal, raw_staff in enumerate(raw_staves):
        staff_path = f"{track_path}.staves[{ordinal}]"
        if not isinstance(raw_staff, Mapping):
            diagnostics.append(
                _error(
                    path=staff_path,
                    code="invalid_canonical_field",
                    message="staves entries must be objects.",
                )
            )
            continue

        staff_index = _coerce_required_int(
            raw_staff.get("staffIndex"),
            path=f"{staff_path}.staffIndex",
            diagnostics=diagnostics,
        )
        if staff_index is None:
            continue

        if staff_index in seen_staff_indexes:
            diagnostics.append(
                _error(
                    path=f"{staff_path}.staffIndex",
                    code="duplicate_staff_index",
                    message=f"staff index {staff_index} appears more than once.",
                )
            )
            continue

        tuning_pitches = _coerce_optional_tuning_pitches(
            raw_staff.get("tuning"),
            path=f"{staff_path}.tuning",
            diagnostics=diagnostics,
        )
        capo_fret = _coerce_optional_int(
            raw_staff.get("capoFret"),
            path=f"{staff_path}.capoFret",
            diagnostics=diagnostics,
        )

        try:
            staff = Staff(
                staff_id=build_staff_id(part_identifier, staff_index),
                part_id=part_identifier,
                staff_index=staff_index,
                tuning_pitches=tuning_pitches,
                capo_fret=capo_fret,
            )
        except ValueError as exc:
            diagnostics.append(
                _error(
                    path=staff_path,
                    code="invalid_canonical_field",
                    message=str(exc),
                )
            )
            continue

        seen_staff_indexes.add(staff_index)
        sortable_staffs.append((staff_index, ordinal, staff))

    sortable_staffs.sort(key=lambda item: (item[0], item[1]))
    emitted_staves.extend(staff for _, _, staff in sortable_staffs)
    return emitted_staves


def _emit_bar_models(
    score: dict[str, Any],
    written_time_map: WrittenTimeMapResult,
) -> tuple[tuple[Bar, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    timeline_bars = score.get("timelineBars")
    if not isinstance(timeline_bars, list):
        diagnostics.append(
            _error(
                path="timelineBars",
                code="missing_canonical_field",
                message="canonical field 'timelineBars' is required.",
            )
        )
        return (), diagnostics

    written_time_entries = {entry.bar_index: entry for entry in written_time_map.bars}
    seen_bar_indexes: set[int] = set()
    sortable_bars: list[tuple[int, int, Bar]] = []

    for ordinal, timeline_bar in enumerate(timeline_bars):
        bar_path = f"timelineBars[{ordinal}]"
        if not isinstance(timeline_bar, Mapping):
            diagnostics.append(
                _error(
                    path=bar_path,
                    code="invalid_canonical_field",
                    message="timelineBars entries must be objects.",
                )
            )
            continue

        bar_index = _coerce_required_int(
            timeline_bar.get("index"),
            path=f"{bar_path}.index",
            diagnostics=diagnostics,
        )
        if bar_index is None:
            continue

        if bar_index in seen_bar_indexes:
            diagnostics.append(
                _error(
                    path=f"{bar_path}.index",
                    code="duplicate_bar_index",
                    message=f"bar index {bar_index} appears more than once.",
                )
            )
            continue

        written_time_entry = written_time_entries.get(bar_index)
        if written_time_entry is None:
            diagnostics.append(
                _error(
                    path=bar_path,
                    code="missing_written_time_entry",
                    message=f"no written time map entry exists for bar index {bar_index}.",
                )
            )
            continue

        time_signature = _coerce_time_signature(
            timeline_bar.get("timeSignature"),
            path=f"{bar_path}.timeSignature",
            diagnostics=diagnostics,
        )
        if time_signature is None:
            continue

        key_accidental_count = _coerce_optional_int(
            timeline_bar.get("keyAccidentalCount"),
            path=f"{bar_path}.keyAccidentalCount",
            diagnostics=diagnostics,
        )
        key_mode = _coerce_optional_str(
            timeline_bar.get("keyMode"),
            path=f"{bar_path}.keyMode",
            diagnostics=diagnostics,
        )
        triplet_feel = _coerce_optional_str(
            timeline_bar.get("tripletFeel"),
            path=f"{bar_path}.tripletFeel",
            diagnostics=diagnostics,
        )

        try:
            bar = Bar(
                bar_id=build_bar_id(bar_index),
                bar_index=bar_index,
                start=written_time_entry.start,
                duration=written_time_entry.duration,
                time_signature=time_signature,
                key_accidental_count=key_accidental_count,
                key_mode=key_mode,
                triplet_feel=triplet_feel,
            )
        except ValueError as exc:
            diagnostics.append(
                _error(
                    path=bar_path,
                    code="invalid_canonical_field",
                    message=str(exc),
                )
            )
            continue

        seen_bar_indexes.add(bar_index)
        sortable_bars.append((bar_index, ordinal, bar))

    sortable_bars.sort(key=lambda item: (item[0], item[1]))
    return tuple(bar for _, _, bar in sortable_bars), diagnostics


def _emit_voice_lane_models(
    score: dict[str, Any],
    part_staff_emission: PartStaffEmissionResult,
    bar_emission: BarEmissionResult,
) -> tuple[tuple[VoiceLane, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    tracks = score.get("tracks")
    if not isinstance(tracks, list):
        diagnostics.append(
            _error(
                path="tracks",
                code="missing_canonical_field",
                message="canonical field 'tracks' is required.",
            )
        )
        return (), diagnostics

    known_staff_ids = {staff.staff_id for staff in part_staff_emission.staves}
    bar_ids_by_index = {bar.bar_index: bar.bar_id for bar in bar_emission.bars}
    sortable_voice_lanes: list[tuple[int, str, int, VoiceLane]] = []

    for track_index, track in enumerate(tracks):
        sortable_voice_lanes.extend(
            _emit_voice_lanes_for_track(
                track=track,
                track_path=f"tracks[{track_index}]",
                known_staff_ids=known_staff_ids,
                bar_ids_by_index=bar_ids_by_index,
                diagnostics=diagnostics,
            )
        )

    sortable_voice_lanes.sort(
        key=lambda item: (item[0], item[1], item[2], item[3].voice_lane_id)
    )
    return tuple(item[3] for item in sortable_voice_lanes), diagnostics


def _emit_voice_lanes_for_track(
    track: object,
    track_path: str,
    known_staff_ids: set[str],
    bar_ids_by_index: dict[int, str],
    diagnostics: list[IrBuildDiagnostic],
) -> list[tuple[int, str, int, VoiceLane]]:
    if not isinstance(track, Mapping):
        diagnostics.append(
            _error(
                path=track_path,
                code="invalid_canonical_field",
                message="tracks entries must be objects.",
            )
        )
        return []

    track_identity = track.get("id")
    if not isinstance(track_identity, int | str):
        diagnostics.append(
            _error(
                path=f"{track_path}.id",
                code="invalid_canonical_field",
                message="tracks entries must include an integer or string id.",
            )
        )
        return []

    staves = _require_list_field(
        track,
        field_name="staves",
        path=f"{track_path}.staves",
        diagnostics=diagnostics,
    )
    if staves is None:
        return []

    part_identifier = build_part_id(track_identity)
    sortable_voice_lanes: list[tuple[int, str, int, VoiceLane]] = []
    for staff_ordinal, raw_staff in enumerate(staves):
        sortable_voice_lanes.extend(
            _emit_voice_lanes_for_staff(
                context=_VoiceLaneStaffBuildContext(
                    part_identifier=part_identifier,
                    staff_path=f"{track_path}.staves[{staff_ordinal}]",
                    known_staff_ids=frozenset(known_staff_ids),
                    bar_ids_by_index=bar_ids_by_index,
                ),
                raw_staff=raw_staff,
                diagnostics=diagnostics,
            )
        )

    return sortable_voice_lanes


def _emit_voice_lanes_for_staff(
    context: _VoiceLaneStaffBuildContext,
    raw_staff: object,
    diagnostics: list[IrBuildDiagnostic],
) -> list[tuple[int, str, int, VoiceLane]]:
    if not isinstance(raw_staff, Mapping):
        diagnostics.append(
            _error(
                path=context.staff_path,
                code="invalid_canonical_field",
                message="staves entries must be objects.",
            )
        )
        return []

    staff_index = _coerce_required_int(
        raw_staff.get("staffIndex"),
        path=f"{context.staff_path}.staffIndex",
        diagnostics=diagnostics,
    )
    if staff_index is None:
        return []

    measures = _require_list_field(
        raw_staff,
        field_name="measures",
        path=f"{context.staff_path}.measures",
        diagnostics=diagnostics,
    )
    if measures is None:
        return []

    staff_identifier = build_staff_id(context.part_identifier, staff_index)
    if staff_identifier not in context.known_staff_ids:
        diagnostics.append(
            _error(
                path=context.staff_path,
                code="missing_staff_reference",
                message=f"no emitted staff exists for staff id '{staff_identifier}'.",
            )
        )
        return []

    sortable_voice_lanes: list[tuple[int, str, int, VoiceLane]] = []

    for measure_ordinal, measure in enumerate(measures):
        measure_path = f"{context.staff_path}.measures[{measure_ordinal}]"
        if not isinstance(measure, Mapping):
            diagnostics.append(
                _error(
                    path=measure_path,
                    code="invalid_canonical_field",
                    message="measures entries must be objects.",
                )
            )
            continue

        bar_index = _coerce_required_int(
            measure.get("index"),
            path=f"{measure_path}.index",
            diagnostics=diagnostics,
        )
        if bar_index is None:
            continue

        bar_id = context.bar_ids_by_index.get(bar_index)
        if bar_id is None:
            diagnostics.append(
                _error(
                    path=measure_path,
                    code="missing_bar_reference",
                    message=f"no emitted bar exists for bar index {bar_index}.",
                )
            )
            continue

        voices = _require_list_field(
            measure,
            field_name="voices",
            path=f"{measure_path}.voices",
            diagnostics=diagnostics,
        )
        if voices is None:
            continue

        seen_voice_indexes: set[int] = set()
        for voice_ordinal, voice in enumerate(voices):
            emitted = _emit_voice_lane_for_voice(
                context=_VoiceLaneBuildContext(
                    part_identifier=context.part_identifier,
                    staff_identifier=staff_identifier,
                    bar_index=bar_index,
                    bar_id=bar_id,
                    voice_path=f"{measure_path}.voices[{voice_ordinal}]",
                ),
                voice=voice,
                seen_voice_indexes=seen_voice_indexes,
                diagnostics=diagnostics,
            )
            if emitted is not None:
                sortable_voice_lanes.append(emitted)

    return sortable_voice_lanes


def _emit_voice_lane_for_voice(
    context: _VoiceLaneBuildContext,
    voice: object,
    seen_voice_indexes: set[int],
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[int, str, int, VoiceLane] | None:
    emitted_voice_lane: tuple[int, str, int, VoiceLane] | None = None
    if not isinstance(voice, Mapping):
        diagnostics.append(
            _error(
                path=context.voice_path,
                code="invalid_canonical_field",
                message="voices entries must be objects.",
            )
        )
    else:
        voice_index = _coerce_required_int(
            voice.get("voiceIndex"),
            path=f"{context.voice_path}.voiceIndex",
            diagnostics=diagnostics,
        )
        if voice_index is not None:
            if voice_index in seen_voice_indexes:
                diagnostics.append(
                    _error(
                        path=f"{context.voice_path}.voiceIndex",
                        code="duplicate_voice_index",
                        message=(
                            f"voice index {voice_index} appears more than once "
                            "within the same measure."
                        ),
                    )
                )
            else:
                beats = _require_list_field(
                    voice,
                    field_name="beats",
                    path=f"{context.voice_path}.beats",
                    diagnostics=diagnostics,
                )
                if beats is not None:
                    seen_voice_indexes.add(voice_index)
                    if _voice_contains_authored_content(beats):
                        try:
                            voice_lane = VoiceLane(
                                voice_lane_id=build_voice_lane_id(
                                    context.staff_identifier,
                                    context.bar_index,
                                    voice_index,
                                ),
                                voice_lane_chain_id=build_voice_lane_chain_id(
                                    context.part_identifier,
                                    context.staff_identifier,
                                    voice_index,
                                ),
                                part_id=context.part_identifier,
                                staff_id=context.staff_identifier,
                                bar_id=context.bar_id,
                                voice_index=voice_index,
                            )
                        except ValueError as exc:
                            diagnostics.append(
                                _error(
                                    path=context.voice_path,
                                    code="invalid_canonical_field",
                                    message=str(exc),
                                )
                            )
                        else:
                            emitted_voice_lane = (
                                context.bar_index,
                                context.staff_identifier,
                                voice_index,
                                voice_lane,
                            )

    return emitted_voice_lane


def _emit_onset_group_models(
    score: dict[str, Any],
    written_time_map: WrittenTimeMapResult,
    voice_lane_emission: VoiceLaneEmissionResult,
) -> tuple[tuple[OnsetGroup, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    tracks = score.get("tracks")
    if not isinstance(tracks, list):
        diagnostics.append(
            _error(
                path="tracks",
                code="missing_canonical_field",
                message="canonical field 'tracks' is required.",
            )
        )
        return (), diagnostics

    bar_times = written_time_map.bar_times
    known_voice_lane_ids = {
        voice_lane.voice_lane_id for voice_lane in voice_lane_emission.voice_lanes
    }
    onset_groups: list[OnsetGroup] = []

    for track_index, track in enumerate(tracks):
        onset_groups.extend(
            _emit_onset_groups_for_track(
                track=track,
                track_path=f"tracks[{track_index}]",
                bar_times=bar_times,
                known_voice_lane_ids=known_voice_lane_ids,
                diagnostics=diagnostics,
            )
        )

    return tuple(onset_groups), diagnostics


def _emit_onset_groups_for_track(
    track: object,
    track_path: str,
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]],
    known_voice_lane_ids: set[str],
    diagnostics: list[IrBuildDiagnostic],
) -> list[OnsetGroup]:
    if not isinstance(track, Mapping):
        diagnostics.append(
            _error(
                path=track_path,
                code="invalid_canonical_field",
                message="tracks entries must be objects.",
            )
        )
        return []

    track_identity = track.get("id")
    if not isinstance(track_identity, int | str):
        diagnostics.append(
            _error(
                path=f"{track_path}.id",
                code="invalid_canonical_field",
                message="tracks entries must include an integer or string id.",
            )
        )
        return []

    staves = _require_list_field(
        track,
        field_name="staves",
        path=f"{track_path}.staves",
        diagnostics=diagnostics,
    )
    if staves is None:
        return []

    part_identifier = build_part_id(track_identity)
    onset_groups: list[OnsetGroup] = []
    for staff_ordinal, raw_staff in enumerate(staves):
        onset_groups.extend(
            _emit_onset_groups_for_staff(
                context=_OnsetStaffBuildContext(
                    part_identifier=part_identifier,
                    staff_path=f"{track_path}.staves[{staff_ordinal}]",
                    bar_times=bar_times,
                    known_voice_lane_ids=frozenset(known_voice_lane_ids),
                ),
                raw_staff=raw_staff,
                diagnostics=diagnostics,
            )
        )

    return onset_groups


def _emit_onset_groups_for_staff(
    context: _OnsetStaffBuildContext,
    raw_staff: object,
    diagnostics: list[IrBuildDiagnostic],
) -> list[OnsetGroup]:
    if not isinstance(raw_staff, Mapping):
        diagnostics.append(
            _error(
                path=context.staff_path,
                code="invalid_canonical_field",
                message="staves entries must be objects.",
            )
        )
        return []

    staff_index = _coerce_required_int(
        raw_staff.get("staffIndex"),
        path=f"{context.staff_path}.staffIndex",
        diagnostics=diagnostics,
    )
    if staff_index is None:
        return []

    measures = _require_list_field(
        raw_staff,
        field_name="measures",
        path=f"{context.staff_path}.measures",
        diagnostics=diagnostics,
    )
    if measures is None:
        return []

    staff_identifier = build_staff_id(context.part_identifier, staff_index)
    onset_groups: list[OnsetGroup] = []

    for measure_ordinal, measure in enumerate(measures):
        measure_path = f"{context.staff_path}.measures[{measure_ordinal}]"
        if not isinstance(measure, Mapping):
            diagnostics.append(
                _error(
                    path=measure_path,
                    code="invalid_canonical_field",
                    message="measures entries must be objects.",
                )
            )
            continue

        bar_index = _coerce_required_int(
            measure.get("index"),
            path=f"{measure_path}.index",
            diagnostics=diagnostics,
        )
        if bar_index is None:
            continue

        bar_time = context.bar_times.get(bar_index)
        if bar_time is None:
            diagnostics.append(
                _error(
                    path=measure_path,
                    code="missing_bar_reference",
                    message=f"no written-time map exists for bar index {bar_index}.",
                )
            )
            continue

        voices = _require_list_field(
            measure,
            field_name="voices",
            path=f"{measure_path}.voices",
            diagnostics=diagnostics,
        )
        if voices is None:
            continue

        seen_voice_indexes: set[int] = set()
        bar_start, bar_duration = bar_time
        bar_identifier = build_bar_id(bar_index)

        for voice_ordinal, voice in enumerate(voices):
            onset_groups.extend(
                _emit_onset_groups_for_voice(
                    context=_OnsetVoiceBuildContext(
                        staff_identifier=staff_identifier,
                        bar_index=bar_index,
                        bar_id=bar_identifier,
                        bar_start=bar_start,
                        bar_duration=bar_duration,
                        voice_path=f"{measure_path}.voices[{voice_ordinal}]",
                        known_voice_lane_ids=context.known_voice_lane_ids,
                    ),
                    voice=voice,
                    seen_voice_indexes=seen_voice_indexes,
                    diagnostics=diagnostics,
                )
            )

    return onset_groups


def _emit_onset_groups_for_voice(
    context: _OnsetVoiceBuildContext,
    voice: object,
    seen_voice_indexes: set[int],
    diagnostics: list[IrBuildDiagnostic],
) -> list[OnsetGroup]:
    onset_groups: list[OnsetGroup] = []
    if not isinstance(voice, Mapping):
        diagnostics.append(
            _error(
                path=context.voice_path,
                code="invalid_canonical_field",
                message="voices entries must be objects.",
            )
        )
    else:
        voice_index = _coerce_required_int(
            voice.get("voiceIndex"),
            path=f"{context.voice_path}.voiceIndex",
            diagnostics=diagnostics,
        )
        if voice_index is not None:
            if voice_index in seen_voice_indexes:
                diagnostics.append(
                    _error(
                        path=f"{context.voice_path}.voiceIndex",
                        code="duplicate_voice_index",
                        message=(
                            f"voice index {voice_index} appears more than once "
                            "within the same measure."
                        ),
                    )
                )
            else:
                beats = _require_list_field(
                    voice,
                    field_name="beats",
                    path=f"{context.voice_path}.beats",
                    diagnostics=diagnostics,
                )
                if beats is not None:
                    seen_voice_indexes.add(voice_index)
                    if beats:
                        voice_lane_identifier = build_voice_lane_id(
                            context.staff_identifier,
                            context.bar_index,
                            voice_index,
                        )
                        if voice_lane_identifier not in context.known_voice_lane_ids:
                            diagnostics.append(
                                _error(
                                    path=f"{context.voice_path}.voiceIndex",
                                    code="missing_voice_lane_reference",
                                    message=(
                                        "no emitted voice lane exists for "
                                        f"'{voice_lane_identifier}'."
                                    ),
                                )
                            )
                        else:
                            attack_order_in_voice = 0
                            for beat_ordinal, beat in enumerate(beats):
                                onset_group = _coerce_onset_group(
                                    raw_beat=beat,
                                    context=_OnsetBeatBuildContext(
                                        voice_lane_id=voice_lane_identifier,
                                        bar_index=context.bar_index,
                                        bar_id=context.bar_id,
                                        bar_start=context.bar_start,
                                        bar_duration=context.bar_duration,
                                        beat_path=(
                                            f"{context.voice_path}.beats[{beat_ordinal}]"
                                        ),
                                        attack_order_in_voice=attack_order_in_voice,
                                    ),
                                    diagnostics=diagnostics,
                                )
                                if onset_group is not None:
                                    onset_groups.append(onset_group)
                                    attack_order_in_voice += 1

    return onset_groups


def _coerce_onset_group(
    raw_beat: object,
    context: _OnsetBeatBuildContext,
    diagnostics: list[IrBuildDiagnostic],
) -> OnsetGroup | None:
    beat_core = _coerce_beat_core_fields(
        raw_beat=raw_beat,
        beat_path=context.beat_path,
        bar_index=context.bar_index,
        bar_duration=context.bar_duration,
        diagnostics=diagnostics,
    )
    if beat_core is None or not isinstance(raw_beat, Mapping):
        return None

    offset, duration, notes = beat_core
    grace_type = _coerce_optional_str(
        raw_beat.get("graceType"),
        path=f"{context.beat_path}.graceType",
        diagnostics=diagnostics,
    )
    rhythm_shape = _coerce_optional_rhythm_shape(
        raw_beat.get("rhythm"),
        path=f"{context.beat_path}.rhythm",
        diagnostics=diagnostics,
    )
    techniques = _coerce_optional_onset_techniques(
        raw_beat,
        beat_path=context.beat_path,
        diagnostics=diagnostics,
    )

    is_rest = len(notes) == 0
    duration_sounding_max = _coerce_duration_sounding_max(
        notes=notes,
        beat_path=context.beat_path,
        diagnostics=diagnostics,
    )
    time = context.bar_start + offset

    try:
        return OnsetGroup(
            onset_id=build_onset_id(
                context.voice_lane_id, context.attack_order_in_voice
            ),
            voice_lane_id=context.voice_lane_id,
            bar_id=context.bar_id,
            time=time,
            duration_notated=duration,
            is_rest=is_rest,
            attack_order_in_voice=context.attack_order_in_voice,
            duration_sounding_max=duration_sounding_max,
            grace_type=grace_type,
            dynamic_local=None,
            techniques=techniques,
            rhythm_shape=rhythm_shape,
        )
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=context.beat_path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_beat_core_fields(
    raw_beat: object,
    beat_path: str,
    bar_index: int,
    bar_duration: ScoreTime,
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[ScoreTime, ScoreTime, list[object]] | None:
    if not isinstance(raw_beat, Mapping):
        diagnostics.append(
            _error(
                path=beat_path,
                code="invalid_canonical_field",
                message="beats entries must be objects.",
            )
        )
        return None

    offset = _coerce_score_time(
        raw_beat.get("offset"),
        path=f"{beat_path}.offset",
        diagnostics=diagnostics,
        require_positive=False,
    )
    duration = _coerce_score_time(
        raw_beat.get("duration"),
        path=f"{beat_path}.duration",
        diagnostics=diagnostics,
        require_positive=True,
    )
    notes = _require_list_field(
        raw_beat,
        field_name="notes",
        path=f"{beat_path}.notes",
        diagnostics=diagnostics,
    )
    if offset is None or duration is None or notes is None:
        return None

    if offset >= bar_duration:
        diagnostics.append(
            _error(
                path=f"{beat_path}.offset",
                code="invalid_canonical_field",
                message=f"beat offset exceeds the duration of bar {bar_index}.",
            )
        )
        return None

    if offset + duration > bar_duration:
        diagnostics.append(
            _error(
                path=f"{beat_path}.duration",
                code="invalid_canonical_field",
                message=(
                    f"beat duration exceeds the remaining space in bar {bar_index}."
                ),
            )
        )
        return None

    return (offset, duration, notes)


def _coerce_duration_sounding_max(
    notes: list[object],
    beat_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> ScoreTime | None:
    sounding_durations: list[ScoreTime] = []

    for note_index, note in enumerate(notes):
        note_path = f"{beat_path}.notes[{note_index}]"
        if not isinstance(note, Mapping):
            diagnostics.append(
                _error(
                    path=note_path,
                    code="invalid_canonical_field",
                    message="notes entries must be objects.",
                )
            )
            continue

        sounding_duration = _coerce_score_time(
            note.get("soundingDuration"),
            path=f"{note_path}.soundingDuration",
            diagnostics=diagnostics,
            require_positive=True,
        )
        if sounding_duration is not None:
            sounding_durations.append(sounding_duration)

    if not sounding_durations:
        return None

    return max(sounding_durations)


def _coerce_optional_rhythm_shape(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> RhythmShape | None:
    if value is None:
        return None

    rhythm = _coerce_required_mapping(value, path, diagnostics)
    if rhythm is None:
        return None

    base_value = _coerce_optional_str(
        rhythm.get("baseValue"),
        path=f"{path}.baseValue",
        diagnostics=diagnostics,
    )
    augmentation_dots = _coerce_required_int(
        rhythm.get("augmentationDots"),
        path=f"{path}.augmentationDots",
        diagnostics=diagnostics,
    )
    primary_tuplet = _coerce_optional_tuplet_ratio(
        rhythm.get("primaryTuplet"),
        path=f"{path}.primaryTuplet",
        diagnostics=diagnostics,
    )
    secondary_tuplet = _coerce_optional_tuplet_ratio(
        rhythm.get("secondaryTuplet"),
        path=f"{path}.secondaryTuplet",
        diagnostics=diagnostics,
    )
    if base_value is None or augmentation_dots is None:
        return None

    try:
        return RhythmShape(
            base_value=RhythmBaseValue(base_value),
            augmentation_dots=augmentation_dots,
            primary_tuplet=primary_tuplet,
            secondary_tuplet=secondary_tuplet,
        )
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_optional_tuplet_ratio(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> TupletRatio | None:
    if value is None:
        return None

    tuplet_ratio = _coerce_required_mapping(value, path, diagnostics)
    if tuplet_ratio is None:
        return None

    numerator = _coerce_required_int(
        tuplet_ratio.get("numerator"),
        path=f"{path}.numerator",
        diagnostics=diagnostics,
    )
    denominator = _coerce_required_int(
        tuplet_ratio.get("denominator"),
        path=f"{path}.denominator",
        diagnostics=diagnostics,
    )
    if numerator is None or denominator is None:
        return None

    try:
        return TupletRatio(numerator=numerator, denominator=denominator)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_optional_onset_techniques(
    raw_beat: Mapping[str, Any],
    beat_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> TechniquePayload | None:
    palm_muted = _coerce_optional_bool_field(
        raw_beat.get("palmMuted"),
        path=f"{beat_path}.palmMuted",
        diagnostics=diagnostics,
    )

    whammy_enabled = False
    whammy_bar = raw_beat.get("whammyBar")
    if whammy_bar is not None:
        whammy_bar_mapping = _coerce_required_mapping(
            whammy_bar,
            path=f"{beat_path}.whammyBar",
            diagnostics=diagnostics,
        )
        if whammy_bar_mapping is not None:
            enabled = _coerce_optional_bool_field(
                whammy_bar_mapping.get("enabled"),
                path=f"{beat_path}.whammyBar.enabled",
                diagnostics=diagnostics,
            )
            whammy_enabled = bool(enabled)

    _warn_for_unsupported_onset_techniques(
        raw_beat=raw_beat,
        beat_path=beat_path,
        diagnostics=diagnostics,
    )

    generic = GenericTechniqueFlags(
        palm_muted=bool(palm_muted),
    )
    string_fretted = (
        StringFrettedTechniquePayload(whammy_enabled=True) if whammy_enabled else None
    )

    if generic == GenericTechniqueFlags() and string_fretted is None:
        return None

    return TechniquePayload(generic=generic, string_fretted=string_fretted)


def _warn_for_unsupported_onset_techniques(
    raw_beat: Mapping[str, Any],
    beat_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for field_name in UNSUPPORTED_ONSET_TECHNIQUE_FIELDS:
        enabled = _coerce_optional_bool_field(
            raw_beat.get(field_name),
            path=f"{beat_path}.{field_name}",
            diagnostics=diagnostics,
        )
        if enabled:
            diagnostics.append(
                _warning(
                    path=f"{beat_path}.{field_name}",
                    code="unsupported_onset_technique",
                    message=(
                        f"beat-local technique '{field_name}' is not yet represented "
                        "in the IR onset technique payload."
                    ),
                )
            )


def _emit_note_event_models(
    score: dict[str, Any],
    written_time_map: WrittenTimeMapResult,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
) -> tuple[tuple[NoteEvent, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    tracks = score.get("tracks")
    if not isinstance(tracks, list):
        diagnostics.append(
            _error(
                path="tracks",
                code="missing_canonical_field",
                message="canonical field 'tracks' is required.",
            )
        )
        return (), diagnostics

    voice_lanes_by_id = {
        voice_lane.voice_lane_id: voice_lane
        for voice_lane in voice_lane_emission.voice_lanes
    }
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]] = {}
    for onset_group in onset_group_emission.onset_groups:
        onset_groups_by_voice_lane.setdefault(onset_group.voice_lane_id, {})[
            onset_group.attack_order_in_voice
        ] = onset_group

    note_events: list[NoteEvent] = []
    for track_index, track in enumerate(tracks):
        note_events.extend(
            _emit_note_events_for_track(
                context=_NoteTrackBuildContext(
                    track_path=f"tracks[{track_index}]",
                    bar_times=written_time_map.bar_times,
                    voice_lanes_by_id=voice_lanes_by_id,
                    onset_groups_by_voice_lane=onset_groups_by_voice_lane,
                ),
                track=track,
                diagnostics=diagnostics,
            )
        )

    return tuple(note_events), diagnostics


def _emit_note_events_for_track(
    context: _NoteTrackBuildContext,
    track: object,
    diagnostics: list[IrBuildDiagnostic],
) -> list[NoteEvent]:
    if not isinstance(track, Mapping):
        diagnostics.append(
            _error(
                path=context.track_path,
                code="invalid_canonical_field",
                message="tracks entries must be objects.",
            )
        )
        return []

    track_identity = track.get("id")
    if not isinstance(track_identity, int | str):
        diagnostics.append(
            _error(
                path=f"{context.track_path}.id",
                code="invalid_canonical_field",
                message="tracks entries must include an integer or string id.",
            )
        )
        return []

    staves = _require_list_field(
        track,
        field_name="staves",
        path=f"{context.track_path}.staves",
        diagnostics=diagnostics,
    )
    if staves is None:
        return []

    part_identifier = build_part_id(track_identity)
    note_events: list[NoteEvent] = []
    for staff_ordinal, raw_staff in enumerate(staves):
        note_events.extend(
            _emit_note_events_for_staff(
                context=_NoteStaffBuildContext(
                    part_identifier=part_identifier,
                    staff_path=f"{context.track_path}.staves[{staff_ordinal}]",
                    bar_times=context.bar_times,
                    voice_lanes_by_id=context.voice_lanes_by_id,
                    onset_groups_by_voice_lane=context.onset_groups_by_voice_lane,
                ),
                raw_staff=raw_staff,
                diagnostics=diagnostics,
            )
        )

    return note_events


def _emit_note_events_for_staff(
    context: _NoteStaffBuildContext,
    raw_staff: object,
    diagnostics: list[IrBuildDiagnostic],
) -> list[NoteEvent]:
    if not isinstance(raw_staff, Mapping):
        diagnostics.append(
            _error(
                path=context.staff_path,
                code="invalid_canonical_field",
                message="staves entries must be objects.",
            )
        )
        return []

    staff_index = _coerce_required_int(
        raw_staff.get("staffIndex"),
        path=f"{context.staff_path}.staffIndex",
        diagnostics=diagnostics,
    )
    if staff_index is None:
        return []

    measures = _require_list_field(
        raw_staff,
        field_name="measures",
        path=f"{context.staff_path}.measures",
        diagnostics=diagnostics,
    )
    if measures is None:
        return []

    staff_identifier = build_staff_id(context.part_identifier, staff_index)
    note_events: list[NoteEvent] = []

    for measure_ordinal, measure in enumerate(measures):
        measure_path = f"{context.staff_path}.measures[{measure_ordinal}]"
        if not isinstance(measure, Mapping):
            diagnostics.append(
                _error(
                    path=measure_path,
                    code="invalid_canonical_field",
                    message="measures entries must be objects.",
                )
            )
            continue

        bar_index = _coerce_required_int(
            measure.get("index"),
            path=f"{measure_path}.index",
            diagnostics=diagnostics,
        )
        if bar_index is None:
            continue

        bar_time = context.bar_times.get(bar_index)
        if bar_time is None:
            diagnostics.append(
                _error(
                    path=measure_path,
                    code="missing_bar_reference",
                    message=f"no written-time map exists for bar index {bar_index}.",
                )
            )
            continue

        voices = _require_list_field(
            measure,
            field_name="voices",
            path=f"{measure_path}.voices",
            diagnostics=diagnostics,
        )
        if voices is None:
            continue

        seen_voice_indexes: set[int] = set()
        _, bar_duration = bar_time
        for voice_ordinal, voice in enumerate(voices):
            note_events.extend(
                _emit_note_events_for_voice(
                    context=_NoteVoiceTraversalContext(
                        staff_identifier=staff_identifier,
                        bar_index=bar_index,
                        bar_duration=bar_duration,
                        voice_path=f"{measure_path}.voices[{voice_ordinal}]",
                        voice_lanes_by_id=context.voice_lanes_by_id,
                        onset_groups_by_voice_lane=context.onset_groups_by_voice_lane,
                    ),
                    voice=voice,
                    seen_voice_indexes=seen_voice_indexes,
                    diagnostics=diagnostics,
                )
            )

    return note_events


def _emit_note_events_for_voice(
    context: _NoteVoiceTraversalContext,
    voice: object,
    seen_voice_indexes: set[int],
    diagnostics: list[IrBuildDiagnostic],
) -> list[NoteEvent]:
    note_events: list[NoteEvent] = []
    resolved_inputs = _resolve_note_voice_inputs(
        context=context,
        voice=voice,
        seen_voice_indexes=seen_voice_indexes,
        diagnostics=diagnostics,
    )
    if resolved_inputs is None:
        return note_events

    voice_lane, onset_groups_by_attack_order, beats = resolved_inputs
    attack_order_in_voice = 0
    for beat_ordinal, beat in enumerate(beats):
        beat_path = f"{context.voice_path}.beats[{beat_ordinal}]"
        beat_core = _coerce_beat_core_fields(
            raw_beat=beat,
            beat_path=beat_path,
            bar_index=context.bar_index,
            bar_duration=context.bar_duration,
            diagnostics=diagnostics,
        )
        if beat_core is None:
            continue
        _, _, notes = beat_core

        onset_group = onset_groups_by_attack_order.get(attack_order_in_voice)
        if onset_group is None:
            diagnostics.append(
                _error(
                    path=beat_path,
                    code="missing_onset_group_reference",
                    message=(
                        "no emitted onset group exists for "
                        f"voice lane '{voice_lane.voice_lane_id}' at attack order "
                        f"{attack_order_in_voice}."
                    ),
                )
            )
            attack_order_in_voice += 1
            continue

        note_events.extend(
            _emit_note_events_for_beat(
                context=_NoteBeatBuildContext(
                    voice_lane=voice_lane,
                ),
                onset_group=onset_group,
                notes=notes,
                beat_path=beat_path,
                diagnostics=diagnostics,
            )
        )
        attack_order_in_voice += 1

    return note_events


def _resolve_note_voice_inputs(
    context: _NoteVoiceTraversalContext,
    voice: object,
    seen_voice_indexes: set[int],
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[VoiceLane, dict[int, OnsetGroup], list[object]] | None:
    resolved_inputs: tuple[VoiceLane, dict[int, OnsetGroup], list[object]] | None = None
    if not isinstance(voice, Mapping):
        diagnostics.append(
            _error(
                path=context.voice_path,
                code="invalid_canonical_field",
                message="voices entries must be objects.",
            )
        )
    else:
        voice_index = _coerce_required_int(
            voice.get("voiceIndex"),
            path=f"{context.voice_path}.voiceIndex",
            diagnostics=diagnostics,
        )
        if voice_index is not None:
            if voice_index in seen_voice_indexes:
                diagnostics.append(
                    _error(
                        path=f"{context.voice_path}.voiceIndex",
                        code="duplicate_voice_index",
                        message=(
                            f"voice index {voice_index} appears more than once within "
                            "the same measure."
                        ),
                    )
                )
            else:
                beats = _require_list_field(
                    voice,
                    field_name="beats",
                    path=f"{context.voice_path}.beats",
                    diagnostics=diagnostics,
                )
                if beats is not None:
                    seen_voice_indexes.add(voice_index)
                    if beats:
                        voice_lane_identifier = build_voice_lane_id(
                            context.staff_identifier,
                            context.bar_index,
                            voice_index,
                        )
                        voice_lane = context.voice_lanes_by_id.get(
                            voice_lane_identifier
                        )
                        if voice_lane is None:
                            diagnostics.append(
                                _error(
                                    path=f"{context.voice_path}.voiceIndex",
                                    code="missing_voice_lane_reference",
                                    message=(
                                        "no emitted voice lane exists for "
                                        f"'{voice_lane_identifier}'."
                                    ),
                                )
                            )
                        else:
                            onset_groups_by_attack_order = (
                                context.onset_groups_by_voice_lane.get(
                                    voice_lane_identifier
                                )
                            )
                            if onset_groups_by_attack_order is None:
                                diagnostics.append(
                                    _error(
                                        path=context.voice_path,
                                        code="missing_onset_group_reference",
                                        message=(
                                            "no emitted onset groups exist for "
                                            f"voice lane '{voice_lane_identifier}'."
                                        ),
                                    )
                                )
                            else:
                                resolved_inputs = (
                                    voice_lane,
                                    onset_groups_by_attack_order,
                                    beats,
                                )

    return resolved_inputs


def _emit_note_events_for_beat(
    context: _NoteBeatBuildContext,
    onset_group: OnsetGroup,
    notes: list[object],
    beat_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> list[NoteEvent]:
    seeds: list[_NoteSeed] = []
    for note_index, raw_note in enumerate(notes):
        seed = _coerce_note_seed(
            raw_note=raw_note,
            note_path=f"{beat_path}.notes[{note_index}]",
            onset_group=onset_group,
            voice_lane=context.voice_lane,
            diagnostics=diagnostics,
        )
        if seed is not None:
            seeds.append(seed)

    seeds.sort(key=lambda item: item.sort_key())
    note_events: list[NoteEvent] = []
    for note_index, seed in enumerate(seeds):
        try:
            note_events.append(
                NoteEvent(
                    note_id=build_note_id(seed.onset_id, note_index),
                    onset_id=seed.onset_id,
                    part_id=seed.part_id,
                    staff_id=seed.staff_id,
                    time=seed.time,
                    attack_duration=seed.attack_duration,
                    sounding_duration=seed.sounding_duration,
                    pitch=seed.pitch,
                    velocity=seed.velocity,
                    string_number=seed.string_number,
                    show_string_number=seed.show_string_number,
                    techniques=seed.techniques,
                )
            )
        except ValueError as exc:
            diagnostics.append(
                _error(
                    path=seed.path,
                    code="invalid_canonical_field",
                    message=str(exc),
                )
            )

    return note_events


def _coerce_note_seed(
    raw_note: object,
    note_path: str,
    onset_group: OnsetGroup,
    voice_lane: VoiceLane,
    diagnostics: list[IrBuildDiagnostic],
) -> _NoteSeed | None:
    if not isinstance(raw_note, Mapping):
        diagnostics.append(
            _error(
                path=note_path,
                code="invalid_canonical_field",
                message="notes entries must be objects.",
            )
        )
        return None

    attack_duration = _coerce_score_time(
        raw_note.get("duration"),
        path=f"{note_path}.duration",
        diagnostics=diagnostics,
        require_positive=True,
    )
    sounding_duration = _coerce_score_time(
        raw_note.get("soundingDuration"),
        path=f"{note_path}.soundingDuration",
        diagnostics=diagnostics,
        require_positive=True,
    )
    pitch = _coerce_optional_pitch(
        raw_note.get("pitch"),
        path=f"{note_path}.pitch",
        diagnostics=diagnostics,
    )
    velocity = _coerce_optional_int(
        raw_note.get("velocity"),
        path=f"{note_path}.velocity",
        diagnostics=diagnostics,
    )
    string_number = _coerce_optional_int(
        raw_note.get("stringNumber"),
        path=f"{note_path}.stringNumber",
        diagnostics=diagnostics,
    )
    show_string_number = _coerce_optional_bool_field(
        raw_note.get("showStringNumber"),
        path=f"{note_path}.showStringNumber",
        diagnostics=diagnostics,
    )
    techniques = _coerce_optional_note_techniques(
        raw_note.get("articulation"),
        path=f"{note_path}.articulation",
        diagnostics=diagnostics,
    )
    if attack_duration is None or sounding_duration is None:
        return None

    return _NoteSeed(
        path=note_path,
        onset_id=onset_group.onset_id,
        part_id=voice_lane.part_id,
        staff_id=voice_lane.staff_id,
        time=onset_group.time,
        attack_duration=attack_duration,
        sounding_duration=sounding_duration,
        pitch=pitch,
        velocity=velocity,
        string_number=string_number,
        show_string_number=show_string_number,
        techniques=techniques,
    )


def _coerce_optional_pitch(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> Pitch | None:
    if value is None:
        return None

    pitch = _coerce_required_mapping(value, path, diagnostics)
    if pitch is None:
        return None

    step = _coerce_optional_str(
        pitch.get("step"),
        path=f"{path}.step",
        diagnostics=diagnostics,
    )
    octave = _coerce_required_int(
        pitch.get("octave"),
        path=f"{path}.octave",
        diagnostics=diagnostics,
    )
    accidental = _coerce_optional_str(
        pitch.get("accidental"),
        path=f"{path}.accidental",
        diagnostics=diagnostics,
    )
    if step is None or octave is None:
        return None

    try:
        return Pitch(step=step, octave=octave, accidental=accidental)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_optional_note_techniques(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> TechniquePayload | None:
    if value is None:
        return None

    articulation = _coerce_required_mapping(value, path, diagnostics)
    if articulation is None:
        return None

    accent = _coerce_optional_int(
        articulation.get("accent"),
        path=f"{path}.accent",
        diagnostics=diagnostics,
    )
    ornament = _coerce_optional_str(
        articulation.get("ornament"),
        path=f"{path}.ornament",
        diagnostics=diagnostics,
    )
    vibrato = _coerce_optional_str(
        articulation.get("vibrato"),
        path=f"{path}.vibrato",
        diagnostics=diagnostics,
    )
    trill = _coerce_optional_int(
        articulation.get("trill"),
        path=f"{path}.trill",
        diagnostics=diagnostics,
    )
    generic = GenericTechniqueFlags(
        tie_origin=bool(
            _coerce_optional_bool_field(
                articulation.get("tieOrigin"),
                path=f"{path}.tieOrigin",
                diagnostics=diagnostics,
            )
        ),
        tie_destination=bool(
            _coerce_optional_bool_field(
                articulation.get("tieDestination"),
                path=f"{path}.tieDestination",
                diagnostics=diagnostics,
            )
        ),
        legato_origin=bool(
            _coerce_optional_bool_field(
                articulation.get("hopoOrigin"),
                path=f"{path}.hopoOrigin",
                diagnostics=diagnostics,
            )
        ),
        legato_destination=bool(
            _coerce_optional_bool_field(
                articulation.get("hopoDestination"),
                path=f"{path}.hopoDestination",
                diagnostics=diagnostics,
            )
        ),
        accent=accent,
        ornament=ornament,
        vibrato=vibrato,
        let_ring=bool(
            _coerce_optional_bool_field(
                articulation.get("letRing"),
                path=f"{path}.letRing",
                diagnostics=diagnostics,
            )
        ),
        muted=bool(
            _coerce_optional_bool_field(
                articulation.get("muted"),
                path=f"{path}.muted",
                diagnostics=diagnostics,
            )
        ),
        palm_muted=bool(
            _coerce_optional_bool_field(
                articulation.get("palmMuted"),
                path=f"{path}.palmMuted",
                diagnostics=diagnostics,
            )
        ),
        trill=trill,
    )

    slide_types = _coerce_optional_int_array(
        articulation.get("slides"),
        path=f"{path}.slides",
        diagnostics=diagnostics,
    )
    hopo_type = _coerce_optional_int(
        articulation.get("hopoType"),
        path=f"{path}.hopoType",
        diagnostics=diagnostics,
    )
    tapped = bool(
        _coerce_optional_bool_field(
            articulation.get("tapped"),
            path=f"{path}.tapped",
            diagnostics=diagnostics,
        )
    )
    left_hand_tapped = bool(
        _coerce_optional_bool_field(
            articulation.get("leftHandTapped"),
            path=f"{path}.leftHandTapped",
            diagnostics=diagnostics,
        )
    )
    harmonic = _coerce_optional_harmonic_mapping(
        articulation.get("harmonic"),
        path=f"{path}.harmonic",
        diagnostics=diagnostics,
    )
    bend_enabled = _coerce_optional_enabled_mapping(
        articulation.get("bend"),
        path=f"{path}.bend",
        diagnostics=diagnostics,
    )

    string_fretted = None
    if (
        slide_types is not None
        or hopo_type is not None
        or tapped
        or left_hand_tapped
        or harmonic is not None
        or bend_enabled
    ):
        harmonic_type, harmonic_kind, harmonic_fret = (None, None, None)
        if harmonic is not None:
            harmonic_type, harmonic_kind, harmonic_fret = harmonic
        string_fretted = StringFrettedTechniquePayload(
            slide_types=() if slide_types is None else slide_types,
            hopo_type=hopo_type,
            tapped=tapped,
            left_hand_tapped=left_hand_tapped,
            harmonic_type=harmonic_type,
            harmonic_kind=harmonic_kind,
            harmonic_fret=harmonic_fret,
            bend_enabled=bend_enabled,
        )

    if generic == GenericTechniqueFlags() and string_fretted is None:
        return None

    return TechniquePayload(generic=generic, string_fretted=string_fretted)


def _coerce_optional_int_array(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[int, ...] | None:
    if value is None:
        return None

    if not isinstance(value, list):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be an array when present.",
            )
        )
        return None

    integers: list[int] = []
    for index, item in enumerate(value):
        integer = _coerce_required_int(
            item,
            path=f"{path}[{index}]",
            diagnostics=diagnostics,
        )
        if integer is None:
            return None
        integers.append(integer)

    return tuple(integers)


def _coerce_optional_harmonic_mapping(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[int | None, int | None, float | None] | None:
    if value is None:
        return None

    harmonic = _coerce_required_mapping(value, path, diagnostics)
    if harmonic is None:
        return None

    enabled = _coerce_optional_bool_field(
        harmonic.get("enabled"),
        path=f"{path}.enabled",
        diagnostics=diagnostics,
    )
    if enabled is False:
        return None

    harmonic_type = _coerce_optional_int(
        harmonic.get("type"),
        path=f"{path}.type",
        diagnostics=diagnostics,
    )
    harmonic_kind = _coerce_optional_int(
        harmonic.get("kind"),
        path=f"{path}.kind",
        diagnostics=diagnostics,
    )
    harmonic_fret = _coerce_optional_number(
        harmonic.get("fret"),
        path=f"{path}.fret",
        diagnostics=diagnostics,
    )
    return (harmonic_type, harmonic_kind, harmonic_fret)


def _coerce_optional_enabled_mapping(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> bool:
    if value is None:
        return False

    mapping = _coerce_required_mapping(value, path, diagnostics)
    if mapping is None:
        return False

    enabled = _coerce_optional_bool_field(
        mapping.get("enabled"),
        path=f"{path}.enabled",
        diagnostics=diagnostics,
    )
    return bool(enabled)


def _emit_point_control_models(
    score: dict[str, Any],
    resolution_context: _ControlResolutionContext,
) -> tuple[tuple[PointControlEvent, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    raw_point_controls = score.get("pointControls")
    if not isinstance(raw_point_controls, list):
        diagnostics.append(
            _error(
                path="pointControls",
                code="missing_canonical_field",
                message="canonical field 'pointControls' is required.",
            )
        )
        return (), diagnostics

    seeds: list[_PointControlSeed] = []
    for index, raw_point_control in enumerate(raw_point_controls):
        control_path = f"pointControls[{index}]"
        seed = _coerce_point_control_seed(
            raw_point_control=raw_point_control,
            control_path=control_path,
            resolution_context=resolution_context,
            diagnostics=diagnostics,
        )
        if seed is not None:
            seeds.append(seed)

    seeds.sort(key=lambda item: item.sort_key())
    scope_ordinals: dict[str, int] = {}
    point_control_events: list[PointControlEvent] = []

    for seed in seeds:
        scope_key = seed.scope.value
        scope_ordinal = scope_ordinals.get(scope_key, 0)
        scope_ordinals[scope_key] = scope_ordinal + 1
        try:
            point_control_events.append(
                PointControlEvent(
                    control_id=build_point_control_id(scope_key, scope_ordinal),
                    kind=seed.kind,
                    scope=seed.scope,
                    target_ref=seed.target_ref,
                    time=seed.time,
                    value=seed.value,
                )
            )
        except ValueError as exc:
            diagnostics.append(
                _error(
                    path=seed.path,
                    code="invalid_canonical_field",
                    message=str(exc),
                )
            )

    return tuple(point_control_events), diagnostics


def _coerce_point_control_seed(
    raw_point_control: object,
    control_path: str,
    resolution_context: _ControlResolutionContext,
    diagnostics: list[IrBuildDiagnostic],
) -> _PointControlSeed | None:
    if not isinstance(raw_point_control, Mapping):
        diagnostics.append(
            _error(
                path=control_path,
                code="invalid_canonical_field",
                message="pointControls entries must be objects.",
            )
        )
        return None

    kind = _coerce_point_control_kind(
        raw_point_control.get("kind"),
        path=f"{control_path}.kind",
        diagnostics=diagnostics,
    )
    position = _resolve_control_position(
        value=raw_point_control.get("position"),
        path=f"{control_path}.position",
        bar_times=resolution_context.bar_times,
        diagnostics=diagnostics,
    )
    resolved_target = _resolve_control_target(
        raw_point_control=raw_point_control,
        control_path=control_path,
        position=position,
        resolution_context=resolution_context,
        diagnostics=diagnostics,
    )
    value = (
        None
        if kind is None
        else _coerce_point_control_value(
            raw_point_control=raw_point_control,
            control_path=control_path,
            kind=kind,
            diagnostics=diagnostics,
        )
    )
    if kind is None or position is None or resolved_target is None or value is None:
        return None

    scope, target_ref = resolved_target
    return _PointControlSeed(
        path=control_path,
        scope=scope,
        target_ref=target_ref,
        time=position.time,
        kind=kind,
        value=value,
    )


def _coerce_point_control_kind(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> PointControlKind | None:
    if value is None:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message="point controls must include a kind.",
            )
        )
        return None

    if not isinstance(value, str):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="point control kind must be a string.",
            )
        )
        return None

    kind = POINT_CONTROL_KIND_MAP.get(value)
    if kind is None:
        diagnostics.append(
            _error(
                path=path,
                code="unsupported_point_control_kind",
                message=f"point control kind '{value}' is not supported.",
            )
        )
        return None

    return kind


def _resolve_control_position(
    value: Any,
    path: str,
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]],
    diagnostics: list[IrBuildDiagnostic],
) -> _ResolvedControlPosition | None:
    position = _coerce_required_mapping(value, path, diagnostics)
    if position is None:
        return None

    bar_index = _coerce_required_int(
        position.get("barIndex"),
        path=f"{path}.barIndex",
        diagnostics=diagnostics,
    )
    offset = _coerce_score_time(
        position.get("offset"),
        path=f"{path}.offset",
        diagnostics=diagnostics,
        require_positive=False,
    )
    if bar_index is None or offset is None:
        return None

    bar_time = bar_times.get(bar_index)
    if bar_time is None:
        diagnostics.append(
            _error(
                path=f"{path}.barIndex",
                code="missing_bar_reference",
                message=f"no written-time map exists for bar index {bar_index}.",
            )
        )
        return None

    bar_start, bar_duration = bar_time
    if offset > bar_duration:
        diagnostics.append(
            _error(
                path=f"{path}.offset",
                code="invalid_canonical_field",
                message=f"control position exceeds the duration of bar {bar_index}.",
            )
        )
        return None

    return _ResolvedControlPosition(bar_index=bar_index, time=bar_start + offset)


def _resolve_control_target(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    position: _ResolvedControlPosition | None,
    resolution_context: _ControlResolutionContext,
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[ControlScope, str] | None:
    resolved_target: tuple[ControlScope, str] | None = None
    scope = _coerce_control_scope(
        raw_point_control.get("scope"),
        path=f"{control_path}.scope",
        diagnostics=diagnostics,
    )
    if scope is None:
        return None

    if scope is ControlScope.SCORE:
        resolved_target = (scope, ControlScope.SCORE.value)
    else:
        part_identifier = _resolve_part_target_identifier(
            raw_point_control=raw_point_control,
            control_path=control_path,
            known_part_ids=resolution_context.known_part_ids,
            diagnostics=diagnostics,
        )
        if part_identifier is not None:
            if scope is ControlScope.PART:
                resolved_target = (scope, part_identifier)
            else:
                staff_identifier = _resolve_staff_target_identifier(
                    raw_point_control=raw_point_control,
                    control_path=control_path,
                    part_identifier=part_identifier,
                    known_staff_ids=resolution_context.known_staff_ids,
                    diagnostics=diagnostics,
                )
                if staff_identifier is not None:
                    if scope is ControlScope.STAFF:
                        resolved_target = (scope, staff_identifier)
                    elif position is not None:
                        voice_lane_identifier = _resolve_voice_lane_target_identifier(
                            raw_point_control=raw_point_control,
                            context=_VoiceLaneTargetResolutionContext(
                                control_path=control_path,
                                staff_identifier=staff_identifier,
                                bar_index=position.bar_index,
                                known_voice_lane_ids=(
                                    resolution_context.known_voice_lane_ids
                                ),
                            ),
                            diagnostics=diagnostics,
                        )
                        if voice_lane_identifier is not None:
                            resolved_target = (scope, voice_lane_identifier)

    return resolved_target


def _coerce_control_scope(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> ControlScope | None:
    if value is None:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message="controls must include a scope.",
            )
        )
        return None

    if not isinstance(value, str):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="control scope must be a string.",
            )
        )
        return None

    scope = SOURCE_SCOPE_MAP.get(value)
    if scope is None:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=f"control scope '{value}' is not supported.",
            )
        )
        return None

    return scope


def _resolve_part_target_identifier(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    known_part_ids: frozenset[str],
    diagnostics: list[IrBuildDiagnostic],
) -> str | None:
    track_identity = _coerce_required_track_identity(
        raw_point_control.get("trackId"),
        path=f"{control_path}.trackId",
        diagnostics=diagnostics,
    )
    if track_identity is None:
        return None

    part_identifier = build_part_id(track_identity)
    if part_identifier not in known_part_ids:
        diagnostics.append(
            _error(
                path=f"{control_path}.trackId",
                code="missing_part_reference",
                message=f"no emitted part exists for track id '{track_identity}'.",
            )
        )
        return None

    return part_identifier


def _resolve_staff_target_identifier(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    part_identifier: str,
    known_staff_ids: frozenset[str],
    diagnostics: list[IrBuildDiagnostic],
) -> str | None:
    staff_index = _coerce_required_int(
        raw_point_control.get("staffIndex"),
        path=f"{control_path}.staffIndex",
        diagnostics=diagnostics,
    )
    if staff_index is None:
        return None

    staff_identifier = build_staff_id(part_identifier, staff_index)
    if staff_identifier not in known_staff_ids:
        diagnostics.append(
            _error(
                path=f"{control_path}.staffIndex",
                code="missing_staff_reference",
                message=f"no emitted staff exists for staff id '{staff_identifier}'.",
            )
        )
        return None

    return staff_identifier


def _resolve_voice_lane_target_identifier(
    raw_point_control: Mapping[str, Any],
    context: _VoiceLaneTargetResolutionContext,
    diagnostics: list[IrBuildDiagnostic],
) -> str | None:
    voice_index = _coerce_required_int(
        raw_point_control.get("voiceIndex"),
        path=f"{context.control_path}.voiceIndex",
        diagnostics=diagnostics,
    )
    if voice_index is None:
        return None

    voice_lane_identifier = build_voice_lane_id(
        context.staff_identifier,
        context.bar_index,
        voice_index,
    )
    if voice_lane_identifier not in context.known_voice_lane_ids:
        diagnostics.append(
            _error(
                path=f"{context.control_path}.voiceIndex",
                code="missing_voice_lane_reference",
                message=(
                    "no emitted voice lane exists for "
                    f"'{voice_lane_identifier}' at the control position."
                ),
            )
        )
        return None

    return voice_lane_identifier


def _coerce_point_control_value(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    kind: PointControlKind,
    diagnostics: list[IrBuildDiagnostic],
) -> TempoChangeValue | DynamicChangeValue | FermataValue | None:
    if kind is PointControlKind.TEMPO_CHANGE:
        return _coerce_tempo_point_control_value(
            raw_point_control=raw_point_control,
            control_path=control_path,
            diagnostics=diagnostics,
        )

    if kind is PointControlKind.DYNAMIC_CHANGE:
        return _coerce_dynamic_point_control_value(
            raw_point_control=raw_point_control,
            control_path=control_path,
            diagnostics=diagnostics,
        )

    return _coerce_fermata_point_control_value(
        raw_point_control=raw_point_control,
        control_path=control_path,
        diagnostics=diagnostics,
    )


def _coerce_tempo_point_control_value(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> TempoChangeValue | None:
    beats_per_minute = _coerce_required_number(
        raw_point_control.get("numericValue"),
        path=f"{control_path}.numericValue",
        diagnostics=diagnostics,
    )
    if beats_per_minute is None:
        return None

    try:
        return TempoChangeValue(beats_per_minute=beats_per_minute)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=f"{control_path}.numericValue",
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_dynamic_point_control_value(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> DynamicChangeValue | None:
    marking = raw_point_control.get("value")
    if marking is None:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="missing_canonical_field",
                message="dynamic point controls must include a value.",
            )
        )
        return None

    if not isinstance(marking, str):
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message="dynamic point control value must be a string.",
            )
        )
        return None

    try:
        return DynamicChangeValue(marking=marking)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_fermata_point_control_value(
    raw_point_control: Mapping[str, Any],
    control_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> FermataValue | None:
    fermata_type = _coerce_optional_str(
        raw_point_control.get("value"),
        path=f"{control_path}.value",
        diagnostics=diagnostics,
    )
    length_scale = _coerce_optional_number(
        raw_point_control.get("length"),
        path=f"{control_path}.length",
        diagnostics=diagnostics,
    )
    _coerce_optional_str(
        raw_point_control.get("placement"),
        path=f"{control_path}.placement",
        diagnostics=diagnostics,
    )
    try:
        return FermataValue(
            fermata_type=fermata_type,
            length_scale=length_scale,
        )
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=control_path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _emit_span_control_models(
    score: dict[str, Any],
    resolution_context: _ControlResolutionContext,
) -> tuple[tuple[SpanControlEvent, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    raw_span_controls = score.get("spanControls")
    if not isinstance(raw_span_controls, list):
        diagnostics.append(
            _error(
                path="spanControls",
                code="missing_canonical_field",
                message="canonical field 'spanControls' is required.",
            )
        )
        return (), diagnostics

    seeds: list[_SpanControlSeed] = []
    for index, raw_span_control in enumerate(raw_span_controls):
        control_path = f"spanControls[{index}]"
        seed = _coerce_span_control_seed(
            raw_span_control=raw_span_control,
            control_path=control_path,
            resolution_context=resolution_context,
            diagnostics=diagnostics,
        )
        if seed is not None:
            seeds.append(seed)

    seeds.sort(key=lambda item: item.sort_key())
    scope_ordinals: dict[str, int] = {}
    span_control_events: list[SpanControlEvent] = []

    for seed in seeds:
        scope_key = seed.scope.value
        scope_ordinal = scope_ordinals.get(scope_key, 0)
        scope_ordinals[scope_key] = scope_ordinal + 1
        try:
            span_control_events.append(
                SpanControlEvent(
                    control_id=build_span_control_id(scope_key, scope_ordinal),
                    kind=seed.kind,
                    scope=seed.scope,
                    target_ref=seed.target_ref,
                    start_time=seed.start_time,
                    end_time=seed.end_time,
                    value=seed.value,
                    start_anchor_ref=seed.start_anchor_ref,
                    end_anchor_ref=seed.end_anchor_ref,
                )
            )
        except ValueError as exc:
            diagnostics.append(
                _error(
                    path=seed.path,
                    code="invalid_canonical_field",
                    message=str(exc),
                )
            )

    return tuple(span_control_events), diagnostics


def _coerce_span_control_seed(
    raw_span_control: object,
    control_path: str,
    resolution_context: _ControlResolutionContext,
    diagnostics: list[IrBuildDiagnostic],
) -> _SpanControlSeed | None:
    if not isinstance(raw_span_control, Mapping):
        diagnostics.append(
            _error(
                path=control_path,
                code="invalid_canonical_field",
                message="spanControls entries must be objects.",
            )
        )
        return None

    kind = _coerce_span_control_kind(
        raw_span_control.get("kind"),
        path=f"{control_path}.kind",
        diagnostics=diagnostics,
    )
    start_position = _resolve_control_position(
        value=raw_span_control.get("start"),
        path=f"{control_path}.start",
        bar_times=resolution_context.bar_times,
        diagnostics=diagnostics,
    )
    end_position = _resolve_control_position(
        value=raw_span_control.get("end"),
        path=f"{control_path}.end",
        bar_times=resolution_context.bar_times,
        diagnostics=diagnostics,
    )
    resolved_target = _resolve_control_target(
        raw_point_control=raw_span_control,
        control_path=control_path,
        position=start_position,
        resolution_context=resolution_context,
        diagnostics=diagnostics,
    )
    value = (
        None
        if kind is None
        else _coerce_span_control_value(
            raw_span_control=raw_span_control,
            control_path=control_path,
            kind=kind,
            diagnostics=diagnostics,
        )
    )
    if (
        kind is None
        or start_position is None
        or end_position is None
        or resolved_target is None
        or value is None
    ):
        return None

    scope, target_ref = resolved_target
    return _SpanControlSeed(
        path=control_path,
        scope=scope,
        target_ref=target_ref,
        start_time=start_position.time,
        end_time=end_position.time,
        kind=kind,
        value=value,
    )


def _coerce_span_control_kind(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> SpanControlKind | None:
    if value is None:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message="span controls must include a kind.",
            )
        )
        return None

    if not isinstance(value, str):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="span control kind must be a string.",
            )
        )
        return None

    kind = SPAN_CONTROL_KIND_MAP.get(value)
    if kind is not None:
        return kind

    severity = (
        DiagnosticSeverity.WARNING if value == "Legato" else DiagnosticSeverity.ERROR
    )
    diagnostics.append(
        IrBuildDiagnostic(
            severity=severity,
            code="unsupported_span_control_kind",
            path=path,
            message=f"span control kind '{value}' is not supported in IR v1.",
        )
    )
    return None


def _coerce_span_control_value(
    raw_span_control: Mapping[str, Any],
    control_path: str,
    kind: SpanControlKind,
    diagnostics: list[IrBuildDiagnostic],
) -> HairpinValue | OttavaValue | None:
    if kind is SpanControlKind.HAIRPIN:
        return _coerce_hairpin_span_control_value(
            raw_span_control=raw_span_control,
            control_path=control_path,
            diagnostics=diagnostics,
        )

    return _coerce_ottava_span_control_value(
        raw_span_control=raw_span_control,
        control_path=control_path,
        diagnostics=diagnostics,
    )


def _coerce_hairpin_span_control_value(
    raw_span_control: Mapping[str, Any],
    control_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> HairpinValue | None:
    raw_value = raw_span_control.get("value")
    if raw_value is None:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="missing_canonical_field",
                message="hairpin span controls must include a value.",
            )
        )
        return None

    if not isinstance(raw_value, str):
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message="hairpin span control value must be a string.",
            )
        )
        return None

    direction = raw_value.strip().casefold()
    if direction == "diminuendo":
        direction = HairpinDirection.DECRESCENDO.value

    try:
        return HairpinValue(direction=direction, niente=False)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _coerce_ottava_span_control_value(
    raw_span_control: Mapping[str, Any],
    control_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> OttavaValue | None:
    raw_value = raw_span_control.get("value")
    if raw_value is None:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="missing_canonical_field",
                message="ottava span controls must include a value.",
            )
        )
        return None

    if not isinstance(raw_value, str):
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message="ottava span control value must be a string.",
            )
        )
        return None

    octave_shift = {
        "8va": 1,
        "8vb": -1,
        "15ma": 2,
        "15mb": -2,
    }.get(raw_value.strip().casefold())
    if octave_shift is None:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message=f"ottava marking '{raw_value}' is not supported.",
            )
        )
        return None

    try:
        return OttavaValue(octave_shift=octave_shift)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=f"{control_path}.value",
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _validate_track_surfaces(
    tracks: list[object],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for track_index, track in enumerate(tracks):
        track_path = f"tracks[{track_index}]"
        if not isinstance(track, Mapping):
            diagnostics.append(
                _error(
                    path=track_path,
                    code="invalid_canonical_field",
                    message="tracks entries must be objects.",
                )
            )
            continue

        staves = _require_list_field(
            track,
            field_name="staves",
            path=f"{track_path}.staves",
            diagnostics=diagnostics,
        )
        if staves is None:
            continue

        for staff_index, staff in enumerate(staves):
            staff_path = f"{track_path}.staves[{staff_index}]"
            if not isinstance(staff, Mapping):
                diagnostics.append(
                    _error(
                        path=staff_path,
                        code="invalid_canonical_field",
                        message="staves entries must be objects.",
                    )
                )
                continue

            measures = _require_list_field(
                staff,
                field_name="measures",
                path=f"{staff_path}.measures",
                diagnostics=diagnostics,
            )
            if measures is None:
                continue

            for measure_index, measure in enumerate(measures):
                measure_path = f"{staff_path}.measures[{measure_index}]"
                if not isinstance(measure, Mapping):
                    diagnostics.append(
                        _error(
                            path=measure_path,
                            code="invalid_canonical_field",
                            message="measures entries must be objects.",
                        )
                    )
                    continue

                voices = _require_list_field(
                    measure,
                    field_name="voices",
                    path=f"{measure_path}.voices",
                    diagnostics=diagnostics,
                )
                if voices is None:
                    continue

                for voice_index, voice in enumerate(voices):
                    voice_path = f"{measure_path}.voices[{voice_index}]"
                    if not isinstance(voice, Mapping):
                        diagnostics.append(
                            _error(
                                path=voice_path,
                                code="invalid_canonical_field",
                                message="voices entries must be objects.",
                            )
                        )
                        continue

                    beats = _require_list_field(
                        voice,
                        field_name="beats",
                        path=f"{voice_path}.beats",
                        diagnostics=diagnostics,
                    )
                    if beats is None:
                        continue

                    _validate_beat_surfaces(beats, voice_path, diagnostics)


def _validate_beat_surfaces(
    beats: list[object],
    voice_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for beat_index, beat in enumerate(beats):
        beat_path = f"{voice_path}.beats[{beat_index}]"
        if not isinstance(beat, Mapping):
            diagnostics.append(
                _error(
                    path=beat_path,
                    code="invalid_canonical_field",
                    message="beats entries must be objects.",
                )
            )
            continue

        _require_score_time_field(
            beat,
            field_name="offset",
            path=f"{beat_path}.offset",
            diagnostics=diagnostics,
            require_positive=False,
        )
        _require_score_time_field(
            beat,
            field_name="duration",
            path=f"{beat_path}.duration",
            diagnostics=diagnostics,
            require_positive=True,
        )

        notes = _require_list_field(
            beat,
            field_name="notes",
            path=f"{beat_path}.notes",
            diagnostics=diagnostics,
        )
        if notes is None:
            continue

        _validate_note_surfaces(notes, beat_path, diagnostics)


def _validate_note_surfaces(
    notes: list[object],
    beat_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for note_index, note in enumerate(notes):
        note_path = f"{beat_path}.notes[{note_index}]"
        if not isinstance(note, Mapping):
            diagnostics.append(
                _error(
                    path=note_path,
                    code="invalid_canonical_field",
                    message="notes entries must be objects.",
                )
            )
            continue

        articulation = note.get("articulation")
        if articulation is None:
            continue

        articulation_path = f"{note_path}.articulation"
        if not isinstance(articulation, Mapping):
            diagnostics.append(
                _error(
                    path=articulation_path,
                    code="invalid_canonical_field",
                    message="articulation must be an object when present.",
                )
            )
            continue

        relations = articulation.get("relations")
        relations_path = f"{articulation_path}.relations"
        if relations is None:
            if _has_relation_hints(articulation):
                diagnostics.append(
                    _warning(
                        path=relations_path,
                        code="missing_canonical_field",
                        message=(
                            "articulation exposes linked-note flags without the "
                            "canonical relations array."
                        ),
                    )
                )
            continue

        if not isinstance(relations, list):
            diagnostics.append(
                _error(
                    path=relations_path,
                    code="invalid_canonical_field",
                    message="relations must be an array when present.",
                )
            )
            continue

        for relation_index, relation in enumerate(relations):
            relation_path = f"{relations_path}[{relation_index}]"
            if not isinstance(relation, Mapping):
                diagnostics.append(
                    _error(
                        path=relation_path,
                        code="invalid_canonical_field",
                        message="relations entries must be objects.",
                    )
                )
                continue

            if (
                not isinstance(relation.get("kind"), str)
                or not relation["kind"].strip()
            ):
                diagnostics.append(
                    _error(
                        path=f"{relation_path}.kind",
                        code="invalid_canonical_field",
                        message="relations entries must include a non-empty kind.",
                    )
                )

            if not isinstance(relation.get("targetNoteId"), int):
                diagnostics.append(
                    _error(
                        path=f"{relation_path}.targetNoteId",
                        code="invalid_canonical_field",
                        message=(
                            "relations entries must include an integer targetNoteId."
                        ),
                    )
                )


def _require_list_field(
    mapping: Mapping[str, Any],
    field_name: str,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> list[object] | None:
    value = mapping.get(field_name)
    if value is None:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message=f"canonical field '{field_name}' is required.",
            )
        )
        return None

    if not isinstance(value, list):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=f"canonical field '{field_name}' must be an array.",
            )
        )
        return None

    return value


def _require_score_time_field(
    mapping: Mapping[str, Any],
    field_name: str,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
    *,
    require_positive: bool,
) -> None:
    if field_name not in mapping:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message=f"canonical field '{field_name}' is required.",
            )
        )
        return

    _coerce_score_time(
        mapping[field_name],
        path=path,
        diagnostics=diagnostics,
        require_positive=require_positive,
    )


def _coerce_score_time(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
    *,
    require_positive: bool,
) -> ScoreTime | None:
    score_time: ScoreTime | None = None
    message: str | None = None

    if value is None:
        message = "canonical ScoreTime field is required."
    elif not isinstance(value, Mapping):
        message = "canonical field must be a ScoreTime object."
    else:
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            message = "ScoreTime fields must include integer numerator and denominator values."
        else:
            try:
                score_time = ScoreTime(numerator=numerator, denominator=denominator)
            except ValueError as exc:
                message = str(exc)
            else:
                if require_positive and score_time.numerator <= 0:
                    message = "ScoreTime durations must be positive."
                elif not require_positive and score_time.numerator < 0:
                    message = "ScoreTime positions must be non-negative."

    if message is not None:
        diagnostics.append(
            _error(
                path=path,
                code=(
                    "missing_canonical_field"
                    if value is None
                    else "invalid_canonical_field"
                ),
                message=message,
            )
        )
        return None

    return score_time


def _coerce_optional_bool(
    value: Any,
    diagnostics: list[IrBuildDiagnostic],
) -> bool:
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    diagnostics.append(
        _error(
            path="anacrusis",
            code="invalid_canonical_field",
            message="anacrusis must be a boolean when present.",
        )
    )
    return False


def _build_bar_geometry_warnings(
    score: dict[str, Any],
    raw_entries: list[tuple[int, int, ScoreTime, ScoreTime]],
    bars: tuple[WrittenTimeMapEntry, ...],
) -> list[IrBuildDiagnostic]:
    if not bars or not raw_entries:
        return []

    first_bar = bars[0]
    _, first_ordinal, _, _ = raw_entries[0]
    nominal_duration = _parse_nominal_bar_duration(score["timelineBars"][first_ordinal])
    if nominal_duration is None:
        return []

    if first_bar.is_anacrusis and first_bar.duration >= nominal_duration:
        return [
            _warning(
                path=f"timelineBars[{first_ordinal}].duration",
                code="suspicious_bar_geometry",
                message=(
                    "anacrusis is true but the first bar is not shorter than its "
                    "nominal time-signature duration."
                ),
            )
        ]

    if not first_bar.is_anacrusis and first_bar.duration < nominal_duration:
        return [
            _warning(
                path=f"timelineBars[{first_ordinal}].duration",
                code="suspicious_bar_geometry",
                message=(
                    "the first bar is shorter than its nominal time-signature "
                    "duration but anacrusis is not set."
                ),
            )
        ]

    return []


def _parse_nominal_bar_duration(timeline_bar: Any) -> ScoreTime | None:
    if not isinstance(timeline_bar, Mapping):
        return None

    time_signature = timeline_bar.get("timeSignature")
    if not isinstance(time_signature, str):
        return None

    numerator_text, separator, denominator_text = time_signature.partition("/")
    if separator != "/":
        return None

    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        return None

    try:
        return ScoreTime(numerator=numerator, denominator=denominator)
    except ValueError:
        return None


def _coerce_required_mapping(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be an object.",
            )
        )
        return None

    return value


def _coerce_required_int(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> int | None:
    if not isinstance(value, int):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be an integer.",
            )
        )
        return None

    return value


def _coerce_required_track_identity(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> int | str | None:
    if isinstance(value, bool) or not isinstance(value, int | str):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be an integer or string track id.",
            )
        )
        return None

    return value


def _coerce_required_number(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be a number.",
            )
        )
        return None

    return float(value)


def _coerce_optional_int(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> int | None:
    if value is None:
        return None

    return _coerce_required_int(value, path, diagnostics)


def _coerce_optional_str(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be a string when present.",
            )
        )
        return None

    normalized = value.strip()
    if not normalized:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="field must be non-empty when present.",
            )
        )
        return None

    return normalized


def _coerce_optional_number(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> float | None:
    if value is None:
        return None

    return _coerce_required_number(value, path, diagnostics)


def _coerce_optional_bool_field(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> bool | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    diagnostics.append(
        _error(
            path=path,
            code="invalid_canonical_field",
            message="field must be a boolean when present.",
        )
    )
    return None


def _coerce_optional_tuning_pitches(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[int, ...] | None:
    if value is None:
        return None

    if not isinstance(value, Mapping):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="tuning must be an object when present.",
            )
        )
        return None

    pitches = value.get("pitches")
    if not isinstance(pitches, list):
        diagnostics.append(
            _error(
                path=f"{path}.pitches",
                code="invalid_canonical_field",
                message="tuning.pitches must be an array when tuning is present.",
            )
        )
        return None

    if any(not isinstance(pitch, int) for pitch in pitches):
        diagnostics.append(
            _error(
                path=f"{path}.pitches",
                code="invalid_canonical_field",
                message="tuning.pitches must contain only integers.",
            )
        )
        return None

    return tuple(pitches)


def _coerce_transposition(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> Transposition | None:
    transposition = _coerce_required_mapping(value, path, diagnostics)
    if transposition is None:
        return None

    chromatic = _coerce_required_int(
        transposition.get("chromatic"),
        path=f"{path}.chromatic",
        diagnostics=diagnostics,
    )
    octave = _coerce_required_int(
        transposition.get("octave"),
        path=f"{path}.octave",
        diagnostics=diagnostics,
    )
    if chromatic is None or octave is None:
        return None

    return Transposition(chromatic=chromatic, octave=octave)


def _coerce_time_signature(
    value: Any,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> TimeSignature | None:
    if not isinstance(value, str):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="timeSignature must be a string.",
            )
        )
        return None

    numerator_text, separator, denominator_text = value.partition("/")
    if separator != "/":
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="timeSignature must use the '<numerator>/<denominator>' form.",
            )
        )
        return None

    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        return TimeSignature(numerator=numerator, denominator=denominator)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return None


def _point_control_value_sort_key(
    value: TempoChangeValue | DynamicChangeValue | FermataValue,
) -> tuple[object, ...]:
    if isinstance(value, TempoChangeValue):
        return (value.beats_per_minute,)

    if isinstance(value, DynamicChangeValue):
        return (value.marking.casefold(), value.marking)

    return (
        "" if value.fermata_type is None else value.fermata_type.casefold(),
        value.fermata_type or "",
        -1.0 if value.length_scale is None else value.length_scale,
    )


def _span_control_value_sort_key(
    value: HairpinValue | OttavaValue,
) -> tuple[object, ...]:
    if isinstance(value, HairpinValue):
        return (value.direction.value, value.niente)

    return (value.octave_shift,)


def _voice_contains_authored_content(beats: list[object]) -> bool:
    return any(isinstance(beat, Mapping) for beat in beats)


def _has_relation_hints(articulation: Mapping[str, Any]) -> bool:
    return any(field_name in articulation for field_name in RELATION_HINT_FIELDS)


def _error(path: str, code: str, message: str) -> IrBuildDiagnostic:
    return IrBuildDiagnostic(
        severity=DiagnosticSeverity.ERROR,
        code=code,
        path=path,
        message=message,
    )


def _warning(path: str, code: str, message: str) -> IrBuildDiagnostic:
    return IrBuildDiagnostic(
        severity=DiagnosticSeverity.WARNING,
        code=code,
        path=path,
        message=message,
    )
