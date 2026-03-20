"""Shared helpers for IR build validation and emission layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motifml.ir.ids import note_sort_key
from motifml.ir.models import (
    ControlScope,
    DynamicChangeValue,
    EdgeType,
    FermataValue,
    HairpinValue,
    OnsetGroup,
    OttavaValue,
    Pitch,
    PointControlKind,
    SpanControlKind,
    TechniquePayload,
    TempoChangeValue,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.models import (
    DiagnosticSeverity,
    IrBuildDiagnostic,
    WrittenTimeMapEntry,
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
NOTE_RELATION_EDGE_TYPE_MAP = {
    "tie": EdgeType.TIE_TO,
    "hammeron": EdgeType.TECHNIQUE_TO,
    "pulloff": EdgeType.TECHNIQUE_TO,
    "slide": EdgeType.TECHNIQUE_TO,
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
IR_DOCUMENT_OUTPUT_ROOT = Path("data/02_intermediate/ir/documents")


@dataclass(frozen=True, slots=True)
class _VoiceLaneBuildContext:
    part_identifier: str
    staff_identifier: str
    bar_index: int
    bar_id: str
    voice_path: str


@dataclass(frozen=True, slots=True)
class _VoiceLaneStaffBuildContext:
    part_identifier: str
    staff_path: str
    known_staff_ids: frozenset[str]
    bar_ids_by_index: dict[int, str]


@dataclass(frozen=True, slots=True)
class _OnsetStaffBuildContext:
    part_identifier: str
    staff_path: str
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _OnsetVoiceBuildContext:
    staff_identifier: str
    bar_index: int
    bar_id: str
    bar_start: ScoreTime
    bar_duration: ScoreTime
    voice_path: str
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _OnsetBeatBuildContext:
    voice_lane_id: str
    bar_index: int
    bar_id: str
    bar_start: ScoreTime
    bar_duration: ScoreTime
    beat_path: str
    attack_order_in_voice: int


@dataclass(frozen=True, slots=True)
class _NoteTrackBuildContext:
    track_path: str
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    voice_lanes_by_id: dict[str, VoiceLane]
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]]


@dataclass(frozen=True, slots=True)
class _NoteStaffBuildContext:
    part_identifier: str
    staff_path: str
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    voice_lanes_by_id: dict[str, VoiceLane]
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]]


@dataclass(frozen=True, slots=True)
class _NoteVoiceTraversalContext:
    staff_identifier: str
    bar_index: int
    bar_duration: ScoreTime
    voice_path: str
    voice_lanes_by_id: dict[str, VoiceLane]
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]]


@dataclass(frozen=True, slots=True)
class _NoteBeatBuildContext:
    voice_lane: VoiceLane


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class _PendingNoteRelation:
    path: str
    kind: str
    source_raw_note_id: int
    target_raw_note_id: int

    def sort_key(self) -> tuple[str, str, int, int]:
        return (
            self.path,
            self.kind.casefold(),
            self.source_raw_note_id,
            self.target_raw_note_id,
        )


@dataclass(frozen=True, slots=True)
class _NoteRelationSeed:
    path: str
    source_raw_note_id: int
    note_seed: _NoteSeed
    relations: tuple[_PendingNoteRelation, ...] = ()

    def sort_key(self) -> tuple[object, ...]:
        return self.note_seed.sort_key()


@dataclass(frozen=True, slots=True)
class _ControlResolutionContext:
    bar_times: dict[int, tuple[ScoreTime, ScoreTime]]
    known_part_ids: frozenset[str]
    known_staff_ids: frozenset[str]
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ResolvedControlPosition:
    bar_index: int
    time: ScoreTime


@dataclass(frozen=True, slots=True)
class _VoiceLaneTargetResolutionContext:
    control_path: str
    staff_identifier: str
    bar_index: int
    known_voice_lane_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
