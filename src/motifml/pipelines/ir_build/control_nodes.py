"""Control-event emission nodes for IR build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.ids import part_id as build_part_id
from motifml.ir.ids import point_control_id as build_point_control_id
from motifml.ir.ids import span_control_id as build_span_control_id
from motifml.ir.ids import staff_id as build_staff_id
from motifml.ir.ids import voice_lane_id as build_voice_lane_id
from motifml.ir.models import (
    ControlScope,
    DynamicChangeValue,
    FermataValue,
    HairpinDirection,
    HairpinValue,
    OttavaValue,
    PointControlEvent,
    PointControlKind,
    SpanControlEvent,
    SpanControlKind,
    TempoChangeValue,
)
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.common import (
    POINT_CONTROL_KIND_MAP,
    SOURCE_SCOPE_MAP,
    SPAN_CONTROL_KIND_MAP,
    _coerce_optional_number,
    _coerce_optional_str,
    _coerce_required_int,
    _coerce_required_mapping,
    _coerce_required_number,
    _coerce_required_track_identity,
    _coerce_score_time,
    _ControlResolutionContext,
    _error,
    _PointControlSeed,
    _ResolvedControlPosition,
    _SpanControlSeed,
    _VoiceLaneTargetResolutionContext,
    _warning,
)
from motifml.pipelines.ir_build.models import (
    DiagnosticSeverity,
    IrBuildDiagnostic,
    PartStaffEmissionResult,
    PointControlEmissionResult,
    SpanControlEmissionResult,
    VoiceLaneEmissionResult,
    WrittenTimeMapResult,
)


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
        overflow_is_warning=True,
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
    *,
    overflow_is_warning: bool = False,
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
        diagnostic = _warning if overflow_is_warning else _error
        code = (
            "out_of_range_control_position"
            if overflow_is_warning
            else "invalid_canonical_field"
        )
        message = (
            f"control position exceeds the duration of bar {bar_index}; "
            "the control is skipped."
            if overflow_is_warning
            else f"control position exceeds the duration of bar {bar_index}."
        )
        diagnostics.append(
            diagnostic(path=f"{path}.offset", code=code, message=message)
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
    if length_scale is not None and length_scale <= 0:
        diagnostics.append(
            _warning(
                path=f"{control_path}.length",
                code="non_positive_fermata_length_scale",
                message=(
                    "fermata point control length must be positive when present; "
                    "dropping the length scale."
                ),
            )
        )
        length_scale = None
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
    raw_end = raw_span_control.get("end")
    end_position: _ResolvedControlPosition | None = None
    if raw_end is None:
        diagnostics.append(
            _warning(
                path=f"{control_path}.end",
                code="open_ended_span_control",
                message=(
                    "span controls without an end position are skipped because "
                    "IR v1 requires bounded spans."
                ),
            )
        )
    else:
        end_position = _resolve_control_position(
            value=raw_end,
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
