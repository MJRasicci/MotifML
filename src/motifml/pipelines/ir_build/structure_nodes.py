"""Part, bar, and voice-lane emission nodes for IR build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.ids import bar_id as build_bar_id
from motifml.ir.ids import part_id as build_part_id
from motifml.ir.ids import staff_id as build_staff_id
from motifml.ir.ids import voice_lane_chain_id as build_voice_lane_chain_id
from motifml.ir.ids import voice_lane_id as build_voice_lane_id
from motifml.ir.models import Bar, Part, Staff, VoiceLane
from motifml.pipelines.ir_build.common import (
    _coerce_optional_int,
    _coerce_optional_str,
    _coerce_optional_tuning_pitches,
    _coerce_required_int,
    _coerce_required_mapping,
    _coerce_time_signature,
    _coerce_transposition,
    _error,
    _require_list_field,
    _voice_contains_authored_content,
    _VoiceLaneBuildContext,
    _VoiceLaneStaffBuildContext,
)
from motifml.pipelines.ir_build.models import (
    BarEmissionResult,
    CanonicalScoreValidationResult,
    IrBuildDiagnostic,
    PartStaffEmissionResult,
    VoiceLaneEmissionResult,
    WrittenTimeMapResult,
)


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
