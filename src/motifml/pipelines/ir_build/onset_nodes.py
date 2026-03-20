"""Onset-group emission nodes for IR build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.ids import bar_id as build_bar_id
from motifml.ir.ids import onset_id as build_onset_id
from motifml.ir.ids import part_id as build_part_id
from motifml.ir.ids import staff_id as build_staff_id
from motifml.ir.ids import voice_lane_id as build_voice_lane_id
from motifml.ir.models import (
    GenericTechniqueFlags,
    OnsetGroup,
    RhythmBaseValue,
    RhythmShape,
    StringFrettedTechniquePayload,
    TechniquePayload,
    TupletRatio,
)
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.common import (
    UNSUPPORTED_ONSET_TECHNIQUE_FIELDS,
    _coerce_optional_bool_field,
    _coerce_optional_str,
    _coerce_required_int,
    _coerce_required_mapping,
    _coerce_score_time,
    _error,
    _OnsetBeatBuildContext,
    _OnsetStaffBuildContext,
    _OnsetVoiceBuildContext,
    _require_list_field,
    _warning,
)
from motifml.pipelines.ir_build.models import (
    IrBuildDiagnostic,
    OnsetGroupEmissionResult,
    VoiceLaneEmissionResult,
    WrittenTimeMapResult,
)


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
