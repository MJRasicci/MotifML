"""Note-event emission nodes for IR build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.ids import note_id as build_note_id
from motifml.ir.ids import part_id as build_part_id
from motifml.ir.ids import staff_id as build_staff_id
from motifml.ir.ids import voice_lane_id as build_voice_lane_id
from motifml.ir.models import (
    GenericTechniqueFlags,
    NoteEvent,
    OnsetGroup,
    Pitch,
    StringFrettedTechniquePayload,
    TechniquePayload,
    VoiceLane,
)
from motifml.pipelines.ir_build.common import (
    _coerce_optional_bool_field,
    _coerce_optional_int,
    _coerce_optional_number,
    _coerce_optional_str,
    _coerce_required_int,
    _coerce_required_mapping,
    _coerce_score_time,
    _error,
    _NoteBeatBuildContext,
    _NoteSeed,
    _NoteStaffBuildContext,
    _NoteTrackBuildContext,
    _NoteVoiceTraversalContext,
    _require_list_field,
)
from motifml.pipelines.ir_build.models import (
    IrBuildDiagnostic,
    NoteEventEmissionResult,
    OnsetGroupEmissionResult,
    VoiceLaneEmissionResult,
    WrittenTimeMapResult,
)
from motifml.pipelines.ir_build.onset_nodes import _coerce_beat_core_fields


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

    # Motif exports unset string numbers as zero-valued placeholders.
    if string_number == 0:
        string_number = None
        if show_string_number is True:
            show_string_number = None

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
