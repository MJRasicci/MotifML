"""Intrinsic-edge emission nodes for IR build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.ids import part_id as build_part_id
from motifml.ir.ids import staff_id as build_staff_id
from motifml.ir.models import Edge, EdgeType, NoteEvent, OnsetGroup, VoiceLane
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.common import (
    NOTE_RELATION_EDGE_TYPE_MAP,
    _coerce_required_int,
    _coerce_required_mapping,
    _error,
    _NoteRelationSeed,
    _NoteStaffBuildContext,
    _NoteTrackBuildContext,
    _NoteVoiceTraversalContext,
    _PendingNoteRelation,
    _require_list_field,
    _warning,
)
from motifml.pipelines.ir_build.models import (
    BarEmissionResult,
    IntrinsicEdgeEmissionResult,
    IrBuildDiagnostic,
    NoteEventEmissionResult,
    OnsetGroupEmissionResult,
    PartStaffEmissionResult,
    VoiceLaneEmissionResult,
)
from motifml.pipelines.ir_build.note_nodes import (
    _coerce_note_seed,
    _resolve_note_voice_inputs,
)
from motifml.pipelines.ir_build.onset_nodes import _coerce_beat_core_fields


def emit_intrinsic_edges(  # noqa: PLR0913
    documents: list[MotifJsonDocument],
    part_staff_emissions: list[PartStaffEmissionResult],
    bar_emissions: list[BarEmissionResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
    onset_group_emissions: list[OnsetGroupEmissionResult],
    note_event_emissions: list[NoteEventEmissionResult],
) -> list[IntrinsicEdgeEmissionResult]:
    """Emit canonical intrinsic edges from authored structure and note relations."""
    part_staff_by_path = {
        result.relative_path: result for result in part_staff_emissions
    }
    bar_by_path = {result.relative_path: result for result in bar_emissions}
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    onset_group_by_path = {
        result.relative_path: result for result in onset_group_emissions
    }
    note_event_by_path = {
        result.relative_path: result for result in note_event_emissions
    }
    emissions: list[IntrinsicEdgeEmissionResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        part_staff_emission = part_staff_by_path.get(document.relative_path)
        bar_emission = bar_by_path.get(document.relative_path)
        voice_lane_emission = voice_lane_by_path.get(document.relative_path)
        onset_group_emission = onset_group_by_path.get(document.relative_path)
        note_event_emission = note_event_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        edges: tuple[Edge, ...] = ()

        if part_staff_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_part_staff_emission",
                    message="part/staff emission must run before intrinsic edges.",
                )
            )
        elif bar_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_bar_emission",
                    message="bar emission must run before intrinsic edges.",
                )
            )
        elif voice_lane_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_voice_lane_emission",
                    message="voice lane emission must run before intrinsic edges.",
                )
            )
        elif onset_group_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_onset_group_emission",
                    message="onset group emission must run before intrinsic edges.",
                )
            )
        elif note_event_emission is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_note_event_emission",
                    message="note event emission must run before intrinsic edges.",
                )
            )
        elif not part_staff_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="part_staff_emission_failed",
                    message=(
                        "intrinsic edges cannot be emitted because part/staff "
                        "emission contains fatal diagnostics."
                    ),
                )
            )
        elif not bar_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="bar_emission_failed",
                    message=(
                        "intrinsic edges cannot be emitted because bar emission "
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
                        "intrinsic edges cannot be emitted because voice lane "
                        "emission contains fatal diagnostics."
                    ),
                )
            )
        elif not onset_group_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="onset_group_emission_failed",
                    message=(
                        "intrinsic edges cannot be emitted because onset group "
                        "emission contains fatal diagnostics."
                    ),
                )
            )
        elif not note_event_emission.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="note_event_emission_failed",
                    message=(
                        "intrinsic edges cannot be emitted because note event "
                        "emission contains fatal diagnostics."
                    ),
                )
            )
        else:
            edges, emitted_diagnostics = _emit_intrinsic_edge_models(
                score=document.score,
                part_staff_emission=part_staff_emission,
                bar_emission=bar_emission,
                voice_lane_emission=voice_lane_emission,
                onset_group_emission=onset_group_emission,
                note_event_emission=note_event_emission,
            )
            diagnostics.extend(emitted_diagnostics)

        emissions.append(
            IntrinsicEdgeEmissionResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                edges=edges,
                diagnostics=tuple(diagnostics),
            )
        )

    return emissions


def _emit_intrinsic_edge_models(  # noqa: PLR0913
    score: dict[str, Any],
    part_staff_emission: PartStaffEmissionResult,
    bar_emission: BarEmissionResult,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
    note_event_emission: NoteEventEmissionResult,
) -> tuple[tuple[Edge, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    known_ids = _build_known_entity_ids(
        part_staff_emission=part_staff_emission,
        bar_emission=bar_emission,
        voice_lane_emission=voice_lane_emission,
        onset_group_emission=onset_group_emission,
        note_event_emission=note_event_emission,
    )
    seen_edges: set[tuple[str, str, str]] = set()
    edges: list[Edge] = []

    _append_contains_edges(
        part_staff_emission=part_staff_emission,
        bar_emission=bar_emission,
        voice_lane_emission=voice_lane_emission,
        onset_group_emission=onset_group_emission,
        note_event_emission=note_event_emission,
        known_ids=known_ids,
        seen_edges=seen_edges,
        edges=edges,
        diagnostics=diagnostics,
    )
    _append_next_in_voice_edges(
        voice_lane_emission=voice_lane_emission,
        onset_group_emission=onset_group_emission,
        known_ids=known_ids,
        seen_edges=seen_edges,
        edges=edges,
        diagnostics=diagnostics,
    )

    (
        raw_note_id_lookup,
        ambiguous_raw_note_ids,
        pending_relations,
        relation_diagnostics,
    ) = _build_note_relation_material(
        score=score,
        bar_emission=bar_emission,
        voice_lane_emission=voice_lane_emission,
        onset_group_emission=onset_group_emission,
        note_event_emission=note_event_emission,
    )
    diagnostics.extend(relation_diagnostics)
    _append_relation_edges(
        pending_relations=pending_relations,
        raw_note_id_lookup=raw_note_id_lookup,
        ambiguous_raw_note_ids=ambiguous_raw_note_ids,
        known_note_ids=known_ids["note"],
        seen_edges=seen_edges,
        edges=edges,
        diagnostics=diagnostics,
    )

    return tuple(edges), diagnostics


def _build_known_entity_ids(
    *,
    part_staff_emission: PartStaffEmissionResult,
    bar_emission: BarEmissionResult,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
    note_event_emission: NoteEventEmissionResult,
) -> dict[str, set[str]]:
    return {
        "part": {part.part_id for part in part_staff_emission.parts},
        "staff": {staff.staff_id for staff in part_staff_emission.staves},
        "bar": {bar.bar_id for bar in bar_emission.bars},
        "voice": {
            voice_lane.voice_lane_id for voice_lane in voice_lane_emission.voice_lanes
        },
        "onset": {
            onset_group.onset_id for onset_group in onset_group_emission.onset_groups
        },
        "note": {note_event.note_id for note_event in note_event_emission.note_events},
    }


def _append_contains_edges(  # noqa: PLR0913
    *,
    part_staff_emission: PartStaffEmissionResult,
    bar_emission: BarEmissionResult,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
    note_event_emission: NoteEventEmissionResult,
    known_ids: dict[str, set[str]],
    seen_edges: set[tuple[str, str, str]],
    edges: list[Edge],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for staff in part_staff_emission.staves:
        _append_edge(
            source_id=staff.part_id,
            target_id=staff.staff_id,
            edge_type=EdgeType.CONTAINS,
            path=f"staves.{staff.staff_id}",
            known_ids=known_ids,
            seen_edges=seen_edges,
            edges=edges,
            diagnostics=diagnostics,
        )

    known_bar_ids = {bar.bar_id for bar in bar_emission.bars}
    for voice_lane in voice_lane_emission.voice_lanes:
        path = f"voiceLanes.{voice_lane.voice_lane_id}"
        if voice_lane.bar_id not in known_bar_ids:
            diagnostics.append(
                _error(
                    path=path,
                    code="missing_bar_reference",
                    message=f"no emitted bar exists for '{voice_lane.bar_id}'.",
                )
            )
            continue
        _append_edge(
            source_id=voice_lane.bar_id,
            target_id=voice_lane.voice_lane_id,
            edge_type=EdgeType.CONTAINS,
            path=path,
            known_ids=known_ids,
            seen_edges=seen_edges,
            edges=edges,
            diagnostics=diagnostics,
        )

    known_voice_lane_ids = {
        voice_lane.voice_lane_id for voice_lane in voice_lane_emission.voice_lanes
    }
    for onset_group in onset_group_emission.onset_groups:
        path = f"onsetGroups.{onset_group.onset_id}"
        if onset_group.voice_lane_id not in known_voice_lane_ids:
            diagnostics.append(
                _error(
                    path=path,
                    code="missing_voice_lane_reference",
                    message=(
                        "no emitted voice lane exists for "
                        f"'{onset_group.voice_lane_id}'."
                    ),
                )
            )
            continue
        _append_edge(
            source_id=onset_group.voice_lane_id,
            target_id=onset_group.onset_id,
            edge_type=EdgeType.CONTAINS,
            path=path,
            known_ids=known_ids,
            seen_edges=seen_edges,
            edges=edges,
            diagnostics=diagnostics,
        )

    known_onset_ids = {
        onset_group.onset_id for onset_group in onset_group_emission.onset_groups
    }
    for note_event in note_event_emission.note_events:
        path = f"noteEvents.{note_event.note_id}"
        if note_event.onset_id not in known_onset_ids:
            diagnostics.append(
                _error(
                    path=path,
                    code="missing_onset_group_reference",
                    message=(
                        f"no emitted onset group exists for '{note_event.onset_id}'."
                    ),
                )
            )
            continue
        _append_edge(
            source_id=note_event.onset_id,
            target_id=note_event.note_id,
            edge_type=EdgeType.CONTAINS,
            path=path,
            known_ids=known_ids,
            seen_edges=seen_edges,
            edges=edges,
            diagnostics=diagnostics,
        )


def _append_next_in_voice_edges(  # noqa: PLR0913
    *,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
    known_ids: dict[str, set[str]],
    seen_edges: set[tuple[str, str, str]],
    edges: list[Edge],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    voice_lanes_by_id = {
        voice_lane.voice_lane_id: voice_lane
        for voice_lane in voice_lane_emission.voice_lanes
    }
    onset_groups_by_chain: dict[str, list[OnsetGroup]] = {}

    for onset_group in onset_group_emission.onset_groups:
        voice_lane = voice_lanes_by_id.get(onset_group.voice_lane_id)
        if voice_lane is None:
            diagnostics.append(
                _error(
                    path=f"onsetGroups.{onset_group.onset_id}",
                    code="missing_voice_lane_reference",
                    message=(
                        "no emitted voice lane exists for "
                        f"'{onset_group.voice_lane_id}'."
                    ),
                )
            )
            continue
        onset_groups_by_chain.setdefault(voice_lane.voice_lane_chain_id, []).append(
            onset_group
        )

    for voice_lane_chain_id, onset_groups in onset_groups_by_chain.items():
        ordered_onsets = sorted(
            onset_groups,
            key=lambda item: (
                item.time,
                item.attack_order_in_voice,
                item.onset_id,
            ),
        )
        for source_onset, target_onset in zip(ordered_onsets, ordered_onsets[1:]):
            _append_edge(
                source_id=source_onset.onset_id,
                target_id=target_onset.onset_id,
                edge_type=EdgeType.NEXT_IN_VOICE,
                path=f"voiceLaneChains.{voice_lane_chain_id}",
                known_ids=known_ids,
                seen_edges=seen_edges,
                edges=edges,
                diagnostics=diagnostics,
            )


def _build_note_relation_material(
    *,
    score: dict[str, Any],
    bar_emission: BarEmissionResult,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
    note_event_emission: NoteEventEmissionResult,
) -> tuple[
    dict[int, str], set[int], tuple[_PendingNoteRelation, ...], list[IrBuildDiagnostic]
]:
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
        return {}, set(), (), diagnostics

    bar_durations_by_index = {bar.bar_index: bar.duration for bar in bar_emission.bars}
    voice_lanes_by_id = {
        voice_lane.voice_lane_id: voice_lane
        for voice_lane in voice_lane_emission.voice_lanes
    }
    onset_groups_by_voice_lane: dict[str, dict[int, OnsetGroup]] = {}
    for onset_group in onset_group_emission.onset_groups:
        onset_groups_by_voice_lane.setdefault(onset_group.voice_lane_id, {})[
            onset_group.attack_order_in_voice
        ] = onset_group

    note_events_by_onset: dict[str, list[NoteEvent]] = {}
    for note_event in note_event_emission.note_events:
        note_events_by_onset.setdefault(note_event.onset_id, []).append(note_event)

    raw_note_id_lookup: dict[int, str] = {}
    ambiguous_raw_note_ids: set[int] = set()
    pending_relations: list[_PendingNoteRelation] = []

    for track_index, track in enumerate(tracks):
        _collect_note_relation_material_for_track(
            context=_NoteTrackBuildContext(
                track_path=f"tracks[{track_index}]",
                bar_times={},
                voice_lanes_by_id=voice_lanes_by_id,
                onset_groups_by_voice_lane=onset_groups_by_voice_lane,
            ),
            bar_durations_by_index=bar_durations_by_index,
            note_events_by_onset=note_events_by_onset,
            track=track,
            raw_note_id_lookup=raw_note_id_lookup,
            ambiguous_raw_note_ids=ambiguous_raw_note_ids,
            pending_relations=pending_relations,
            diagnostics=diagnostics,
        )

    return (
        raw_note_id_lookup,
        ambiguous_raw_note_ids,
        tuple(sorted(pending_relations, key=lambda item: item.sort_key())),
        diagnostics,
    )


def _collect_note_relation_material_for_track(  # noqa: PLR0913
    *,
    context: _NoteTrackBuildContext,
    bar_durations_by_index: dict[int, ScoreTime],
    note_events_by_onset: dict[str, list[NoteEvent]],
    track: object,
    raw_note_id_lookup: dict[int, str],
    ambiguous_raw_note_ids: set[int],
    pending_relations: list[_PendingNoteRelation],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    if not isinstance(track, Mapping):
        diagnostics.append(
            _error(
                path=context.track_path,
                code="invalid_canonical_field",
                message="tracks entries must be objects.",
            )
        )
        return

    track_identity = track.get("id")
    if not isinstance(track_identity, int | str):
        diagnostics.append(
            _error(
                path=f"{context.track_path}.id",
                code="invalid_canonical_field",
                message="tracks entries must include an integer or string id.",
            )
        )
        return

    staves = _require_list_field(
        track,
        field_name="staves",
        path=f"{context.track_path}.staves",
        diagnostics=diagnostics,
    )
    if staves is None:
        return

    part_identifier = build_part_id(track_identity)
    for staff_ordinal, raw_staff in enumerate(staves):
        _collect_note_relation_material_for_staff(
            context=_NoteStaffBuildContext(
                part_identifier=part_identifier,
                staff_path=f"{context.track_path}.staves[{staff_ordinal}]",
                bar_times={},
                voice_lanes_by_id=context.voice_lanes_by_id,
                onset_groups_by_voice_lane=context.onset_groups_by_voice_lane,
            ),
            bar_durations_by_index=bar_durations_by_index,
            note_events_by_onset=note_events_by_onset,
            raw_staff=raw_staff,
            raw_note_id_lookup=raw_note_id_lookup,
            ambiguous_raw_note_ids=ambiguous_raw_note_ids,
            pending_relations=pending_relations,
            diagnostics=diagnostics,
        )


def _collect_note_relation_material_for_staff(  # noqa: PLR0913
    *,
    context: _NoteStaffBuildContext,
    bar_durations_by_index: dict[int, ScoreTime],
    note_events_by_onset: dict[str, list[NoteEvent]],
    raw_staff: object,
    raw_note_id_lookup: dict[int, str],
    ambiguous_raw_note_ids: set[int],
    pending_relations: list[_PendingNoteRelation],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    if not isinstance(raw_staff, Mapping):
        diagnostics.append(
            _error(
                path=context.staff_path,
                code="invalid_canonical_field",
                message="staves entries must be objects.",
            )
        )
        return

    staff_index = _coerce_required_int(
        raw_staff.get("staffIndex"),
        path=f"{context.staff_path}.staffIndex",
        diagnostics=diagnostics,
    )
    if staff_index is None:
        return

    measures = _require_list_field(
        raw_staff,
        field_name="measures",
        path=f"{context.staff_path}.measures",
        diagnostics=diagnostics,
    )
    if measures is None:
        return

    staff_identifier = build_staff_id(context.part_identifier, staff_index)
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

        bar_duration = bar_durations_by_index.get(bar_index)
        if bar_duration is None:
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
            _collect_note_relation_material_for_voice(
                context=_NoteVoiceTraversalContext(
                    staff_identifier=staff_identifier,
                    bar_index=bar_index,
                    bar_duration=bar_duration,
                    voice_path=f"{measure_path}.voices[{voice_ordinal}]",
                    voice_lanes_by_id=context.voice_lanes_by_id,
                    onset_groups_by_voice_lane=context.onset_groups_by_voice_lane,
                ),
                note_events_by_onset=note_events_by_onset,
                voice=voice,
                seen_voice_indexes=seen_voice_indexes,
                raw_note_id_lookup=raw_note_id_lookup,
                ambiguous_raw_note_ids=ambiguous_raw_note_ids,
                pending_relations=pending_relations,
                diagnostics=diagnostics,
            )


def _collect_note_relation_material_for_voice(  # noqa: PLR0913
    *,
    context: _NoteVoiceTraversalContext,
    note_events_by_onset: dict[str, list[NoteEvent]],
    voice: object,
    seen_voice_indexes: set[int],
    raw_note_id_lookup: dict[int, str],
    ambiguous_raw_note_ids: set[int],
    pending_relations: list[_PendingNoteRelation],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    resolved_inputs = _resolve_note_voice_inputs(
        context=context,
        voice=voice,
        seen_voice_indexes=seen_voice_indexes,
        diagnostics=diagnostics,
    )
    if resolved_inputs is None:
        return

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

        expected_note_events = note_events_by_onset.get(onset_group.onset_id, [])
        _collect_note_relation_material_for_beat(
            voice_lane=voice_lane,
            onset_group=onset_group,
            notes=notes,
            beat_path=beat_path,
            expected_note_events=expected_note_events,
            raw_note_id_lookup=raw_note_id_lookup,
            ambiguous_raw_note_ids=ambiguous_raw_note_ids,
            pending_relations=pending_relations,
            diagnostics=diagnostics,
        )
        attack_order_in_voice += 1


def _collect_note_relation_material_for_beat(  # noqa: PLR0913
    *,
    voice_lane: VoiceLane,
    onset_group: OnsetGroup,
    notes: list[object],
    beat_path: str,
    expected_note_events: list[NoteEvent],
    raw_note_id_lookup: dict[int, str],
    ambiguous_raw_note_ids: set[int],
    pending_relations: list[_PendingNoteRelation],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    relation_seeds: list[_NoteRelationSeed] = []
    for note_index, raw_note in enumerate(notes):
        relation_seed = _coerce_note_relation_seed(
            raw_note=raw_note,
            note_path=f"{beat_path}.notes[{note_index}]",
            onset_group=onset_group,
            voice_lane=voice_lane,
            diagnostics=diagnostics,
        )
        if relation_seed is not None:
            relation_seeds.append(relation_seed)

    relation_seeds.sort(key=lambda item: item.sort_key())
    if len(relation_seeds) != len(expected_note_events):
        diagnostics.append(
            _error(
                path=beat_path,
                code="note_event_alignment_failed",
                message=(
                    "raw note ordering could not be aligned with emitted note events "
                    f"for onset '{onset_group.onset_id}'."
                ),
            )
        )
        return

    for relation_seed, note_event in zip(relation_seeds, expected_note_events):
        raw_note_id = relation_seed.source_raw_note_id
        if raw_note_id in raw_note_id_lookup:
            raw_note_id_lookup.pop(raw_note_id, None)
            ambiguous_raw_note_ids.add(raw_note_id)
            diagnostics.append(
                _warning(
                    path=f"{relation_seed.path}.id",
                    code="duplicate_raw_note_id",
                    message=(
                        f"raw note id {raw_note_id} appears more than once within "
                        "the document."
                    ),
                )
            )
        elif raw_note_id in ambiguous_raw_note_ids:
            diagnostics.append(
                _warning(
                    path=f"{relation_seed.path}.id",
                    code="duplicate_raw_note_id",
                    message=(
                        f"raw note id {raw_note_id} appears more than once within "
                        "the document."
                    ),
                )
            )
        else:
            raw_note_id_lookup[raw_note_id] = note_event.note_id

        pending_relations.extend(relation_seed.relations)


def _coerce_note_relation_seed(
    *,
    raw_note: object,
    note_path: str,
    onset_group: OnsetGroup,
    voice_lane: VoiceLane,
    diagnostics: list[IrBuildDiagnostic],
) -> _NoteRelationSeed | None:
    if not isinstance(raw_note, Mapping):
        diagnostics.append(
            _error(
                path=note_path,
                code="invalid_canonical_field",
                message="notes entries must be objects.",
            )
        )
        return None

    note_seed = _coerce_note_seed(
        raw_note=raw_note,
        note_path=note_path,
        onset_group=onset_group,
        voice_lane=voice_lane,
        diagnostics=diagnostics,
    )
    source_raw_note_id = _coerce_required_int(
        raw_note.get("id"),
        path=f"{note_path}.id",
        diagnostics=diagnostics,
    )
    if note_seed is None or source_raw_note_id is None:
        return None

    return _NoteRelationSeed(
        path=note_path,
        source_raw_note_id=source_raw_note_id,
        note_seed=note_seed,
        relations=_coerce_pending_note_relations(
            raw_note.get("articulation"),
            path=f"{note_path}.articulation",
            source_raw_note_id=source_raw_note_id,
            diagnostics=diagnostics,
        ),
    )


def _coerce_pending_note_relations(
    value: Any,
    *,
    path: str,
    source_raw_note_id: int,
    diagnostics: list[IrBuildDiagnostic],
) -> tuple[_PendingNoteRelation, ...]:
    if value is None:
        return ()

    articulation = _coerce_required_mapping(value, path, diagnostics)
    if articulation is None:
        return ()

    raw_relations = articulation.get("relations")
    if raw_relations is None:
        return ()

    if not isinstance(raw_relations, list):
        diagnostics.append(
            _error(
                path=f"{path}.relations",
                code="invalid_canonical_field",
                message="relations must be an array when present.",
            )
        )
        return ()

    pending_relations: list[_PendingNoteRelation] = []
    for relation_index, raw_relation in enumerate(raw_relations):
        relation_path = f"{path}.relations[{relation_index}]"
        if not isinstance(raw_relation, Mapping):
            diagnostics.append(
                _error(
                    path=relation_path,
                    code="invalid_canonical_field",
                    message="relations entries must be objects.",
                )
            )
            continue

        kind_value = raw_relation.get("kind")
        if not isinstance(kind_value, str) or not kind_value.strip():
            diagnostics.append(
                _error(
                    path=f"{relation_path}.kind",
                    code="invalid_canonical_field",
                    message="relations entries must include a non-empty kind.",
                )
            )
            continue

        target_raw_note_id = _coerce_required_int(
            raw_relation.get("targetNoteId"),
            path=f"{relation_path}.targetNoteId",
            diagnostics=diagnostics,
        )
        if target_raw_note_id is None:
            continue

        pending_relations.append(
            _PendingNoteRelation(
                path=relation_path,
                kind=kind_value.strip(),
                source_raw_note_id=source_raw_note_id,
                target_raw_note_id=target_raw_note_id,
            )
        )

    return tuple(pending_relations)


def _append_relation_edges(  # noqa: PLR0913
    *,
    pending_relations: tuple[_PendingNoteRelation, ...],
    raw_note_id_lookup: dict[int, str],
    ambiguous_raw_note_ids: set[int],
    known_note_ids: set[str],
    seen_edges: set[tuple[str, str, str]],
    edges: list[Edge],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    known_ids = {"note": known_note_ids}
    for pending_relation in pending_relations:
        edge_type = _coerce_note_relation_edge_type(
            pending_relation.kind,
            path=f"{pending_relation.path}.kind",
            diagnostics=diagnostics,
        )
        if edge_type is None:
            continue

        source_note_id = _resolve_emitted_note_id(
            raw_note_id=pending_relation.source_raw_note_id,
            path=f"{pending_relation.path}.sourceNoteId",
            raw_note_id_lookup=raw_note_id_lookup,
            ambiguous_raw_note_ids=ambiguous_raw_note_ids,
            diagnostics=diagnostics,
        )
        target_note_id = _resolve_emitted_note_id(
            raw_note_id=pending_relation.target_raw_note_id,
            path=f"{pending_relation.path}.targetNoteId",
            raw_note_id_lookup=raw_note_id_lookup,
            ambiguous_raw_note_ids=ambiguous_raw_note_ids,
            diagnostics=diagnostics,
        )
        if source_note_id is None or target_note_id is None:
            continue

        _append_edge(
            source_id=source_note_id,
            target_id=target_note_id,
            edge_type=edge_type,
            path=pending_relation.path,
            known_ids=known_ids,
            seen_edges=seen_edges,
            edges=edges,
            diagnostics=diagnostics,
        )


def _coerce_note_relation_edge_type(
    value: str,
    *,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> EdgeType | None:
    edge_type = NOTE_RELATION_EDGE_TYPE_MAP.get(value.strip().casefold())
    if edge_type is not None:
        return edge_type

    diagnostics.append(
        _warning(
            path=path,
            code="unsupported_note_relation_kind",
            message=f"note relation kind '{value}' is not supported in IR v1.",
        )
    )
    return None


def _resolve_emitted_note_id(
    *,
    raw_note_id: int,
    path: str,
    raw_note_id_lookup: dict[int, str],
    ambiguous_raw_note_ids: set[int],
    diagnostics: list[IrBuildDiagnostic],
) -> str | None:
    if raw_note_id in ambiguous_raw_note_ids:
        diagnostics.append(
            _warning(
                path=path,
                code="ambiguous_note_reference",
                message=(
                    f"raw note id {raw_note_id} cannot be resolved because it is "
                    "not unique within the document."
                ),
            )
        )
        return None

    note_id = raw_note_id_lookup.get(raw_note_id)
    if note_id is not None:
        return note_id

    diagnostics.append(
        _error(
            path=path,
            code="missing_note_event_reference",
            message=f"no emitted note event exists for raw note id {raw_note_id}.",
        )
    )
    return None


def _append_edge(  # noqa: PLR0913
    *,
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    path: str,
    known_ids: dict[str, set[str]],
    seen_edges: set[tuple[str, str, str]],
    edges: list[Edge],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    if not _validate_edge_endpoint(
        identifier=source_id,
        path=path,
        known_ids=known_ids,
        diagnostics=diagnostics,
    ):
        return
    if not _validate_edge_endpoint(
        identifier=target_id,
        path=path,
        known_ids=known_ids,
        diagnostics=diagnostics,
    ):
        return

    edge_key = (source_id, edge_type.value, target_id)
    if edge_key in seen_edges:
        return

    try:
        edge = Edge(source_id=source_id, target_id=target_id, edge_type=edge_type)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_edge",
                message=str(exc),
            )
        )
        return

    seen_edges.add(edge_key)
    edges.append(edge)


def _validate_edge_endpoint(
    *,
    identifier: str,
    path: str,
    known_ids: dict[str, set[str]],
    diagnostics: list[IrBuildDiagnostic],
) -> bool:
    prefix, _, _ = identifier.partition(":")
    known_identifiers = known_ids.get(prefix)
    if known_identifiers is not None and identifier in known_identifiers:
        return True

    diagnostics.append(
        _error(
            path=path,
            code=_missing_reference_code_for_identifier(identifier),
            message=f"no emitted {_entity_label_for_identifier(identifier)} exists for '{identifier}'.",
        )
    )
    return False


def _missing_reference_code_for_identifier(identifier: str) -> str:
    prefix, _, _ = identifier.partition(":")
    return {
        "part": "missing_part_reference",
        "staff": "missing_staff_reference",
        "bar": "missing_bar_reference",
        "voice": "missing_voice_lane_reference",
        "onset": "missing_onset_group_reference",
        "note": "missing_note_event_reference",
    }.get(prefix, "missing_edge_endpoint_reference")


def _entity_label_for_identifier(identifier: str) -> str:
    prefix, _, _ = identifier.partition(":")
    return {
        "part": "part",
        "staff": "staff",
        "bar": "bar",
        "voice": "voice lane",
        "onset": "onset group",
        "note": "note event",
    }.get(prefix, "edge endpoint")
