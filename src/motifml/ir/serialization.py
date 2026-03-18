"""Canonical JSON serialization helpers for MotifML IR documents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from motifml.ir.ids import (
    bar_sort_key,
    note_sort_key,
    onset_sort_key,
    part_sort_key,
    phrase_sort_key,
    point_control_sort_key,
    span_control_sort_key,
    staff_sort_key,
    voice_lane_sort_key,
)
from motifml.ir.models import (
    Bar,
    DynamicChangeValue,
    Edge,
    FermataValue,
    GeneralTechniquePayload,
    GenericTechniqueFlags,
    HairpinValue,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    OptionalOverlays,
    OptionalViews,
    OttavaValue,
    Part,
    Pitch,
    PointControlEvent,
    PointControlKind,
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


def serialize_document(document: MotifMlIrDocument) -> str:
    """Serialize an IR document into canonical, byte-stable JSON."""
    canonical_document = _canonicalize_document(document)
    payload = _serialize_value(canonical_document)
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def deserialize_document(
    payload: str | bytes | Mapping[str, Any],
) -> MotifMlIrDocument:
    """Deserialize canonical JSON content into an IR document."""
    if isinstance(payload, bytes):
        loaded = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        loaded = json.loads(payload)
    else:
        loaded = dict(payload)

    return _deserialize_document_mapping(loaded)


def _canonicalize_document(document: MotifMlIrDocument) -> MotifMlIrDocument:
    sorted_parts = tuple(
        sorted(document.parts, key=lambda part: part_sort_key(part.part_id))
    )
    sorted_staves = tuple(
        sorted(
            document.staves,
            key=lambda staff: staff_sort_key(
                staff.part_id, staff.staff_index, staff.staff_id
            ),
        )
    )
    sorted_bars = tuple(
        sorted(document.bars, key=lambda bar: bar_sort_key(bar.bar_index, bar.bar_id))
    )
    bar_index_by_id = {bar.bar_id: bar.bar_index for bar in sorted_bars}
    sorted_voice_lanes = tuple(
        sorted(
            document.voice_lanes,
            key=lambda voice_lane: voice_lane_sort_key(
                bar_index_by_id[voice_lane.bar_id],
                voice_lane.staff_id,
                voice_lane.voice_index,
                voice_lane.voice_lane_id,
            ),
        )
    )
    sorted_onsets = tuple(
        sorted(
            document.onset_groups,
            key=lambda onset: onset_sort_key(
                onset.voice_lane_id,
                onset.time,
                onset.attack_order_in_voice,
                onset.onset_id,
            ),
        )
    )
    onset_index_by_id = {
        onset.onset_id: index for index, onset in enumerate(sorted_onsets)
    }
    sorted_notes = tuple(
        sorted(
            document.note_events,
            key=lambda note: (
                onset_index_by_id[note.onset_id],
                *note_sort_key(note.string_number, note.pitch, note.note_id),
            ),
        )
    )
    sorted_point_controls = tuple(
        sorted(
            document.point_control_events,
            key=lambda control: point_control_sort_key(
                control.scope.value,
                control.target_ref,
                control.time,
                control.control_id,
            ),
        )
    )
    sorted_span_controls = tuple(
        sorted(
            document.span_control_events,
            key=lambda control: span_control_sort_key(
                control.scope.value,
                control.target_ref,
                control.start_time,
                control.end_time,
                control.control_id,
            ),
        )
    )
    sorted_edges = tuple(
        sorted(
            document.edges,
            key=lambda edge: (edge.source_id, edge.edge_type.value, edge.target_id),
        )
    )

    sorted_phrase_spans = tuple(
        sorted(
            document.optional_overlays.phrase_spans,
            key=lambda phrase_span: phrase_sort_key(
                str(phrase_span["scope_ref"]),
                _deserialize_score_time(phrase_span["start_time"]),
                _deserialize_score_time(phrase_span["end_time"]),
                str(phrase_span["phrase_id"]),
            ),
        )
        if _are_serialized_phrase_spans(document.optional_overlays.phrase_spans)
        else document.optional_overlays.phrase_spans
    )

    return MotifMlIrDocument(
        metadata=document.metadata,
        parts=sorted_parts,
        staves=sorted_staves,
        bars=sorted_bars,
        voice_lanes=sorted_voice_lanes,
        point_control_events=sorted_point_controls,
        span_control_events=sorted_span_controls,
        onset_groups=sorted_onsets,
        note_events=sorted_notes,
        edges=sorted_edges,
        optional_overlays=OptionalOverlays(phrase_spans=sorted_phrase_spans),
        optional_views=document.optional_views,
    )


def _serialize_value(value: Any) -> Any:
    if isinstance(value, ScoreTime):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
        }

    if is_dataclass(value):
        payload: dict[str, Any] = {}
        for field_info in fields(value):
            field_value = getattr(value, field_info.name)
            if field_value is None:
                continue

            payload[field_info.name] = _serialize_value(field_value)

        return payload

    if isinstance(value, tuple | list):
        return [_serialize_value(item) for item in value]

    if hasattr(value, "value"):
        return value.value

    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
            if item is not None
        }

    return value


def _deserialize_document_mapping(payload: Mapping[str, Any]) -> MotifMlIrDocument:
    return MotifMlIrDocument(
        metadata=_deserialize_metadata(payload["metadata"]),
        parts=tuple(_deserialize_part(item) for item in payload.get("parts", [])),
        staves=tuple(_deserialize_staff(item) for item in payload.get("staves", [])),
        bars=tuple(_deserialize_bar(item) for item in payload.get("bars", [])),
        voice_lanes=tuple(
            _deserialize_voice_lane(item) for item in payload.get("voice_lanes", [])
        ),
        point_control_events=tuple(
            _deserialize_point_control_event(item)
            for item in payload.get("point_control_events", [])
        ),
        span_control_events=tuple(
            _deserialize_span_control_event(item)
            for item in payload.get("span_control_events", [])
        ),
        onset_groups=tuple(
            _deserialize_onset_group(item) for item in payload.get("onset_groups", [])
        ),
        note_events=tuple(
            _deserialize_note_event(item) for item in payload.get("note_events", [])
        ),
        edges=tuple(_deserialize_edge(item) for item in payload.get("edges", [])),
        optional_overlays=_deserialize_optional_overlays(
            payload.get("optional_overlays", {})
        ),
        optional_views=_deserialize_optional_views(payload.get("optional_views", {})),
    )


def _deserialize_metadata(payload: Mapping[str, Any]) -> IrDocumentMetadata:
    return IrDocumentMetadata(
        ir_schema_version=str(payload["ir_schema_version"]),
        corpus_build_version=str(payload["corpus_build_version"]),
        generator_version=str(payload["generator_version"]),
        source_document_hash=str(payload["source_document_hash"]),
        time_unit=str(payload.get("time_unit", "whole_note_fraction")),
        compiled_resolution_hint=(
            int(payload["compiled_resolution_hint"])
            if payload.get("compiled_resolution_hint") is not None
            else None
        ),
    )


def _deserialize_part(payload: Mapping[str, Any]) -> Part:
    return Part(
        part_id=str(payload["part_id"]),
        instrument_family=int(payload["instrument_family"]),
        instrument_kind=int(payload["instrument_kind"]),
        role=int(payload["role"]),
        transposition=_deserialize_transposition(payload["transposition"]),
        staff_ids=tuple(str(item) for item in payload.get("staff_ids", [])),
    )


def _deserialize_staff(payload: Mapping[str, Any]) -> Staff:
    return Staff(
        staff_id=str(payload["staff_id"]),
        part_id=str(payload["part_id"]),
        staff_index=int(payload["staff_index"]),
        tuning_pitches=(
            tuple(int(item) for item in payload["tuning_pitches"])
            if payload.get("tuning_pitches") is not None
            else None
        ),
        capo_fret=(
            int(payload["capo_fret"]) if payload.get("capo_fret") is not None else None
        ),
    )


def _deserialize_bar(payload: Mapping[str, Any]) -> Bar:
    return Bar(
        bar_id=str(payload["bar_id"]),
        bar_index=int(payload["bar_index"]),
        start=_deserialize_score_time(payload["start"]),
        duration=_deserialize_score_time(payload["duration"]),
        time_signature=_deserialize_time_signature(payload["time_signature"]),
        key_accidental_count=(
            int(payload["key_accidental_count"])
            if payload.get("key_accidental_count") is not None
            else None
        ),
        key_mode=(
            str(payload["key_mode"]) if payload.get("key_mode") is not None else None
        ),
        triplet_feel=(
            str(payload["triplet_feel"])
            if payload.get("triplet_feel") is not None
            else None
        ),
        anacrusis_context=(
            str(payload["anacrusis_context"])
            if payload.get("anacrusis_context") is not None
            else None
        ),
    )


def _deserialize_voice_lane(payload: Mapping[str, Any]) -> VoiceLane:
    return VoiceLane(
        voice_lane_id=str(payload["voice_lane_id"]),
        voice_lane_chain_id=str(payload["voice_lane_chain_id"]),
        part_id=str(payload["part_id"]),
        staff_id=str(payload["staff_id"]),
        bar_id=str(payload["bar_id"]),
        voice_index=int(payload["voice_index"]),
    )


def _deserialize_onset_group(payload: Mapping[str, Any]) -> OnsetGroup:
    return OnsetGroup(
        onset_id=str(payload["onset_id"]),
        voice_lane_id=str(payload["voice_lane_id"]),
        bar_id=str(payload["bar_id"]),
        time=_deserialize_score_time(payload["time"]),
        duration_notated=_deserialize_score_time(payload["duration_notated"]),
        is_rest=bool(payload["is_rest"]),
        attack_order_in_voice=int(payload["attack_order_in_voice"]),
        duration_sounding_max=(
            _deserialize_score_time(payload["duration_sounding_max"])
            if payload.get("duration_sounding_max") is not None
            else None
        ),
        grace_type=(
            str(payload["grace_type"])
            if payload.get("grace_type") is not None
            else None
        ),
        dynamic_local=(
            str(payload["dynamic_local"])
            if payload.get("dynamic_local") is not None
            else None
        ),
        techniques=(
            _deserialize_technique_payload(payload["techniques"])
            if payload.get("techniques") is not None
            else None
        ),
        rhythm_shape=(
            _deserialize_rhythm_shape(payload["rhythm_shape"])
            if payload.get("rhythm_shape") is not None
            else None
        ),
    )


def _deserialize_note_event(payload: Mapping[str, Any]) -> NoteEvent:
    return NoteEvent(
        note_id=str(payload["note_id"]),
        onset_id=str(payload["onset_id"]),
        part_id=str(payload["part_id"]),
        staff_id=str(payload["staff_id"]),
        time=_deserialize_score_time(payload["time"]),
        attack_duration=_deserialize_score_time(payload["attack_duration"]),
        sounding_duration=_deserialize_score_time(payload["sounding_duration"]),
        pitch=(
            _deserialize_pitch(payload["pitch"])
            if payload.get("pitch") is not None
            else None
        ),
        velocity=(
            int(payload["velocity"]) if payload.get("velocity") is not None else None
        ),
        string_number=(
            int(payload["string_number"])
            if payload.get("string_number") is not None
            else None
        ),
        show_string_number=(
            bool(payload["show_string_number"])
            if payload.get("show_string_number") is not None
            else None
        ),
        techniques=(
            _deserialize_technique_payload(payload["techniques"])
            if payload.get("techniques") is not None
            else None
        ),
    )


def _deserialize_point_control_event(payload: Mapping[str, Any]) -> PointControlEvent:
    kind = PointControlKind(str(payload["kind"]))
    return PointControlEvent(
        control_id=str(payload["control_id"]),
        kind=kind,
        scope=str(payload["scope"]),
        target_ref=str(payload["target_ref"]),
        time=_deserialize_score_time(payload["time"]),
        value=_deserialize_point_control_value(kind, payload["value"]),
    )


def _deserialize_span_control_event(payload: Mapping[str, Any]) -> SpanControlEvent:
    kind = SpanControlKind(str(payload["kind"]))
    return SpanControlEvent(
        control_id=str(payload["control_id"]),
        kind=kind,
        scope=str(payload["scope"]),
        target_ref=str(payload["target_ref"]),
        start_time=_deserialize_score_time(payload["start_time"]),
        end_time=_deserialize_score_time(payload["end_time"]),
        value=_deserialize_span_control_value(kind, payload["value"]),
        start_anchor_ref=(
            str(payload["start_anchor_ref"])
            if payload.get("start_anchor_ref") is not None
            else None
        ),
        end_anchor_ref=(
            str(payload["end_anchor_ref"])
            if payload.get("end_anchor_ref") is not None
            else None
        ),
    )


def _deserialize_edge(payload: Mapping[str, Any]) -> Edge:
    return Edge(
        source_id=str(payload["source_id"]),
        target_id=str(payload["target_id"]),
        edge_type=str(payload["edge_type"]),
    )


def _deserialize_optional_overlays(payload: Mapping[str, Any]) -> OptionalOverlays:
    return OptionalOverlays(phrase_spans=tuple(payload.get("phrase_spans", [])))


def _deserialize_optional_views(payload: Mapping[str, Any]) -> OptionalViews:
    return OptionalViews(
        playback_instances=tuple(payload.get("playback_instances", [])),
        derived_edge_sets=tuple(payload.get("derived_edge_sets", [])),
    )


def _deserialize_transposition(payload: Mapping[str, Any]) -> Transposition:
    return Transposition(
        chromatic=int(payload["chromatic"]),
        octave=int(payload["octave"]),
    )


def _deserialize_time_signature(payload: Mapping[str, Any]) -> TimeSignature:
    return TimeSignature(
        numerator=int(payload["numerator"]),
        denominator=int(payload["denominator"]),
    )


def _deserialize_tuplet_ratio(payload: Mapping[str, Any]) -> TupletRatio:
    return TupletRatio(
        numerator=int(payload["numerator"]),
        denominator=int(payload["denominator"]),
    )


def _deserialize_rhythm_shape(payload: Mapping[str, Any]) -> RhythmShape:
    return RhythmShape(
        base_value=str(payload["base_value"]),
        augmentation_dots=int(payload.get("augmentation_dots", 0)),
        primary_tuplet=(
            _deserialize_tuplet_ratio(payload["primary_tuplet"])
            if payload.get("primary_tuplet") is not None
            else None
        ),
        secondary_tuplet=(
            _deserialize_tuplet_ratio(payload["secondary_tuplet"])
            if payload.get("secondary_tuplet") is not None
            else None
        ),
    )


def _deserialize_pitch(payload: Mapping[str, Any]) -> Pitch:
    return Pitch(
        step=str(payload["step"]),
        accidental=(
            str(payload["accidental"])
            if payload.get("accidental") is not None
            else None
        ),
        octave=int(payload["octave"]),
    )


def _deserialize_technique_payload(payload: Mapping[str, Any]) -> TechniquePayload:
    return TechniquePayload(
        generic=_deserialize_generic_technique_flags(payload.get("generic", {})),
        general=(
            _deserialize_general_technique_payload(payload["general"])
            if payload.get("general") is not None
            else None
        ),
        string_fretted=(
            _deserialize_string_fretted_technique_payload(payload["string_fretted"])
            if payload.get("string_fretted") is not None
            else None
        ),
    )


def _deserialize_generic_technique_flags(
    payload: Mapping[str, Any],
) -> GenericTechniqueFlags:
    return GenericTechniqueFlags(
        tie_origin=bool(payload.get("tie_origin", False)),
        tie_destination=bool(payload.get("tie_destination", False)),
        legato_origin=bool(payload.get("legato_origin", False)),
        legato_destination=bool(payload.get("legato_destination", False)),
        accent=int(payload["accent"]) if payload.get("accent") is not None else None,
        ornament=(
            str(payload["ornament"]) if payload.get("ornament") is not None else None
        ),
        vibrato=str(payload["vibrato"]) if payload.get("vibrato") is not None else None,
        let_ring=bool(payload.get("let_ring", False)),
        muted=bool(payload.get("muted", False)),
        palm_muted=bool(payload.get("palm_muted", False)),
        trill=int(payload["trill"]) if payload.get("trill") is not None else None,
    )


def _deserialize_general_technique_payload(
    payload: Mapping[str, Any],
) -> GeneralTechniquePayload:
    return GeneralTechniquePayload(
        ornament=(
            str(payload["ornament"]) if payload.get("ornament") is not None else None
        )
    )


def _deserialize_string_fretted_technique_payload(
    payload: Mapping[str, Any],
) -> StringFrettedTechniquePayload:
    return StringFrettedTechniquePayload(
        slide_types=tuple(int(item) for item in payload.get("slide_types", [])),
        hopo_type=(
            int(payload["hopo_type"]) if payload.get("hopo_type") is not None else None
        ),
        tapped=bool(payload.get("tapped", False)),
        left_hand_tapped=bool(payload.get("left_hand_tapped", False)),
        harmonic_type=(
            int(payload["harmonic_type"])
            if payload.get("harmonic_type") is not None
            else None
        ),
        harmonic_kind=(
            int(payload["harmonic_kind"])
            if payload.get("harmonic_kind") is not None
            else None
        ),
        harmonic_fret=(
            float(payload["harmonic_fret"])
            if payload.get("harmonic_fret") is not None
            else None
        ),
        bend_enabled=bool(payload.get("bend_enabled", False)),
        whammy_enabled=bool(payload.get("whammy_enabled", False)),
    )


def _deserialize_point_control_value(
    kind: PointControlKind, payload: Mapping[str, Any]
) -> TempoChangeValue | DynamicChangeValue | FermataValue:
    if kind is PointControlKind.TEMPO_CHANGE:
        return TempoChangeValue(beats_per_minute=float(payload["beats_per_minute"]))

    if kind is PointControlKind.DYNAMIC_CHANGE:
        return DynamicChangeValue(marking=str(payload["marking"]))

    return FermataValue(
        fermata_type=(
            str(payload["fermata_type"])
            if payload.get("fermata_type") is not None
            else None
        ),
        length_scale=(
            float(payload["length_scale"])
            if payload.get("length_scale") is not None
            else None
        ),
    )


def _deserialize_span_control_value(
    kind: SpanControlKind, payload: Mapping[str, Any]
) -> HairpinValue | OttavaValue:
    if kind is SpanControlKind.HAIRPIN:
        return HairpinValue(
            direction=str(payload["direction"]),
            niente=bool(payload.get("niente", False)),
        )

    return OttavaValue(octave_shift=int(payload["octave_shift"]))


def _deserialize_score_time(payload: Mapping[str, Any]) -> ScoreTime:
    return ScoreTime(
        numerator=int(payload["numerator"]),
        denominator=int(payload["denominator"]),
    )


def _are_serialized_phrase_spans(phrase_spans: tuple[object, ...]) -> bool:
    return all(
        isinstance(item, Mapping)
        and {"scope_ref", "start_time", "end_time", "phrase_id"} <= set(item.keys())
        for item in phrase_spans
    )
