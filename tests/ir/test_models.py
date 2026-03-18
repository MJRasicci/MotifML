"""Unit tests for IR value models and payload types."""

from __future__ import annotations

import pytest

from motifml.ir.models import (
    Bar,
    ControlScope,
    DynamicChangeValue,
    FermataValue,
    GeneralTechniquePayload,
    GenericTechniqueFlags,
    HairpinDirection,
    HairpinValue,
    NoteEvent,
    OnsetGroup,
    OttavaValue,
    Part,
    Pitch,
    PitchStep,
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

EXPECTED_CHROMATIC_OFFSET = -2
EXPECTED_WRITTEN_MINUS_SOUNDING = 10
EXPECTED_BEATS_PER_MINUTE = 120.0
EXPECTED_GENERIC_ACCENT = 2
EXPECTED_CAPO_FRET = 2
EXPECTED_STRING_NUMBER = 1
EXPECTED_STAFF_IDS = ("staff:part:track-7:0", "staff:part:track-7:1")


def test_transposition_keeps_chromatic_and_octave_context_minimal():
    transposition = Transposition(chromatic=EXPECTED_CHROMATIC_OFFSET, octave=1)

    assert transposition.chromatic == EXPECTED_CHROMATIC_OFFSET
    assert transposition.octave == 1
    assert (
        transposition.written_minus_sounding_semitones
        == EXPECTED_WRITTEN_MINUS_SOUNDING
    )


def test_time_signature_requires_positive_values_and_exposes_bar_duration():
    time_signature = TimeSignature(numerator=6, denominator=8)

    assert time_signature.bar_duration == ScoreTime(3, 4)

    with pytest.raises(ValueError, match="numerator"):
        TimeSignature(numerator=0, denominator=4)

    with pytest.raises(ValueError, match="denominator"):
        TimeSignature(numerator=4, denominator=0)


def test_rhythm_shape_supports_enum_coercion_and_tuplet_structure():
    rhythm_shape = RhythmShape(
        base_value="Eighth",
        augmentation_dots=1,
        primary_tuplet=TupletRatio(3, 2),
    )

    assert rhythm_shape.base_value is RhythmBaseValue.EIGHTH
    assert rhythm_shape.primary_tuplet == TupletRatio(3, 2)

    with pytest.raises(ValueError, match="secondary_tuplet"):
        RhythmShape(
            base_value=RhythmBaseValue.QUARTER,
            secondary_tuplet=TupletRatio(5, 4),
        )

    with pytest.raises(ValueError, match="augmentation_dots"):
        RhythmShape(base_value=RhythmBaseValue.QUARTER, augmentation_dots=-1)


def test_pitch_validates_spelling_and_produces_a_stable_sort_key():
    pitch = Pitch(step="C", accidental=" # ", octave=4)

    assert pitch.step is PitchStep.C
    assert pitch.accidental == "#"
    assert pitch.sort_key() == (4, "C", "#")

    with pytest.raises(ValueError, match="step must be one of"):
        Pitch(step="H", octave=4)

    with pytest.raises(ValueError, match="accidental"):
        Pitch(step=PitchStep.C, accidental=" ", octave=4)


def test_control_payloads_use_strongly_typed_models_with_validation():
    tempo = TempoChangeValue(beats_per_minute=EXPECTED_BEATS_PER_MINUTE)
    dynamic = DynamicChangeValue(marking=" mf ")
    fermata = FermataValue(fermata_type="upright", length_scale=1.5)
    hairpin = HairpinValue(direction="crescendo", niente=True)
    ottava = OttavaValue(octave_shift=1)

    assert tempo.beats_per_minute == EXPECTED_BEATS_PER_MINUTE
    assert dynamic.marking == "mf"
    assert fermata.fermata_type == "upright"
    assert hairpin.direction is HairpinDirection.CRESCENDO
    assert ottava.octave_shift == 1

    with pytest.raises(ValueError, match="beats_per_minute"):
        TempoChangeValue(beats_per_minute=0)

    with pytest.raises(ValueError, match="marking"):
        DynamicChangeValue(marking=" ")

    with pytest.raises(ValueError, match="length_scale"):
        FermataValue(length_scale=0)

    with pytest.raises(ValueError, match="direction must be one of"):
        HairpinValue(direction="sideways")

    with pytest.raises(ValueError, match="octave_shift"):
        OttavaValue(octave_shift=0)


def test_technique_payloads_remain_typed_without_using_free_form_dicts():
    techniques = TechniquePayload(
        generic=GenericTechniqueFlags(
            accent=2,
            ornament=" turn ",
            let_ring=True,
            trill=3,
        ),
        general=GeneralTechniquePayload(ornament=" mordent "),
        string_fretted=StringFrettedTechniquePayload(
            slide_types=(1, 2),
            hopo_type=1,
            harmonic_type=2,
            harmonic_kind=3,
            harmonic_fret=5.0,
            bend_enabled=True,
        ),
    )

    assert techniques.generic.accent == EXPECTED_GENERIC_ACCENT
    assert techniques.generic.ornament == "turn"
    assert techniques.general == GeneralTechniquePayload(ornament="mordent")
    assert techniques.string_fretted == StringFrettedTechniquePayload(
        slide_types=(1, 2),
        hopo_type=1,
        harmonic_type=2,
        harmonic_kind=3,
        harmonic_fret=5.0,
        bend_enabled=True,
    )

    with pytest.raises(ValueError, match="accent"):
        GenericTechniqueFlags(accent=-1)

    with pytest.raises(ValueError, match="slide_types"):
        StringFrettedTechniquePayload(slide_types=(1, -1))


def test_structure_models_validate_identity_relationships_and_optional_fields():
    part = Part(
        part_id="part:track-7",
        instrument_family=1,
        instrument_kind=2,
        role=3,
        transposition=Transposition(chromatic=0, octave=0),
        staff_ids=EXPECTED_STAFF_IDS,
    )
    staff = Staff(
        staff_id=EXPECTED_STAFF_IDS[0],
        part_id=part.part_id,
        staff_index=0,
        tuning_pitches=(64, 59, 55, 50, 45, 40),
        capo_fret=EXPECTED_CAPO_FRET,
    )
    bar = Bar(
        bar_id="bar:0",
        bar_index=0,
        start=ScoreTime(0, 1),
        duration=ScoreTime(4, 4),
        time_signature=TimeSignature(4, 4),
        key_mode=" major ",
        triplet_feel=" swing ",
    )
    voice_lane = VoiceLane(
        voice_lane_id="voice:staff:part:track-7:0:0:0",
        voice_lane_chain_id="voice-chain:part:track-7:staff:part:track-7:0:0",
        part_id=part.part_id,
        staff_id=staff.staff_id,
        bar_id=bar.bar_id,
        voice_index=0,
    )

    assert part.staff_ids == EXPECTED_STAFF_IDS
    assert staff.capo_fret == EXPECTED_CAPO_FRET
    assert bar.key_mode == "major"
    assert bar.triplet_feel == "swing"
    assert voice_lane.voice_index == 0


def test_structure_models_reject_inconsistent_ownership_and_invalid_geometry():
    with pytest.raises(ValueError, match="staff_ids must contain at least one"):
        Part(
            part_id="part:track-7",
            instrument_family=1,
            instrument_kind=2,
            role=3,
            transposition=Transposition(),
            staff_ids=(),
        )

    with pytest.raises(ValueError, match="owning part_id"):
        Staff(
            staff_id="staff:part:track-8:0",
            part_id="part:track-7",
            staff_index=0,
        )

    with pytest.raises(ValueError, match="duration must be positive"):
        Bar(
            bar_id="bar:0",
            bar_index=0,
            start=ScoreTime(0, 1),
            duration=ScoreTime(0, 1),
            time_signature=TimeSignature(4, 4),
        )

    with pytest.raises(ValueError, match="voice_lane_chain_id"):
        VoiceLane(
            voice_lane_id="voice:staff:part:track-7:0:0:0",
            voice_lane_chain_id="voice-chain:part:track-7:staff:part:track-8:0:0",
            part_id="part:track-7",
            staff_id="staff:part:track-7:0",
            bar_id="bar:0",
            voice_index=0,
        )


def test_event_models_accept_valid_onsets_notes_and_controls():
    onset = OnsetGroup(
        onset_id="onset:voice:staff:part:track-7:0:0:0:0",
        voice_lane_id="voice:staff:part:track-7:0:0:0",
        bar_id="bar:0",
        time=ScoreTime(0, 1),
        duration_notated=ScoreTime(1, 4),
        is_rest=False,
        attack_order_in_voice=0,
        duration_sounding_max=ScoreTime(1, 2),
        dynamic_local=" mf ",
        techniques=TechniquePayload(),
        rhythm_shape=RhythmShape(base_value=RhythmBaseValue.QUARTER),
    )
    note = NoteEvent(
        note_id="note:onset:voice:staff:part:track-7:0:0:0:0:0",
        onset_id=onset.onset_id,
        part_id="part:track-7",
        staff_id="staff:part:track-7:0",
        time=ScoreTime(0, 1),
        attack_duration=ScoreTime(1, 4),
        sounding_duration=ScoreTime(1, 2),
        pitch=Pitch(step=PitchStep.C, accidental="#", octave=4),
        velocity=96,
        string_number=EXPECTED_STRING_NUMBER,
        show_string_number=True,
        techniques=TechniquePayload(),
    )
    point_control = PointControlEvent(
        control_id="ctrlp:score:0",
        kind=PointControlKind.TEMPO_CHANGE,
        scope=ControlScope.SCORE,
        target_ref="score",
        time=ScoreTime(0, 1),
        value=TempoChangeValue(beats_per_minute=EXPECTED_BEATS_PER_MINUTE),
    )
    span_control = SpanControlEvent(
        control_id="ctrls:staff:0",
        kind=SpanControlKind.HAIRPIN,
        scope=ControlScope.STAFF,
        target_ref="staff:part:track-7:0",
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 2),
        value=HairpinValue(direction=HairpinDirection.CRESCENDO),
        start_anchor_ref=onset.onset_id,
        end_anchor_ref=note.note_id,
    )

    assert onset.dynamic_local == "mf"
    assert note.show_string_number is True
    assert point_control.kind is PointControlKind.TEMPO_CHANGE
    assert span_control.end_anchor_ref == note.note_id


def test_event_models_reject_invalid_constraints_and_kind_payload_mismatches():
    with pytest.raises(ValueError, match="duration_sounding_max"):
        OnsetGroup(
            onset_id="onset:voice:staff:part:track-7:0:0:0:0",
            voice_lane_id="voice:staff:part:track-7:0:0:0",
            bar_id="bar:0",
            time=ScoreTime(0, 1),
            duration_notated=ScoreTime(1, 4),
            is_rest=True,
            attack_order_in_voice=0,
            duration_sounding_max=ScoreTime(1, 4),
        )

    with pytest.raises(ValueError, match="sounding_duration"):
        NoteEvent(
            note_id="note:onset:voice:staff:part:track-7:0:0:0:0:0",
            onset_id="onset:voice:staff:part:track-7:0:0:0:0",
            part_id="part:track-7",
            staff_id="staff:part:track-7:0",
            time=ScoreTime(0, 1),
            attack_duration=ScoreTime(1, 4),
            sounding_duration=ScoreTime(0, 1),
        )

    with pytest.raises(ValueError, match="show_string_number"):
        NoteEvent(
            note_id="note:onset:voice:staff:part:track-7:0:0:0:0:0",
            onset_id="onset:voice:staff:part:track-7:0:0:0:0",
            part_id="part:track-7",
            staff_id="staff:part:track-7:0",
            time=ScoreTime(0, 1),
            attack_duration=ScoreTime(1, 4),
            sounding_duration=ScoreTime(1, 4),
            show_string_number=True,
        )

    with pytest.raises(ValueError, match="TempoChangeValue"):
        PointControlEvent(
            control_id="ctrlp:score:0",
            kind=PointControlKind.TEMPO_CHANGE,
            scope=ControlScope.SCORE,
            target_ref="score",
            time=ScoreTime(0, 1),
            value=DynamicChangeValue(marking="mf"),
        )

    with pytest.raises(ValueError, match="after start_time"):
        SpanControlEvent(
            control_id="ctrls:staff:0",
            kind=SpanControlKind.HAIRPIN,
            scope=ControlScope.STAFF,
            target_ref="staff:part:track-7:0",
            start_time=ScoreTime(1, 2),
            end_time=ScoreTime(1, 2),
            value=HairpinValue(direction=HairpinDirection.CRESCENDO),
        )
