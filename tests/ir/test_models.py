"""Unit tests for IR value models and payload types."""

from __future__ import annotations

import pytest

from motifml.ir.models import (
    DynamicChangeValue,
    FermataValue,
    GeneralTechniquePayload,
    GenericTechniqueFlags,
    HairpinDirection,
    HairpinValue,
    OttavaValue,
    Pitch,
    PitchStep,
    RhythmBaseValue,
    RhythmShape,
    StringFrettedTechniquePayload,
    TechniquePayload,
    TempoChangeValue,
    TimeSignature,
    Transposition,
    TupletRatio,
)
from motifml.ir.time import ScoreTime

EXPECTED_CHROMATIC_OFFSET = -2
EXPECTED_WRITTEN_MINUS_SOUNDING = 10
EXPECTED_BEATS_PER_MINUTE = 120.0
EXPECTED_GENERIC_ACCENT = 2


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
