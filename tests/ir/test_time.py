"""Unit tests for the IR rational time model."""

from __future__ import annotations

import pytest

from motifml.ir.time import ScoreTime

EXPECTED_REDUCED_NUMERATOR = 3
EXPECTED_REDUCED_DENOMINATOR = 4


def test_score_time_reduces_to_lowest_terms_on_construction():
    reduced = ScoreTime(numerator=6, denominator=8)

    assert reduced == ScoreTime(
        numerator=EXPECTED_REDUCED_NUMERATOR,
        denominator=EXPECTED_REDUCED_DENOMINATOR,
    )
    assert reduced.numerator == EXPECTED_REDUCED_NUMERATOR
    assert reduced.denominator == EXPECTED_REDUCED_DENOMINATOR


def test_score_time_can_be_built_from_integer_components_and_converted_to_float():
    time = ScoreTime.from_numerator_denominator(2, 8)

    assert time == ScoreTime.from_fraction(1, 4)
    assert time.to_float() == pytest.approx(0.25)


def test_score_time_preserves_tuplet_exactness_in_arithmetic():
    triplet_eighth = ScoreTime(numerator=1, denominator=12)

    assert triplet_eighth + triplet_eighth == ScoreTime(numerator=1, denominator=6)
    assert triplet_eighth + ScoreTime(numerator=1, denominator=6) == ScoreTime(
        numerator=1,
        denominator=4,
    )


def test_score_time_handles_dotted_durations_and_compound_meter_ordering():
    dotted_quarter = ScoreTime(numerator=3, denominator=8)
    quarter = ScoreTime(numerator=1, denominator=4)
    compound_bar = ScoreTime(numerator=6, denominator=8)

    assert dotted_quarter > quarter
    assert compound_bar == ScoreTime(numerator=3, denominator=4)
    assert dotted_quarter.to_float() == pytest.approx(0.375)
    assert sorted([dotted_quarter, quarter, compound_bar]) == [
        quarter,
        dotted_quarter,
        compound_bar,
    ]


def test_score_time_normalizes_zero_to_a_canonical_hashable_value():
    zero_a = ScoreTime(numerator=0, denominator=8)
    zero_b = ScoreTime(numerator=0, denominator=1)

    assert zero_a == zero_b
    assert zero_a.denominator == 1
    assert len({zero_a, zero_b}) == 1


def test_score_time_rejects_non_positive_denominators():
    with pytest.raises(ValueError, match="greater than zero"):
        ScoreTime(numerator=1, denominator=0)

    with pytest.raises(ValueError, match="greater than zero"):
        ScoreTime(numerator=1, denominator=-8)


def test_score_time_rejects_negative_durations_when_validated():
    negative_duration = ScoreTime(numerator=1, denominator=8) - ScoreTime(
        numerator=1,
        denominator=4,
    )

    assert negative_duration == ScoreTime(numerator=-1, denominator=8)

    with pytest.raises(ValueError, match="duration must be non-negative"):
        negative_duration.require_non_negative("duration")
