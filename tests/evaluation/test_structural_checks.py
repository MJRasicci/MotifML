"""Tests for decoded-sequence structural evaluation checks."""

from __future__ import annotations

import math

from motifml.evaluation.structural_checks import (
    DecodedTokenSequence,
    evaluate_structural_quality,
)


def test_evaluate_structural_quality_accepts_well_formed_sequences() -> None:
    reference_sequences = (
        DecodedTokenSequence(
            sequence_id="reference-a",
            tokens=(
                "<bos>",
                "STRUCTURE:PART",
                "STRUCTURE:STAFF",
                "STRUCTURE:BAR",
                "STRUCTURE:VOICE_LANE",
                "STRUCTURE:ONSET_GROUP",
                "NOTE_PITCH:C4",
                "NOTE_DURATION:96",
                "TIME_SHIFT:96",
                "NOTE_PITCH:D4",
                "NOTE_DURATION:96",
                "<eos>",
            ),
        ),
    )

    report = evaluate_structural_quality(
        reference_sequences,
        reference_sequences=reference_sequences,
    )

    assert report.sequence_count == 1
    assert math.isclose(report.valid_transition_rate, 1.0)
    assert math.isclose(report.boundary_order_pass_rate, 1.0)
    assert math.isclose(report.out_of_range_pitch_fraction, 0.0)
    assert math.isclose(report.duration_distribution_total_variation, 0.0)
    assert math.isclose(report.generated_unk_rate, 0.0)
    assert report.failures == ()


def test_evaluate_structural_quality_reports_transition_boundary_and_drift_failures() -> (
    None
):
    reference_sequences = (
        DecodedTokenSequence(
            sequence_id="reference-a",
            tokens=(
                "<bos>",
                "STRUCTURE:PART",
                "STRUCTURE:STAFF",
                "STRUCTURE:BAR",
                "NOTE_PITCH:C4",
                "NOTE_DURATION:96",
                "<eos>",
            ),
        ),
    )
    generated_sequences = (
        DecodedTokenSequence(
            sequence_id="generated-a",
            tokens=(
                "<bos>",
                "NOTE_DURATION:48",
                "NOTE_PITCH:C6",
                "STRUCTURE:BAR",
                "STRUCTURE:PART",
                "<unk>",
                "<eos>",
            ),
        ),
    )

    report = evaluate_structural_quality(
        generated_sequences,
        reference_sequences=reference_sequences,
    )

    assert report.sequence_count == 1
    assert report.valid_transition_rate < 1.0
    assert report.boundary_order_pass_rate == 0.0
    assert report.generated_unk_token_count == 1
    assert math.isclose(report.generated_unk_rate, 1 / 7)
    assert math.isclose(report.out_of_range_pitch_fraction, 1.0)
    assert report.duration_distribution_total_variation == 1.0
    assert {failure.check_name for failure in report.failures} == {
        "transition",
        "boundary_order",
    }
    assert any(failure.sequence_id == "generated-a" for failure in report.failures)
