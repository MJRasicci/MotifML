"""Tests for IR-build shared helpers."""

from __future__ import annotations

from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.common import _coerce_score_time


def test_coerce_score_time_snaps_decimal_approximation_to_exact_fraction() -> None:
    diagnostics = []

    result = _coerce_score_time(
        {"numerator": 333333, "denominator": 1000000},
        path="pointControls[1].position.offset",
        diagnostics=diagnostics,
        require_positive=False,
    )

    assert result == ScoreTime(1, 3)
    assert diagnostics == []


def test_coerce_score_time_preserves_exact_musical_tuplets() -> None:
    diagnostics = []

    result = _coerce_score_time(
        {"numerator": 1, "denominator": 20},
        path="pointControls[1].position.offset",
        diagnostics=diagnostics,
        require_positive=False,
    )

    assert result == ScoreTime(1, 20)
    assert diagnostics == []


def test_coerce_score_time_does_not_snap_unrelated_large_denominators() -> None:
    diagnostics = []

    result = _coerce_score_time(
        {"numerator": 102741, "denominator": 1000000},
        path="pointControls[1].position.offset",
        diagnostics=diagnostics,
        require_positive=False,
    )

    assert result == ScoreTime(102741, 1000000)
    assert diagnostics == []
