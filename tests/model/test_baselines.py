"""Tests for trivial next-token baseline helpers."""

from __future__ import annotations

import math

from motifml.model import (
    FrequencyNextTokenBaseline,
    FrequencyNextTokenBaselineMetrics,
    build_baseline_comparison_report,
)

EXPECTED_REPEATED_TRANSITION_TOKEN_COUNT = 2
EXPECTED_UNSEEN_PAIR_TOKEN_COUNT = 2


def test_frequency_next_token_baseline_scores_repeated_transitions() -> None:
    baseline = FrequencyNextTokenBaseline.fit(
        (
            (1, 2, 3, 2, 3),
            (4, 2, 3),
        )
    )

    metrics = baseline.score_token_sequences(((9, 2, 3),), top_k=1)

    assert metrics == FrequencyNextTokenBaselineMetrics(
        baseline_name="frequency_next_token_v1",
        token_count=EXPECTED_REPEATED_TRANSITION_TOKEN_COUNT,
        top_k=1,
        cross_entropy_loss=metrics.cross_entropy_loss,
        perplexity=metrics.perplexity,
        accuracy=1.0,
        top_k_accuracy=1.0,
    )
    assert metrics.cross_entropy_loss >= 0.0
    assert metrics.perplexity >= 1.0


def test_frequency_next_token_baseline_uses_smoothed_fallback_for_unseen_pairs() -> (
    None
):
    baseline = FrequencyNextTokenBaseline.fit(((1, 2, 3),))

    metrics = baseline.score_token_sequences(((7, 8, 9),), top_k=2)

    assert metrics.token_count == EXPECTED_UNSEEN_PAIR_TOKEN_COUNT
    assert math.isfinite(metrics.cross_entropy_loss)
    assert math.isfinite(metrics.perplexity)
    assert 0.0 <= metrics.accuracy <= 1.0


def test_build_baseline_comparison_report_returns_report_friendly_payload() -> None:
    report = build_baseline_comparison_report(
        {
            "cross_entropy_loss": 1.25,
            "perplexity": 3.5,
            "accuracy": 0.6,
            "top_k_accuracy": 0.8,
        },
        FrequencyNextTokenBaselineMetrics(
            baseline_name="frequency_next_token_v1",
            token_count=64,
            top_k=5,
            cross_entropy_loss=1.75,
            perplexity=5.0,
            accuracy=0.5,
            top_k_accuracy=0.7,
        ),
    )

    assert report == {
        "baseline_name": "frequency_next_token_v1",
        "token_count": 64,
        "top_k": 5,
        "metrics": {
            "cross_entropy_loss": {
                "model": 1.25,
                "baseline": 1.75,
                "delta": 0.5,
                "higher_is_better": False,
                "improved": True,
            },
            "perplexity": {
                "model": 3.5,
                "baseline": 5.0,
                "delta": 1.5,
                "higher_is_better": False,
                "improved": True,
            },
            "accuracy": {
                "model": 0.6,
                "baseline": 0.5,
                "delta": 0.09999999999999998,
                "higher_is_better": True,
                "improved": True,
            },
            "top_k_accuracy": {
                "model": 0.8,
                "baseline": 0.7,
                "delta": 0.10000000000000009,
                "higher_is_better": True,
                "improved": True,
            },
        },
    }
