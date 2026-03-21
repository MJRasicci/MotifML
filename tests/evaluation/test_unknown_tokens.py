"""Tests for unknown-token usage reporting and guardrails."""

from __future__ import annotations

import pytest

from motifml.evaluation.unknown_tokens import (
    build_unknown_token_usage_report,
    raise_if_unknown_token_rate_exceeds,
)

EXPECTED_TOKEN_COUNT = 5
EXPECTED_UNK_TOKEN_COUNT = 3


def test_build_unknown_token_usage_report_counts_unknown_tokens_and_rate() -> None:
    report = build_unknown_token_usage_report(
        ((1, 3, 3), (2, 3)),
        unk_token=3,
        maximum_unk_rate=0.75,
    )

    assert report.token_count == EXPECTED_TOKEN_COUNT
    assert report.unk_token_count == EXPECTED_UNK_TOKEN_COUNT
    assert report.unk_rate == pytest.approx(0.6)
    assert report.passed is True


def test_raise_if_unknown_token_rate_exceeds_fails_fast_for_high_unknown_usage() -> (
    None
):
    report = build_unknown_token_usage_report(
        (("<unk>", "<unk>", "NOTE_PITCH:C4"),),
        unk_token="<unk>",
        maximum_unk_rate=0.5,
    )

    with pytest.raises(
        ValueError, match="generated samples <unk> rate exceeds threshold"
    ):
        raise_if_unknown_token_rate_exceeds(
            report,
            context="generated samples",
        )
