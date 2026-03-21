"""Tests for baseline quantitative evaluation metrics."""

from __future__ import annotations

import math

import torch
from torch import nn

from motifml.evaluation.metrics import (
    QuantitativeMetricTotals,
    accumulate_quantitative_metric_totals,
    build_frequency_baseline_comparison,
    evaluate_causal_language_model,
    finalize_quantitative_metrics,
)
from motifml.training.data_loading import TokenWindowBatch
from motifml.training.training_loop import compute_causal_language_model_loss

EXPECTED_ACTIVE_TOKEN_COUNT = 2
EXPECTED_FINALIZED_TOKEN_COUNT = 4
EXPECTED_TOP_K = 5


def test_accumulate_quantitative_metric_totals_matches_controlled_logits() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, -4.0],
                [0.0, 1.0, 2.0],
                [-2.0, -1.0, 3.0],
            ]
        ],
        dtype=torch.float32,
    )
    target_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    totals = accumulate_quantitative_metric_totals(
        logits,
        target_ids,
        attention_mask,
        top_k=2,
    )

    expected_loss = float(
        compute_causal_language_model_loss(
            logits,
            target_ids,
            attention_mask,
        ).item()
    )
    assert totals.token_count == EXPECTED_ACTIVE_TOKEN_COUNT
    assert math.isclose(
        totals.total_negative_log_likelihood,
        expected_loss * EXPECTED_ACTIVE_TOKEN_COUNT,
    )
    assert totals.exact_match_count == 1
    assert totals.top_k_match_count == EXPECTED_ACTIVE_TOKEN_COUNT


def test_finalize_quantitative_metrics_converts_totals_to_report_surface() -> None:
    metrics = finalize_quantitative_metrics(
        QuantitativeMetricTotals(
            token_count=EXPECTED_FINALIZED_TOKEN_COUNT,
            total_negative_log_likelihood=2.0,
            exact_match_count=3,
            top_k_match_count=EXPECTED_FINALIZED_TOKEN_COUNT,
        ),
        top_k=EXPECTED_TOP_K,
    )

    assert metrics.token_count == EXPECTED_FINALIZED_TOKEN_COUNT
    assert metrics.top_k == EXPECTED_TOP_K
    assert math.isclose(metrics.cross_entropy_loss, 0.5)
    assert math.isclose(metrics.perplexity, math.exp(0.5))
    assert math.isclose(metrics.accuracy, 0.75)
    assert math.isclose(metrics.top_k_accuracy, 1.0)


def test_evaluate_causal_language_model_aggregates_multiple_batches() -> None:
    model = _StaticBatchLogitModel(
        (
            torch.tensor(
                [
                    [
                        [3.0, 0.0, -1.0],
                        [0.0, 2.0, 1.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 5.0],
                        [4.0, 1.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
        )
    )
    batches = (
        TokenWindowBatch(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            target_ids=torch.tensor([[0, 1]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.bool),
            splits=("validation",),
            shard_ids=("global",),
            relative_paths=("fixtures/a.json",),
            document_ids=("doc-a",),
            window_indices=(0,),
            window_start_offsets=(0,),
        ),
        TokenWindowBatch(
            input_ids=torch.tensor([[2, 1]], dtype=torch.long),
            target_ids=torch.tensor([[2, 0]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.bool),
            splits=("validation",),
            shard_ids=("global",),
            relative_paths=("fixtures/b.json",),
            document_ids=("doc-b",),
            window_indices=(0,),
            window_start_offsets=(0,),
        ),
    )

    metrics = evaluate_causal_language_model(
        model,
        evaluation_batches=batches,
        device=torch.device("cpu"),
        top_k=2,
    )

    assert metrics.token_count == EXPECTED_FINALIZED_TOKEN_COUNT
    assert 0.0 <= metrics.cross_entropy_loss
    assert metrics.perplexity >= 1.0
    assert math.isclose(metrics.accuracy, 1.0)
    assert math.isclose(metrics.top_k_accuracy, 1.0)


def test_build_frequency_baseline_comparison_reports_model_and_baseline_metrics() -> (
    None
):
    comparison = build_frequency_baseline_comparison(
        training_token_sequences=((0, 1, 1, 2), (0, 1, 2)),
        evaluation_token_sequences=((0, 1, 2),),
        model_metrics={
            "token_count": 2,
            "top_k": 2,
            "cross_entropy_loss": 0.25,
            "perplexity": math.exp(0.25),
            "accuracy": 1.0,
            "top_k_accuracy": 1.0,
        },
        top_k=2,
    )

    assert comparison["baseline_metrics"]["baseline_name"] == "frequency_next_token_v1"
    assert comparison["baseline_metrics"]["top_k"] == EXPECTED_ACTIVE_TOKEN_COUNT
    assert comparison["comparison"]["metrics"]["accuracy"]["model"] == 1.0
    assert "cross_entropy_loss" in comparison["comparison"]["metrics"]


class _StaticBatchLogitModel(nn.Module):
    def __init__(self, logits_by_batch: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self._logits_by_batch = list(logits_by_batch)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids, attention_mask
        if not self._logits_by_batch:
            raise AssertionError("No static logits remain for the requested batch.")
        return self._logits_by_batch.pop(0)
