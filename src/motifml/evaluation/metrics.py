"""Quantitative evaluation helpers for baseline causal language modeling."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from motifml.datasets.json_dataset import to_json_compatible
from motifml.model.baselines import (
    FrequencyNextTokenBaseline,
    build_baseline_comparison_report,
)
from motifml.training.data_loading import TokenWindowBatch

_LOGITS_RANK = 3


@dataclass(frozen=True, slots=True)
class QuantitativeMetricTotals:
    """Running masked totals for causal language model evaluation."""

    token_count: int = 0
    total_negative_log_likelihood: float = 0.0
    exact_match_count: int = 0
    top_k_match_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_count",
            _require_non_negative_int(self.token_count, "token_count"),
        )
        object.__setattr__(
            self,
            "total_negative_log_likelihood",
            _normalize_non_negative_float(
                self.total_negative_log_likelihood,
                "total_negative_log_likelihood",
            ),
        )
        object.__setattr__(
            self,
            "exact_match_count",
            _require_non_negative_int(self.exact_match_count, "exact_match_count"),
        )
        object.__setattr__(
            self,
            "top_k_match_count",
            _require_non_negative_int(self.top_k_match_count, "top_k_match_count"),
        )
        if self.exact_match_count > self.token_count:
            raise ValueError("exact_match_count must not exceed token_count.")
        if self.top_k_match_count > self.token_count:
            raise ValueError("top_k_match_count must not exceed token_count.")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the accumulated totals for debugging surfaces."""
        return to_json_compatible(self)

    def merge(
        self,
        other: QuantitativeMetricTotals,
    ) -> QuantitativeMetricTotals:
        """Merge two masked metric accumulators."""
        if not isinstance(other, QuantitativeMetricTotals):
            raise ValueError("other must be QuantitativeMetricTotals.")
        return QuantitativeMetricTotals(
            token_count=self.token_count + other.token_count,
            total_negative_log_likelihood=(
                self.total_negative_log_likelihood + other.total_negative_log_likelihood
            ),
            exact_match_count=self.exact_match_count + other.exact_match_count,
            top_k_match_count=self.top_k_match_count + other.top_k_match_count,
        )


@dataclass(frozen=True, slots=True)
class QuantitativeMetrics:
    """Stable quantitative report surface for one evaluated split."""

    token_count: int
    top_k: int
    cross_entropy_loss: float
    perplexity: float
    accuracy: float
    top_k_accuracy: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_count",
            _require_positive_int(self.token_count, "token_count"),
        )
        object.__setattr__(self, "top_k", _require_positive_int(self.top_k, "top_k"))
        object.__setattr__(
            self,
            "cross_entropy_loss",
            _normalize_non_negative_float(
                self.cross_entropy_loss,
                "cross_entropy_loss",
            ),
        )
        object.__setattr__(
            self,
            "perplexity",
            _normalize_non_negative_float(self.perplexity, "perplexity"),
        )
        object.__setattr__(
            self,
            "accuracy",
            _normalize_probability(self.accuracy, "accuracy"),
        )
        object.__setattr__(
            self,
            "top_k_accuracy",
            _normalize_probability(self.top_k_accuracy, "top_k_accuracy"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the quantitative metrics into JSON-compatible form."""
        return to_json_compatible(self)


def accumulate_quantitative_metric_totals(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    top_k: int,
) -> QuantitativeMetricTotals:
    """Accumulate masked loss and accuracy totals for one batch of logits."""
    normalized_top_k = _require_positive_int(top_k, "top_k")
    _validate_metric_inputs(logits, target_ids, attention_mask)

    active_positions = attention_mask.bool()
    if not bool(active_positions.any()):
        return QuantitativeMetricTotals()

    active_logits = logits[active_positions]
    active_targets = target_ids[active_positions]
    token_count = int(active_targets.numel())
    negative_log_likelihood = float(
        F.cross_entropy(active_logits, active_targets, reduction="sum").item()
    )
    exact_match_count = int(
        active_logits.argmax(dim=-1).eq(active_targets).sum().item()
    )
    effective_top_k = min(normalized_top_k, active_logits.shape[-1])
    top_k_predictions = torch.topk(
        active_logits,
        k=effective_top_k,
        dim=-1,
    ).indices
    top_k_match_count = int(
        top_k_predictions.eq(active_targets.unsqueeze(-1)).any(dim=-1).sum().item()
    )

    return QuantitativeMetricTotals(
        token_count=token_count,
        total_negative_log_likelihood=negative_log_likelihood,
        exact_match_count=exact_match_count,
        top_k_match_count=top_k_match_count,
    )


def finalize_quantitative_metrics(
    totals: QuantitativeMetricTotals,
    *,
    top_k: int,
) -> QuantitativeMetrics:
    """Finalize accumulated masked totals into stable split metrics."""
    normalized_top_k = _require_positive_int(top_k, "top_k")
    if totals.token_count <= 0:
        raise ValueError("totals must contain at least one active token.")
    average_loss = totals.total_negative_log_likelihood / totals.token_count
    return QuantitativeMetrics(
        token_count=totals.token_count,
        top_k=normalized_top_k,
        cross_entropy_loss=average_loss,
        perplexity=math.exp(average_loss),
        accuracy=totals.exact_match_count / totals.token_count,
        top_k_accuracy=totals.top_k_match_count / totals.token_count,
    )


def evaluate_causal_language_model(
    model: nn.Module,
    *,
    evaluation_batches: Iterable[TokenWindowBatch],
    device: torch.device,
    top_k: int,
) -> QuantitativeMetrics:
    """Evaluate a causal language model over one iterable of lazy token batches."""
    normalized_top_k = _require_positive_int(top_k, "top_k")
    model.eval()
    totals = QuantitativeMetricTotals()
    with torch.no_grad():
        for batch in evaluation_batches:
            logits = model(
                batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            totals = totals.merge(
                accumulate_quantitative_metric_totals(
                    logits,
                    batch.target_ids.to(device),
                    batch.attention_mask.to(device),
                    top_k=normalized_top_k,
                )
            )
    return finalize_quantitative_metrics(totals, top_k=normalized_top_k)


def build_frequency_baseline_comparison(
    *,
    training_token_sequences: Iterable[Sequence[int]],
    evaluation_token_sequences: Iterable[Sequence[int]],
    model_metrics: QuantitativeMetrics | Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Fit and score the trivial frequency baseline against one evaluation split."""
    normalized_top_k = _require_positive_int(top_k, "top_k")
    typed_model_metrics = (
        model_metrics
        if isinstance(model_metrics, QuantitativeMetrics)
        else QuantitativeMetrics(
            token_count=int(model_metrics["token_count"]),
            top_k=int(model_metrics["top_k"]),
            cross_entropy_loss=float(model_metrics["cross_entropy_loss"]),
            perplexity=float(model_metrics["perplexity"]),
            accuracy=float(model_metrics["accuracy"]),
            top_k_accuracy=float(model_metrics["top_k_accuracy"]),
        )
    )
    baseline = FrequencyNextTokenBaseline.fit(training_token_sequences)
    baseline_metrics = baseline.score_token_sequences(
        evaluation_token_sequences,
        top_k=normalized_top_k,
    )
    return {
        "baseline_metrics": baseline_metrics.to_json_dict(),
        "comparison": build_baseline_comparison_report(
            typed_model_metrics.to_json_dict(),
            baseline_metrics,
        ),
    }


def _validate_metric_inputs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if logits.ndim != _LOGITS_RANK:
        raise ValueError("logits must be a rank-3 tensor.")
    if target_ids.ndim != logits.ndim - 1:
        raise ValueError("target_ids must be a rank-2 tensor.")
    if attention_mask.ndim != logits.ndim - 1:
        raise ValueError("attention_mask must be a rank-2 tensor.")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("target_ids must match the batch and sequence logits shape.")
    if attention_mask.shape != target_ids.shape:
        raise ValueError("attention_mask must match the target_ids shape.")


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _normalize_non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric.")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return normalized


def _normalize_probability(value: Any, field_name: str) -> float:
    normalized = _normalize_non_negative_float(value, field_name)
    if normalized > 1.0:
        raise ValueError(f"{field_name} must satisfy 0.0 <= value <= 1.0.")
    return normalized


__all__ = [
    "QuantitativeMetrics",
    "QuantitativeMetricTotals",
    "accumulate_quantitative_metric_totals",
    "build_frequency_baseline_comparison",
    "evaluate_causal_language_model",
    "finalize_quantitative_metrics",
]
