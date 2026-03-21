"""Simple next-token baselines for training sanity checks and report comparison."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible

_HIGHER_IS_BETTER_METRICS = frozenset({"accuracy", "top_k_accuracy"})
_LOWER_IS_BETTER_METRICS = frozenset({"cross_entropy_loss", "perplexity"})


@dataclass(frozen=True, slots=True)
class FrequencyNextTokenBaselineMetrics:
    """Stable report surface for one trivial next-token baseline evaluation."""

    baseline_name: str
    token_count: int
    top_k: int
    cross_entropy_loss: float
    perplexity: float
    accuracy: float
    top_k_accuracy: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_name",
            _normalize_non_empty_text(self.baseline_name, "baseline_name"),
        )
        _require_positive_int(self.token_count, "token_count")
        _require_positive_int(self.top_k, "top_k")
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
        """Serialize the baseline metrics into a JSON-compatible mapping."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FrequencyNextTokenBaseline:
    """Conditional frequency baseline for next-token prediction."""

    transition_counts: dict[int, dict[int, int]]
    fallback_target_counts: dict[int, int]
    vocabulary_size: int
    baseline_name: str = "frequency_next_token_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transition_counts",
            {
                _normalize_non_negative_int(token_id, "transition token id"): {
                    _normalize_non_negative_int(next_token_id, "next token id"): (
                        _require_positive_int(count, "transition count")
                    )
                    for next_token_id, count in sorted(targets.items())
                }
                for token_id, targets in sorted(self.transition_counts.items())
            },
        )
        object.__setattr__(
            self,
            "fallback_target_counts",
            {
                _normalize_non_negative_int(token_id, "fallback token id"): (
                    _require_positive_int(count, "fallback target count")
                )
                for token_id, count in sorted(self.fallback_target_counts.items())
            },
        )
        _require_positive_int(self.vocabulary_size, "vocabulary_size")
        object.__setattr__(
            self,
            "baseline_name",
            _normalize_non_empty_text(self.baseline_name, "baseline_name"),
        )

    @classmethod
    def fit(
        cls,
        token_sequences: Iterable[Sequence[int]],
        *,
        baseline_name: str = "frequency_next_token_v1",
    ) -> FrequencyNextTokenBaseline:
        """Fit the conditional frequency baseline from token-id sequences."""
        transition_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        fallback_counts: Counter[int] = Counter()
        observed_vocabulary: set[int] = set()
        observed_pairs = 0

        for sequence in token_sequences:
            normalized_sequence = _normalize_token_sequence(sequence)
            observed_vocabulary.update(normalized_sequence)
            for current_token, next_token in zip(
                normalized_sequence,
                normalized_sequence[1:],
                strict=False,
            ):
                transition_counts[current_token][next_token] += 1
                fallback_counts[next_token] += 1
                observed_pairs += 1

        if observed_pairs == 0:
            raise ValueError(
                "FrequencyNextTokenBaseline.fit requires at least one next-token pair."
            )

        return cls(
            transition_counts={
                token_id: dict(sorted(counter.items()))
                for token_id, counter in sorted(transition_counts.items())
            },
            fallback_target_counts=dict(sorted(fallback_counts.items())),
            vocabulary_size=len(observed_vocabulary),
            baseline_name=baseline_name,
        )

    def score_token_sequences(
        self,
        token_sequences: Iterable[Sequence[int]],
        *,
        top_k: int = 5,
    ) -> FrequencyNextTokenBaselineMetrics:
        """Evaluate the fitted baseline over one iterable of token-id sequences."""
        normalized_top_k = _require_positive_int(top_k, "top_k")
        token_count = 0
        exact_hits = 0
        top_k_hits = 0
        negative_log_likelihood = 0.0

        for sequence in token_sequences:
            normalized_sequence = _normalize_token_sequence(sequence)
            for current_token, next_token in zip(
                normalized_sequence,
                normalized_sequence[1:],
                strict=False,
            ):
                probabilities = self._smoothed_next_token_probabilities(current_token)
                ranked_tokens = tuple(
                    token_id
                    for token_id, _ in sorted(
                        probabilities.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                )
                predicted_token = ranked_tokens[0]
                if predicted_token == next_token:
                    exact_hits += 1
                if next_token in ranked_tokens[:normalized_top_k]:
                    top_k_hits += 1
                negative_log_likelihood += -math.log(
                    self._smoothed_target_probability(current_token, next_token)
                )
                token_count += 1

        if token_count <= 0:
            raise ValueError(
                "score_token_sequences requires at least one next-token target."
            )

        average_loss = negative_log_likelihood / token_count
        return FrequencyNextTokenBaselineMetrics(
            baseline_name=self.baseline_name,
            token_count=token_count,
            top_k=normalized_top_k,
            cross_entropy_loss=average_loss,
            perplexity=math.exp(average_loss),
            accuracy=exact_hits / token_count,
            top_k_accuracy=top_k_hits / token_count,
        )

    def _smoothed_next_token_probabilities(
        self,
        current_token: int,
    ) -> dict[int, float]:
        counts = self.transition_counts.get(current_token, self.fallback_target_counts)
        total_count = sum(counts.values())
        denominator = total_count + self.vocabulary_size + 1
        return {
            token_id: (counts.get(token_id, 0) + 1) / denominator
            for token_id in self.fallback_target_counts
        }

    def _smoothed_target_probability(
        self,
        current_token: int,
        next_token: int,
    ) -> float:
        counts = self.transition_counts.get(current_token, self.fallback_target_counts)
        total_count = sum(counts.values())
        denominator = total_count + self.vocabulary_size + 1
        return (counts.get(next_token, 0) + 1) / denominator


def build_baseline_comparison_report(
    model_metrics: Mapping[str, float],
    baseline_metrics: FrequencyNextTokenBaselineMetrics,
) -> dict[str, Any]:
    """Build a stable comparison payload for report surfaces."""
    comparison_metrics: dict[str, Any] = {}
    for metric_name in (
        "cross_entropy_loss",
        "perplexity",
        "accuracy",
        "top_k_accuracy",
    ):
        if metric_name not in model_metrics:
            continue
        model_value = float(model_metrics[metric_name])
        baseline_value = float(getattr(baseline_metrics, metric_name))
        higher_is_better = metric_name in _HIGHER_IS_BETTER_METRICS
        if metric_name in _LOWER_IS_BETTER_METRICS:
            delta = baseline_value - model_value
        else:
            delta = model_value - baseline_value
        comparison_metrics[metric_name] = {
            "model": model_value,
            "baseline": baseline_value,
            "delta": delta,
            "higher_is_better": higher_is_better,
            "improved": delta > 0.0,
        }

    return {
        "baseline_name": baseline_metrics.baseline_name,
        "token_count": baseline_metrics.token_count,
        "top_k": baseline_metrics.top_k,
        "metrics": comparison_metrics,
    }


def _normalize_token_sequence(sequence: Sequence[int]) -> tuple[int, ...]:
    if isinstance(sequence, str | bytes):
        raise ValueError("token_sequences must contain token-id sequences.")
    return tuple(
        _normalize_non_negative_int(token_id, "token id") for token_id in sequence
    )


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_non_negative_int(value: Any, field_name: str) -> int:
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
    "FrequencyNextTokenBaseline",
    "FrequencyNextTokenBaselineMetrics",
    "build_baseline_comparison_report",
]
