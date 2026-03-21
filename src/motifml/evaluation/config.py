"""Typed configuration objects for baseline evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible
from motifml.training.contracts import DatasetSplit


@dataclass(frozen=True, slots=True)
class QualitativeSamplingParameters:
    """Configuration surface for deterministic qualitative sample extraction."""

    samples_per_split: int = 2
    prompt_token_count: int = 32
    summary_token_limit: int = 24

    def __post_init__(self) -> None:
        _require_positive_int(self.samples_per_split, "samples_per_split")
        _require_positive_int(self.prompt_token_count, "prompt_token_count")
        _require_positive_int(self.summary_token_limit, "summary_token_limit")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the qualitative parameters into JSON-compatible form."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class EvaluationParameters:
    """Configuration surface loaded from ``params:evaluation``."""

    device: str = "cpu"
    batch_size: int = 8
    top_k: int = 5
    decode_max_tokens: int = 256
    splits: tuple[DatasetSplit, ...] = (DatasetSplit.VALIDATION, DatasetSplit.TEST)
    qualitative: QualitativeSamplingParameters | Mapping[str, Any] = field(
        default_factory=QualitativeSamplingParameters
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device", _normalize_non_empty_text(self.device, "device")
        )
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.top_k, "top_k")
        _require_positive_int(self.decode_max_tokens, "decode_max_tokens")
        normalized_splits = tuple(DatasetSplit(split) for split in self.splits)
        if not normalized_splits:
            raise ValueError("splits must contain at least one dataset split.")
        object.__setattr__(self, "splits", normalized_splits)
        object.__setattr__(
            self,
            "qualitative",
            coerce_qualitative_sampling_parameters(self.qualitative),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the evaluation parameters into JSON-compatible form."""
        return to_json_compatible(self)


def coerce_qualitative_sampling_parameters(
    value: QualitativeSamplingParameters | Mapping[str, Any],
) -> QualitativeSamplingParameters:
    """Coerce qualitative-sampling config payloads into the typed contract."""
    if isinstance(value, QualitativeSamplingParameters):
        return value
    return QualitativeSamplingParameters(
        samples_per_split=int(value.get("samples_per_split", 2)),
        prompt_token_count=int(value.get("prompt_token_count", 32)),
        summary_token_limit=int(value.get("summary_token_limit", 24)),
    )


def coerce_evaluation_parameters(
    value: EvaluationParameters | Mapping[str, Any],
) -> EvaluationParameters:
    """Coerce ``params:evaluation`` into the typed evaluation config."""
    if isinstance(value, EvaluationParameters):
        return value
    return EvaluationParameters(
        device=str(value.get("device", "cpu")),
        batch_size=int(value.get("batch_size", 8)),
        top_k=int(value.get("top_k", 5)),
        decode_max_tokens=int(value.get("decode_max_tokens", 256)),
        splits=tuple(value.get("splits", ("validation", "test"))),
        qualitative=value.get("qualitative", {}),
    )


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


__all__ = [
    "EvaluationParameters",
    "QualitativeSamplingParameters",
    "coerce_evaluation_parameters",
    "coerce_qualitative_sampling_parameters",
]
