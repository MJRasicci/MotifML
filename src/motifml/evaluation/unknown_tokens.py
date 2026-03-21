"""Unknown-token usage reporting and guardrails for baseline evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from motifml.datasets.json_dataset import to_json_compatible

TokenValue = TypeVar("TokenValue")


@dataclass(frozen=True, slots=True)
class UnknownTokenUsageReport:
    """Stable unknown-token usage summary for one evaluated token surface."""

    token_count: int
    unk_token_count: int
    unk_rate: float
    maximum_unk_rate: float
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_count",
            _require_non_negative_int(self.token_count, "token_count"),
        )
        object.__setattr__(
            self,
            "unk_token_count",
            _require_non_negative_int(self.unk_token_count, "unk_token_count"),
        )
        if self.unk_token_count > self.token_count:
            raise ValueError("unk_token_count must not exceed token_count.")
        object.__setattr__(
            self, "unk_rate", _normalize_fraction(self.unk_rate, "unk_rate")
        )
        object.__setattr__(
            self,
            "maximum_unk_rate",
            _normalize_fraction(self.maximum_unk_rate, "maximum_unk_rate"),
        )
        object.__setattr__(self, "passed", bool(self.passed))

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the report into JSON-compatible form."""
        return to_json_compatible(self)


def build_unknown_token_usage_report(
    token_sequences: Iterable[Sequence[TokenValue]],
    *,
    unk_token: TokenValue,
    maximum_unk_rate: float,
) -> UnknownTokenUsageReport:
    """Measure unknown-token usage for one iterable of token sequences."""
    normalized_maximum_unk_rate = _normalize_fraction(
        maximum_unk_rate,
        "maximum_unk_rate",
    )
    token_count = 0
    unk_token_count = 0
    for sequence in token_sequences:
        token_count += len(sequence)
        unk_token_count += sum(1 for token in sequence if token == unk_token)

    unk_rate = (unk_token_count / token_count) if token_count > 0 else 0.0
    return UnknownTokenUsageReport(
        token_count=token_count,
        unk_token_count=unk_token_count,
        unk_rate=unk_rate,
        maximum_unk_rate=normalized_maximum_unk_rate,
        passed=unk_rate <= normalized_maximum_unk_rate,
    )


def raise_if_unknown_token_rate_exceeds(
    report: UnknownTokenUsageReport,
    *,
    context: str,
) -> None:
    """Fail fast when one unknown-token report exceeds its configured threshold."""
    normalized_context = str(context).strip()
    if not normalized_context:
        raise ValueError("context must be non-empty.")
    if report.passed:
        return

    raise ValueError(
        f"{normalized_context} <unk> rate exceeds threshold: "
        f"{report.unk_rate:.4f} ({report.unk_token_count}/{report.token_count}, "
        f"max {report.maximum_unk_rate:.4f})."
    )


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _normalize_fraction(value: float, field_name: str) -> float:
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return normalized


__all__ = [
    "UnknownTokenUsageReport",
    "build_unknown_token_usage_report",
    "raise_if_unknown_token_rate_exceeds",
]
