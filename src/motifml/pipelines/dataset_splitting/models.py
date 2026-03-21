"""Typed configuration for deterministic dataset splitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GroupingStrategy(StrEnum):
    """Supported grouping strategies for deterministic split assignment."""

    DOCUMENT_ID = "document_id"
    RELATIVE_PATH = "relative_path"
    PARENT_DIRECTORY = "parent_directory"


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Configured split weights for train, validation, and test."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        normalized_values = {
            "train": float(self.train),
            "validation": float(self.validation),
            "test": float(self.test),
        }
        for field_name, value in normalized_values.items():
            if value < 0:
                raise ValueError(f"{field_name} ratio must be non-negative.")
            object.__setattr__(self, field_name, value)

        if self.total <= 0:
            raise ValueError("At least one split ratio must be greater than zero.")

    @property
    def total(self) -> float:
        """Return the sum of configured ratio weights."""
        return self.train + self.validation + self.test

    def normalized(self) -> SplitRatios:
        """Normalize ratio weights so their total equals one."""
        total = self.total
        return SplitRatios(
            train=self.train / total,
            validation=self.validation / total,
            test=self.test / total,
        )


@dataclass(frozen=True, slots=True)
class DataSplitParameters:
    """Configuration surface for score-level deterministic split assignment."""

    ratios: SplitRatios = field(default_factory=SplitRatios)
    hash_seed: str = "17"
    grouping_strategy: GroupingStrategy = GroupingStrategy.DOCUMENT_ID
    grouping_key_fallback: GroupingStrategy = GroupingStrategy.RELATIVE_PATH

    def __post_init__(self) -> None:
        ratios = (
            self.ratios
            if isinstance(self.ratios, SplitRatios)
            else SplitRatios(**dict(self.ratios))
        )
        object.__setattr__(self, "ratios", ratios)
        object.__setattr__(
            self, "hash_seed", _normalize_text(self.hash_seed, "hash_seed")
        )
        object.__setattr__(
            self,
            "grouping_strategy",
            GroupingStrategy(self.grouping_strategy),
        )
        object.__setattr__(
            self,
            "grouping_key_fallback",
            GroupingStrategy(self.grouping_key_fallback),
        )


def coerce_data_split_parameters(
    value: DataSplitParameters | Mapping[str, Any],
) -> DataSplitParameters:
    """Coerce Kedro-loaded split parameters into the typed config model."""
    if isinstance(value, DataSplitParameters):
        return value

    ratios_value = value.get("ratios", {})
    return DataSplitParameters(
        ratios=(
            ratios_value
            if isinstance(ratios_value, SplitRatios)
            else SplitRatios(
                train=float(ratios_value.get("train", 0.8)),
                validation=float(ratios_value.get("validation", 0.1)),
                test=float(ratios_value.get("test", 0.1)),
            )
        ),
        hash_seed=str(value.get("hash_seed", "17")),
        grouping_strategy=value.get(
            "grouping_strategy",
            GroupingStrategy.DOCUMENT_ID,
        ),
        grouping_key_fallback=value.get(
            "grouping_key_fallback",
            GroupingStrategy.RELATIVE_PATH,
        ),
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


__all__ = [
    "DataSplitParameters",
    "GroupingStrategy",
    "SplitRatios",
    "coerce_data_split_parameters",
]
