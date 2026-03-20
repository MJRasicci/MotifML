"""Typed configuration and outputs for IR tokenization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from motifml.pipelines.feature_extraction.models import ProjectionType


class PaddingStrategy(StrEnum):
    """Supported padding strategies for model-input preparation."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class TokenizationParameters:
    """Configuration surface for the tokenization pipeline."""

    vocabulary_strategy: str = "projection_native"
    max_sequence_length: int = 8
    padding_strategy: PaddingStrategy = PaddingStrategy.RIGHT
    time_resolution: int = 96

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary_strategy",
            _normalize_text(self.vocabulary_strategy, "vocabulary_strategy"),
        )
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        object.__setattr__(
            self,
            "padding_strategy",
            PaddingStrategy(self.padding_strategy),
        )
        if self.time_resolution <= 0:
            raise ValueError("time_resolution must be positive.")


@dataclass(frozen=True)
class ModelInputRecord:
    """One tokenized example derived from a projected feature record."""

    relative_path: str
    projection_type: ProjectionType
    vocabulary_strategy: str
    time_resolution: int
    original_token_count: int
    tokens: tuple[str, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "projection_type",
            ProjectionType(self.projection_type),
        )
        object.__setattr__(
            self,
            "vocabulary_strategy",
            _normalize_text(self.vocabulary_strategy, "vocabulary_strategy"),
        )
        if self.time_resolution <= 0:
            raise ValueError("time_resolution must be positive.")
        if self.original_token_count < 0:
            raise ValueError("original_token_count must be non-negative.")
        if len(self.tokens) != len(self.attention_mask):
            raise ValueError("tokens and attention_mask must have matching lengths.")


@dataclass(frozen=True)
class ModelInputSet:
    """Collection of model-input records emitted by tokenization."""

    parameters: TokenizationParameters
    records: tuple[ModelInputRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda item: item.relative_path.casefold())),
        )


def coerce_tokenization_parameters(
    value: TokenizationParameters | Mapping[str, Any],
) -> TokenizationParameters:
    """Coerce Kedro parameter mappings into the typed tokenization config."""
    if isinstance(value, TokenizationParameters):
        return value

    return TokenizationParameters(
        vocabulary_strategy=str(value.get("vocabulary_strategy", "projection_native")),
        max_sequence_length=int(value.get("max_sequence_length", 8)),
        padding_strategy=value.get("padding_strategy", PaddingStrategy.RIGHT),
        time_resolution=int(value.get("time_resolution", 96)),
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized
