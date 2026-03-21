"""Typed configuration objects for MotifML model architectures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible


class ModelArchitecture(StrEnum):
    """Supported model architecture identifiers."""

    DECODER_ONLY_TRANSFORMER = "decoder_only_transformer"


class PositionalEncodingType(StrEnum):
    """Supported positional-encoding implementations."""

    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"


@dataclass(frozen=True, slots=True)
class DecoderOnlyTransformerParameters:
    """Configuration surface loaded from ``params:model`` for the baseline model."""

    architecture: ModelArchitecture = ModelArchitecture.DECODER_ONLY_TRANSFORMER
    embedding_dim: int = 256
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    positional_encoding: PositionalEncodingType = PositionalEncodingType.LEARNED

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", ModelArchitecture(self.architecture))
        _require_positive_int(self.embedding_dim, "embedding_dim")
        _require_positive_int(self.hidden_size, "hidden_size")
        _require_positive_int(self.num_layers, "num_layers")
        _require_positive_int(self.num_heads, "num_heads")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        object.__setattr__(
            self,
            "dropout",
            _normalize_dropout(self.dropout),
        )
        object.__setattr__(
            self,
            "positional_encoding",
            PositionalEncodingType(self.positional_encoding),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the model parameters into a JSON-compatible mapping."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class DecoderOnlyTransformerConfig:
    """Frozen runtime configuration for one concrete Transformer instance."""

    architecture: ModelArchitecture
    vocabulary_size: int
    context_length: int
    embedding_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    positional_encoding: PositionalEncodingType
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", ModelArchitecture(self.architecture))
        _require_positive_int(self.vocabulary_size, "vocabulary_size")
        _require_positive_int(self.context_length, "context_length")
        _require_positive_int(self.embedding_dim, "embedding_dim")
        _require_positive_int(self.hidden_size, "hidden_size")
        _require_positive_int(self.num_layers, "num_layers")
        _require_positive_int(self.num_heads, "num_heads")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        object.__setattr__(self, "dropout", _normalize_dropout(self.dropout))
        object.__setattr__(
            self,
            "positional_encoding",
            PositionalEncodingType(self.positional_encoding),
        )
        if self.pad_token_id is not None:
            object.__setattr__(
                self,
                "pad_token_id",
                _normalize_non_negative_int(self.pad_token_id, "pad_token_id"),
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the runtime config into a JSON-compatible mapping."""
        return to_json_compatible(self)

    @classmethod
    def from_parameters(
        cls,
        parameters: DecoderOnlyTransformerParameters | Mapping[str, Any],
        *,
        vocabulary_size: int,
        context_length: int,
        pad_token_id: int | None = None,
    ) -> DecoderOnlyTransformerConfig:
        """Build a concrete runtime config from ``params:model`` plus dataset state."""
        typed_parameters = coerce_decoder_only_transformer_parameters(parameters)
        return cls(
            architecture=typed_parameters.architecture,
            vocabulary_size=vocabulary_size,
            context_length=context_length,
            embedding_dim=typed_parameters.embedding_dim,
            hidden_size=typed_parameters.hidden_size,
            num_layers=typed_parameters.num_layers,
            num_heads=typed_parameters.num_heads,
            dropout=typed_parameters.dropout,
            positional_encoding=typed_parameters.positional_encoding,
            pad_token_id=pad_token_id,
        )


def coerce_decoder_only_transformer_parameters(
    value: DecoderOnlyTransformerParameters | Mapping[str, Any],
) -> DecoderOnlyTransformerParameters:
    """Coerce ``params:model`` into the typed baseline model parameters."""
    if isinstance(value, DecoderOnlyTransformerParameters):
        return value
    return DecoderOnlyTransformerParameters(
        architecture=ModelArchitecture(
            str(
                value.get(
                    "architecture",
                    ModelArchitecture.DECODER_ONLY_TRANSFORMER.value,
                )
            )
        ),
        embedding_dim=int(value.get("embedding_dim", 256)),
        hidden_size=int(value.get("hidden_size", 256)),
        num_layers=int(value.get("num_layers", 4)),
        num_heads=int(value.get("num_heads", 4)),
        dropout=float(value.get("dropout", 0.1)),
        positional_encoding=PositionalEncodingType(
            str(value.get("positional_encoding", PositionalEncodingType.LEARNED.value))
        ),
    )


def build_decoder_only_transformer_config(
    model_parameters: DecoderOnlyTransformerParameters | Mapping[str, Any],
    *,
    vocabulary_size: int,
    context_length: int,
    pad_token_id: int | None = None,
) -> DecoderOnlyTransformerConfig:
    """Build one runtime config for the baseline decoder-only Transformer."""
    return DecoderOnlyTransformerConfig.from_parameters(
        model_parameters,
        vocabulary_size=vocabulary_size,
        context_length=context_length,
        pad_token_id=pad_token_id,
    )


def _require_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _normalize_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _normalize_dropout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("dropout must be a numeric probability.")
    normalized = float(value)
    if normalized < 0.0 or normalized >= 1.0:
        raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0.")
    return normalized


__all__ = [
    "DecoderOnlyTransformerConfig",
    "DecoderOnlyTransformerParameters",
    "ModelArchitecture",
    "PositionalEncodingType",
    "build_decoder_only_transformer_config",
    "coerce_decoder_only_transformer_parameters",
]
