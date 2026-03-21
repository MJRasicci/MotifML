"""Typed metadata models for the normalization contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizationParameters:
    """Frozen configuration for the normalized-IR contract surface."""

    contract_name: str = "motifml.normalized_ir"
    contract_version: str = "1.0.0"
    serialized_document_format: str = "motifml.ir.document"
    normalization_strategy: str = "passthrough_v1"
    allow_optional_overlays: bool = True
    allow_optional_views: bool = True
    task_agnostic_guarantees: tuple[str, ...] = (
        "stable_source_relative_identity",
        "task_agnostic_domain_truth",
        "no_model_specific_flattening",
        "no_model_specific_windowing",
    )
    forbidden_model_fields: tuple[str, ...] = (
        "attention_mask",
        "input_ids",
        "model_input_version",
        "padding_strategy",
        "split",
        "split_version",
        "target_ids",
        "token_count",
        "token_ids",
        "training_run_id",
        "vocabulary_version",
        "window_start_offsets",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_name", _normalize_text(self.contract_name, "contract_name")
        )
        object.__setattr__(
            self,
            "contract_version",
            _normalize_text(self.contract_version, "contract_version"),
        )
        object.__setattr__(
            self,
            "serialized_document_format",
            _normalize_text(
                self.serialized_document_format,
                "serialized_document_format",
            ),
        )
        object.__setattr__(
            self,
            "normalization_strategy",
            _normalize_text(
                self.normalization_strategy,
                "normalization_strategy",
            ),
        )
        object.__setattr__(
            self,
            "allow_optional_overlays",
            bool(self.allow_optional_overlays),
        )
        object.__setattr__(
            self, "allow_optional_views", bool(self.allow_optional_views)
        )
        guarantees = tuple(
            _normalize_text(value, "task_agnostic_guarantees")
            for value in self.task_agnostic_guarantees
        )
        if not guarantees:
            raise ValueError(
                "task_agnostic_guarantees must contain at least one guarantee."
            )
        object.__setattr__(self, "task_agnostic_guarantees", guarantees)
        forbidden_fields = tuple(
            _normalize_text(value, "forbidden_model_fields")
            for value in self.forbidden_model_fields
        )
        if not forbidden_fields:
            raise ValueError(
                "forbidden_model_fields must contain at least one forbidden field."
            )
        object.__setattr__(self, "forbidden_model_fields", forbidden_fields)


@dataclass(frozen=True, slots=True)
class NormalizedIrVersionMetadata:
    """Persisted normalized-IR version metadata for `03_primary`."""

    normalized_ir_version: str
    contract_name: str
    contract_version: str
    serialized_document_format: str
    normalization_strategy: str
    upstream_ir_schema_version: str
    task_agnostic_guarantees: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_ir_version",
            _normalize_text(self.normalized_ir_version, "normalized_ir_version"),
        )
        object.__setattr__(
            self, "contract_name", _normalize_text(self.contract_name, "contract_name")
        )
        object.__setattr__(
            self,
            "contract_version",
            _normalize_text(self.contract_version, "contract_version"),
        )
        object.__setattr__(
            self,
            "serialized_document_format",
            _normalize_text(
                self.serialized_document_format,
                "serialized_document_format",
            ),
        )
        object.__setattr__(
            self,
            "normalization_strategy",
            _normalize_text(
                self.normalization_strategy,
                "normalization_strategy",
            ),
        )
        object.__setattr__(
            self,
            "upstream_ir_schema_version",
            _normalize_text(
                self.upstream_ir_schema_version,
                "upstream_ir_schema_version",
            ),
        )
        guarantees = tuple(
            _normalize_text(value, "task_agnostic_guarantees")
            for value in self.task_agnostic_guarantees
        )
        if not guarantees:
            raise ValueError(
                "task_agnostic_guarantees must contain at least one guarantee."
            )
        object.__setattr__(self, "task_agnostic_guarantees", guarantees)


def coerce_normalization_parameters(
    value: NormalizationParameters | Mapping[str, Any],
) -> NormalizationParameters:
    """Coerce Kedro-loaded normalization parameters into the typed contract model."""
    if isinstance(value, NormalizationParameters):
        return value

    return NormalizationParameters(
        contract_name=str(value.get("contract_name", "motifml.normalized_ir")),
        contract_version=str(value.get("contract_version", "1.0.0")),
        serialized_document_format=str(
            value.get("serialized_document_format", "motifml.ir.document")
        ),
        normalization_strategy=str(
            value.get("normalization_strategy", "passthrough_v1")
        ),
        allow_optional_overlays=bool(value.get("allow_optional_overlays", True)),
        allow_optional_views=bool(value.get("allow_optional_views", True)),
        task_agnostic_guarantees=tuple(
            value.get(
                "task_agnostic_guarantees",
                (
                    "stable_source_relative_identity",
                    "task_agnostic_domain_truth",
                    "no_model_specific_flattening",
                    "no_model_specific_windowing",
                ),
            )
        ),
        forbidden_model_fields=tuple(
            value.get(
                "forbidden_model_fields",
                (
                    "attention_mask",
                    "input_ids",
                    "model_input_version",
                    "padding_strategy",
                    "split",
                    "split_version",
                    "target_ids",
                    "token_count",
                    "token_ids",
                    "training_run_id",
                    "vocabulary_version",
                    "window_start_offsets",
                ),
            )
        ),
    )


def coerce_normalized_ir_version_metadata(
    value: NormalizedIrVersionMetadata | Mapping[str, Any],
) -> NormalizedIrVersionMetadata:
    """Coerce JSON-loaded normalized-IR version metadata into the typed model."""
    if isinstance(value, NormalizedIrVersionMetadata):
        return value

    return NormalizedIrVersionMetadata(
        normalized_ir_version=str(value["normalized_ir_version"]),
        contract_name=str(value["contract_name"]),
        contract_version=str(value["contract_version"]),
        serialized_document_format=str(value["serialized_document_format"]),
        normalization_strategy=str(value["normalization_strategy"]),
        upstream_ir_schema_version=str(value["upstream_ir_schema_version"]),
        task_agnostic_guarantees=tuple(value["task_agnostic_guarantees"]),
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


__all__ = [
    "NormalizationParameters",
    "NormalizedIrVersionMetadata",
    "coerce_normalization_parameters",
    "coerce_normalized_ir_version_metadata",
]
