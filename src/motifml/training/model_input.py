"""Typed row-level contracts for tokenized ``05_model_input`` persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible
from motifml.training.contracts import DatasetSplit
from motifml.training.special_token_policy import coerce_special_token_policy

_ALLOWED_PADDING_STRATEGIES = frozenset({"none", "left", "right"})


@dataclass(frozen=True, slots=True)
class TokenizedDocumentRow:
    """One persisted tokenized-document row for the real model-input contract."""

    relative_path: str
    document_id: str
    split: DatasetSplit
    split_version: str
    projection_type: str
    sequence_mode: str
    normalized_ir_version: str
    feature_version: str
    vocabulary_version: str
    model_input_version: str
    storage_schema_version: str
    token_count: int
    token_ids: tuple[int, ...]
    window_start_offsets: tuple[int, ...]
    context_length: int
    stride: int
    padding_strategy: str
    special_token_policy: dict[str, Any]
    inspection_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relative_path", _normalize_text(self.relative_path, "relative_path")
        )
        object.__setattr__(
            self, "document_id", _normalize_text(self.document_id, "document_id")
        )
        object.__setattr__(self, "split", DatasetSplit(self.split))
        object.__setattr__(
            self, "split_version", _normalize_text(self.split_version, "split_version")
        )
        object.__setattr__(
            self,
            "projection_type",
            _normalize_text(self.projection_type, "projection_type"),
        )
        object.__setattr__(
            self, "sequence_mode", _normalize_text(self.sequence_mode, "sequence_mode")
        )
        object.__setattr__(
            self,
            "normalized_ir_version",
            _normalize_text(
                self.normalized_ir_version,
                "normalized_ir_version",
            ),
        )
        object.__setattr__(
            self,
            "feature_version",
            _normalize_text(self.feature_version, "feature_version"),
        )
        object.__setattr__(
            self,
            "vocabulary_version",
            _normalize_text(self.vocabulary_version, "vocabulary_version"),
        )
        object.__setattr__(
            self,
            "model_input_version",
            _normalize_text(self.model_input_version, "model_input_version"),
        )
        object.__setattr__(
            self,
            "storage_schema_version",
            _normalize_text(self.storage_schema_version, "storage_schema_version"),
        )
        _require_non_negative_int(self.token_count, "token_count")
        normalized_token_ids = tuple(
            _normalize_non_negative_int(token_id, field_name=f"token_ids[{index}]")
            for index, token_id in enumerate(self.token_ids)
        )
        if self.token_count != len(normalized_token_ids):
            raise ValueError("token_count must match the number of token_ids.")
        object.__setattr__(self, "token_ids", normalized_token_ids)

        normalized_offsets = tuple(
            _normalize_non_negative_int(
                offset,
                field_name=f"window_start_offsets[{index}]",
            )
            for index, offset in enumerate(self.window_start_offsets)
        )
        if normalized_offsets != tuple(sorted(normalized_offsets)):
            raise ValueError("window_start_offsets must be sorted in ascending order.")
        if len(normalized_offsets) != len(set(normalized_offsets)):
            raise ValueError("window_start_offsets must be unique.")
        if normalized_token_ids and normalized_offsets:
            max_offset = max(normalized_offsets)
            if max_offset >= len(normalized_token_ids):
                raise ValueError(
                    "window_start_offsets must point inside the token_ids sequence."
                )
        object.__setattr__(self, "window_start_offsets", normalized_offsets)

        _require_positive_int(self.context_length, "context_length")
        _require_positive_int(self.stride, "stride")
        object.__setattr__(
            self,
            "padding_strategy",
            _normalize_padding_strategy(self.padding_strategy),
        )
        object.__setattr__(
            self,
            "special_token_policy",
            coerce_special_token_policy(self.special_token_policy).to_version_payload(),
        )
        object.__setattr__(
            self,
            "inspection_metadata",
            _normalize_optional_json_mapping(
                self.inspection_metadata,
                "inspection_metadata",
            ),
        )

    def to_row_dict(self) -> dict[str, Any]:
        """Serialize the tokenized row into a Parquet-friendly Python mapping."""
        return {
            "relative_path": self.relative_path,
            "document_id": self.document_id,
            "split": self.split.value,
            "split_version": self.split_version,
            "projection_type": self.projection_type,
            "sequence_mode": self.sequence_mode,
            "normalized_ir_version": self.normalized_ir_version,
            "feature_version": self.feature_version,
            "vocabulary_version": self.vocabulary_version,
            "model_input_version": self.model_input_version,
            "storage_schema_version": self.storage_schema_version,
            "token_count": self.token_count,
            "token_ids": list(self.token_ids),
            "window_start_offsets": list(self.window_start_offsets),
            "context_length": self.context_length,
            "stride": self.stride,
            "padding_strategy": self.padding_strategy,
            "special_token_policy": dict(self.special_token_policy),
            "inspection_metadata": (
                None
                if self.inspection_metadata is None
                else to_json_compatible(self.inspection_metadata)
            ),
        }

    @classmethod
    def from_row_dict(cls, payload: Mapping[str, Any]) -> TokenizedDocumentRow:
        """Deserialize one persisted tokenized-document row."""
        return cls(
            relative_path=str(payload["relative_path"]),
            document_id=str(payload["document_id"]),
            split=DatasetSplit(str(payload["split"])),
            split_version=str(payload["split_version"]),
            projection_type=str(payload["projection_type"]),
            sequence_mode=str(payload["sequence_mode"]),
            normalized_ir_version=str(payload["normalized_ir_version"]),
            feature_version=str(payload["feature_version"]),
            vocabulary_version=str(payload["vocabulary_version"]),
            model_input_version=str(payload["model_input_version"]),
            storage_schema_version=str(payload["storage_schema_version"]),
            token_count=int(payload["token_count"]),
            token_ids=tuple(payload.get("token_ids", ())),
            window_start_offsets=tuple(payload.get("window_start_offsets", ())),
            context_length=int(payload["context_length"]),
            stride=int(payload["stride"]),
            padding_strategy=str(payload["padding_strategy"]),
            special_token_policy=_mapping_from_payload(payload, "special_token_policy"),
            inspection_metadata=_optional_mapping_from_payload(
                payload,
                "inspection_metadata",
            ),
        )


def coerce_tokenized_document_row(
    value: TokenizedDocumentRow | Mapping[str, Any],
) -> TokenizedDocumentRow:
    """Coerce one loaded payload into the typed tokenized-document row."""
    if isinstance(value, TokenizedDocumentRow):
        return value
    return TokenizedDocumentRow.from_row_dict(value)


def coerce_tokenized_document_rows(
    rows: Sequence[TokenizedDocumentRow | Mapping[str, Any]],
) -> tuple[TokenizedDocumentRow, ...]:
    """Coerce an iterable of persisted row payloads into typed row models."""
    return tuple(coerce_tokenized_document_row(row) for row in rows)


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_padding_strategy(value: str) -> str:
    normalized = _normalize_text(value, "padding_strategy").lower()
    if normalized not in _ALLOWED_PADDING_STRATEGIES:
        allowed_values = ", ".join(sorted(_ALLOWED_PADDING_STRATEGIES))
        raise ValueError("padding_strategy must be one of: " f"{allowed_values}.")
    return normalized


def _normalize_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> None:
    _normalize_non_negative_int(value, field_name=field_name)


def _require_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")


def _normalize_optional_json_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided.")
    serialized = to_json_compatible(dict(value))
    if not isinstance(serialized, dict):
        raise ValueError(f"{field_name} must serialize to a mapping.")
    return {
        str(key): _normalize_json_value(item)
        for key, item in sorted(serialized.items(), key=lambda item: str(item[0]))
    }


def _normalize_json_value(value: Any) -> Any:
    serialized = to_json_compatible(value)
    if isinstance(serialized, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(serialized.items(), key=lambda item: str(item[0]))
        }
    if isinstance(serialized, list):
        return [_normalize_json_value(item) for item in serialized]
    return serialized


def _mapping_from_payload(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return {str(item_key): item for item_key, item in value.items()}


def _optional_mapping_from_payload(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping when provided.")
    return {str(item_key): item for item_key, item in value.items()}


__all__ = [
    "TokenizedDocumentRow",
    "coerce_tokenized_document_row",
    "coerce_tokenized_document_rows",
]
