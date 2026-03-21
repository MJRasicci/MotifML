"""Frozen storage-schema helpers for Parquet-backed tokenized model input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

MODEL_INPUT_STORAGE_BACKEND = "parquet"
MODEL_INPUT_STORAGE_SCHEMA_VERSION = "parquet-v1"
MODEL_INPUT_RECORD_SUFFIX = ".model_input.parquet"
MODEL_INPUT_PARAMETERS_FILENAME = "parameters.json"
MODEL_INPUT_STORAGE_SCHEMA_FILENAME = "storage_schema.json"
MODEL_INPUT_PARTITION_FIELDS = ("split", "shard_id")
MODEL_INPUT_PARTITION_FIELD_COUNT = 2


@dataclass(frozen=True, slots=True)
class ModelInputStorageSchema:
    """Physical layout contract for Parquet-backed tokenized document persistence."""

    backend: str = MODEL_INPUT_STORAGE_BACKEND
    storage_schema_version: str = MODEL_INPUT_STORAGE_SCHEMA_VERSION
    record_suffix: str = MODEL_INPUT_RECORD_SUFFIX
    partition_fields: tuple[str, str] = MODEL_INPUT_PARTITION_FIELDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _normalize_text(self.backend, "backend"))
        object.__setattr__(
            self,
            "storage_schema_version",
            _normalize_text(
                self.storage_schema_version,
                "storage_schema_version",
            ),
        )
        normalized_record_suffix = _normalize_text(self.record_suffix, "record_suffix")
        if not normalized_record_suffix.startswith("."):
            raise ValueError("record_suffix must start with '.'.")
        object.__setattr__(self, "record_suffix", normalized_record_suffix)
        if len(self.partition_fields) != MODEL_INPUT_PARTITION_FIELD_COUNT:
            raise ValueError("partition_fields must contain split and shard_id.")
        object.__setattr__(
            self,
            "partition_fields",
            tuple(
                _normalize_text(field_name, "partition_fields")
                for field_name in self.partition_fields
            ),
        )

    def record_path(self, *, split: str, shard_id: str, relative_path: str) -> str:
        """Build one canonical Parquet record path under the dataset root."""
        normalized_split = _normalize_segment(split, "split")
        normalized_shard_id = _normalize_segment(shard_id, "shard_id")
        normalized_relative_path = _normalize_relative_path(relative_path)
        return (
            PurePosixPath("records")
            / normalized_split
            / normalized_shard_id
            / f"{normalized_relative_path}{self.record_suffix}"
        ).as_posix()

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the storage schema for metadata persistence and inspection."""
        return {
            "backend": self.backend,
            "storage_schema_version": self.storage_schema_version,
            "record_suffix": self.record_suffix,
            "partition_fields": list(self.partition_fields),
            "parameters_filename": MODEL_INPUT_PARAMETERS_FILENAME,
            "storage_schema_filename": MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
        }


def coerce_model_input_storage_schema(
    value: ModelInputStorageSchema | Mapping[str, Any] | None,
) -> ModelInputStorageSchema:
    """Coerce model-input storage settings into the frozen Parquet schema helper."""
    if value is None:
        return ModelInputStorageSchema()
    if isinstance(value, ModelInputStorageSchema):
        return value

    return ModelInputStorageSchema(
        backend=str(value.get("backend", MODEL_INPUT_STORAGE_BACKEND)),
        storage_schema_version=str(
            value.get("storage_schema_version", value.get("schema_version", ""))
            or MODEL_INPUT_STORAGE_SCHEMA_VERSION
        ),
        record_suffix=str(value.get("record_suffix", MODEL_INPUT_RECORD_SUFFIX)),
        partition_fields=tuple(
            value.get("partition_fields", MODEL_INPUT_PARTITION_FIELDS)
        ),
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_segment(value: str, field_name: str) -> str:
    normalized = _normalize_text(value, field_name)
    if "/" in normalized:
        raise ValueError(f"{field_name} must be one path segment.")
    if normalized in {".", ".."}:
        raise ValueError(f"{field_name} must not escape the dataset root.")
    return normalized


def _normalize_relative_path(relative_path: str) -> str:
    normalized = _normalize_text(relative_path, "relative_path")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError("relative_path must not be absolute.")
    if normalized in {".", ""} or any(part == ".." for part in path.parts):
        raise ValueError("relative_path must point inside the dataset root.")
    return path.as_posix()


__all__ = [
    "MODEL_INPUT_PARAMETERS_FILENAME",
    "MODEL_INPUT_PARTITION_FIELD_COUNT",
    "MODEL_INPUT_PARTITION_FIELDS",
    "MODEL_INPUT_RECORD_SUFFIX",
    "MODEL_INPUT_STORAGE_BACKEND",
    "MODEL_INPUT_STORAGE_SCHEMA_FILENAME",
    "MODEL_INPUT_STORAGE_SCHEMA_VERSION",
    "ModelInputStorageSchema",
    "coerce_model_input_storage_schema",
]
