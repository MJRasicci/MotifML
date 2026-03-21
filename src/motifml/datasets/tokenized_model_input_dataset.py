"""Parquet-backed Kedro dataset for tokenized ``05_model_input`` rows."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from kedro.io import AbstractDataset, DatasetError

from motifml.datasets.json_dataset import to_json_compatible
from motifml.datasets.model_input_storage import (
    MODEL_INPUT_PARAMETERS_FILENAME,
    MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
    ModelInputStorageSchema,
    coerce_model_input_storage_schema,
)
from motifml.training.contracts import DatasetSplit
from motifml.training.model_input import (
    TokenizedDocumentRow,
    coerce_tokenized_document_row,
    coerce_tokenized_document_rows,
    sort_tokenized_document_rows,
)

_MINIMUM_RECORD_PARTITION_PARTS = 3


class TokenizedModelInputDataset(AbstractDataset[Any, Any]):
    """Persist tokenized document rows as one Parquet file per source document."""

    def __init__(
        self,
        filepath: str,
        *,
        shard_id: str | None = None,
        split: str | None = None,
        storage_schema: ModelInputStorageSchema | dict[str, Any] | None = None,
    ) -> None:
        self._filepath = Path(filepath)
        self._shard_id = _normalize_optional_shard_id(shard_id)
        self._split = _normalize_optional_split(split)
        self._storage_schema = coerce_model_input_storage_schema(storage_schema)

    def load(self) -> Any:
        """Load persisted tokenized rows and shared metadata from disk."""
        if not self._filepath.exists():
            raise DatasetError(
                "Tokenized model-input directory does not exist: "
                f"{self._filepath.as_posix()}"
            )

        parameters = self._load_json(self._filepath / MODEL_INPUT_PARAMETERS_FILENAME)
        storage_schema = self._load_storage_schema()
        records = self._load_records(storage_schema)
        payload: dict[str, Any] = {
            "parameters": parameters,
            "storage_schema": storage_schema.to_json_dict(),
            "records": [record.to_row_dict() for record in records],
        }
        if self._shard_id is not None:
            payload["shard_id"] = self._shard_id
        if self._split is not None:
            payload["split"] = self._split.value
        return payload

    def save(self, data: Any) -> None:
        """Persist typed tokenized-document rows to deterministic Parquet paths."""
        if not isinstance(data, dict):
            raise DatasetError(
                "Tokenized model-input datasets must serialize to a mapping payload."
            )

        rows = coerce_tokenized_document_rows(data.get("records", ()))
        shard_id = _effective_shard_id(
            configured_shard_id=self._shard_id,
            payload_shard_id=data.get("shard_id"),
        )
        parameters = data.get("parameters")
        storage_schema = coerce_model_input_storage_schema(
            data.get("storage_schema", self._storage_schema)
        )
        self._validate_storage_schema(storage_schema)

        self._filepath.mkdir(parents=True, exist_ok=True)
        self._save_json(
            self._filepath / MODEL_INPUT_PARAMETERS_FILENAME,
            parameters,
        )
        self._save_json(
            self._filepath / MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
            storage_schema.to_json_dict(),
        )
        for row in sort_tokenized_document_rows(rows):
            if self._split is not None and row.split is not self._split:
                raise DatasetError(
                    "Row split does not match the dataset split filter: "
                    f"{row.split.value} != {self._split.value}."
                )
            self._save_row(
                row=row,
                shard_id=shard_id,
                storage_schema=storage_schema,
            )

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "shard_id": self._shard_id,
            "split": None if self._split is None else self._split.value,
            "storage_schema": self._storage_schema.to_json_dict(),
        }

    def _load_records(
        self,
        storage_schema: ModelInputStorageSchema,
    ) -> tuple[TokenizedDocumentRow, ...]:
        records_root = self._filepath / "records"
        if not records_root.exists():
            return ()

        loaded_rows: list[TokenizedDocumentRow] = []
        observed_schema: pa.Schema | None = None
        for path in sorted(records_root.rglob(f"*{storage_schema.record_suffix}")):
            split, shard_id, relative_path = _record_partition_fields(
                path,
                records_root=records_root,
                storage_schema=storage_schema,
            )
            if self._split is not None and split is not self._split:
                continue
            if self._shard_id is not None and shard_id != self._shard_id:
                continue

            table = pq.read_table(path)
            current_schema = table.schema.remove_metadata()
            if observed_schema is None:
                observed_schema = current_schema
            elif not current_schema.equals(observed_schema):
                raise DatasetError(
                    "Tokenized model-input dataset contains incompatible Parquet "
                    f"schemas, including {path.as_posix()}."
                )
            rows = table.to_pylist()
            if len(rows) != 1:
                raise DatasetError(
                    "Each tokenized model-input Parquet file must contain exactly "
                    f"one row: {path.as_posix()}."
                )

            row = coerce_tokenized_document_row(rows[0])
            if row.split is not split:
                raise DatasetError(
                    "Persisted row split does not match its partition path: "
                    f"{path.as_posix()}."
                )
            if row.relative_path != relative_path:
                raise DatasetError(
                    "Persisted row relative_path does not match its file path: "
                    f"{path.as_posix()}."
                )
            loaded_rows.append(row)

        return sort_tokenized_document_rows(loaded_rows)

    def _load_storage_schema(self) -> ModelInputStorageSchema:
        schema_path = self._filepath / MODEL_INPUT_STORAGE_SCHEMA_FILENAME
        payload = self._load_json(schema_path)
        if payload is None:
            return self._storage_schema
        storage_schema = coerce_model_input_storage_schema(payload)
        self._validate_storage_schema(storage_schema)
        return storage_schema

    def _save_row(
        self,
        *,
        row: TokenizedDocumentRow,
        shard_id: str,
        storage_schema: ModelInputStorageSchema,
    ) -> None:
        target_path = self._filepath / storage_schema.record_path(
            split=row.split.value,
            shard_id=shard_id,
            relative_path=row.relative_path,
        )
        table = pa.Table.from_pylist([row.to_row_dict()])
        if target_path.exists():
            existing_table = pq.read_table(target_path)
            if (
                existing_table.schema.remove_metadata().equals(
                    table.schema.remove_metadata()
                )
                and existing_table.to_pylist() == table.to_pylist()
            ):
                return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target_path)

    def _validate_storage_schema(
        self,
        storage_schema: ModelInputStorageSchema,
    ) -> None:
        if storage_schema != self._storage_schema:
            raise DatasetError(
                "Configured storage schema does not match the frozen dataset schema."
            )

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        if payload is None:
            return
        serializable = to_json_compatible(payload)
        serialized_text = json.dumps(
            serializable,
            indent=2,
            ensure_ascii=True,
            check_circular=False,
        )
        serialized_bytes = f"{serialized_text}\n".encode()
        if path.exists():
            if path.stat().st_size == len(serialized_bytes) and (
                path.read_bytes() == serialized_bytes
            ):
                return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialized_bytes)


def _effective_shard_id(
    *,
    configured_shard_id: str | None,
    payload_shard_id: Any,
) -> str:
    normalized_payload = _normalize_optional_shard_id(payload_shard_id)
    shard_id = configured_shard_id or normalized_payload
    if shard_id is None:
        raise DatasetError(
            "TokenizedModelInputDataset.save requires a concrete shard_id in the "
            "dataset config or payload."
        )
    return shard_id


def _record_partition_fields(
    path: Path,
    *,
    records_root: Path,
    storage_schema: ModelInputStorageSchema,
) -> tuple[DatasetSplit, str, str]:
    relative = path.relative_to(records_root)
    parts = relative.parts
    if len(parts) < _MINIMUM_RECORD_PARTITION_PARTS:
        raise DatasetError(
            "Tokenized model-input records must be partitioned by split and shard: "
            f"{path.as_posix()}."
        )
    split = DatasetSplit(parts[0])
    shard_id = _normalize_shard_id(parts[1])
    relative_path = PurePosixPath(*parts[2:]).as_posix()
    if not relative_path.endswith(storage_schema.record_suffix):
        raise DatasetError(
            "Tokenized model-input record path does not match the configured suffix: "
            f"{path.as_posix()}."
        )
    relative_path = relative_path.removesuffix(storage_schema.record_suffix)
    return split, shard_id, relative_path


def _normalize_optional_shard_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized == "__all__":
        return None
    return _normalize_shard_id(normalized)


def _normalize_shard_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise DatasetError("shard_id must be non-empty.")
    if "/" in normalized or normalized in {".", ".."}:
        raise DatasetError("shard_id must be one safe path segment.")
    return normalized


def _normalize_optional_split(value: Any) -> DatasetSplit | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized == "__all__":
        return None
    return DatasetSplit(normalized)


__all__ = ["TokenizedModelInputDataset"]
