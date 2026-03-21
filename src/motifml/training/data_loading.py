"""Lazy training-time loaders over Parquet-backed ``05_model_input`` rows."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from kedro.io import DatasetError
from torch.utils.data import IterableDataset

from motifml.datasets.model_input_storage import (
    MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
    ModelInputStorageSchema,
    coerce_model_input_storage_schema,
)
from motifml.training.contracts import DatasetSplit
from motifml.training.model_input import (
    TokenizedDocumentRow,
    coerce_tokenized_document_row,
)


@dataclass(frozen=True, slots=True)
class LoadedTokenizedDocument:
    """One lazily loaded tokenized document plus its persisted shard metadata."""

    shard_id: str
    record_path: str
    row: TokenizedDocumentRow


def load_tokenized_document_row_file(
    path: str | Path,
) -> TokenizedDocumentRow:
    """Load one persisted Parquet row from the tokenized model-input dataset."""
    record_path = Path(path)
    table = pq.read_table(record_path)
    rows = table.to_pylist()
    if len(rows) != 1:
        raise DatasetError(
            "Each tokenized model-input Parquet file must contain exactly one row: "
            f"{record_path.as_posix()}."
        )
    return coerce_tokenized_document_row(rows[0])


def discover_model_input_shards(
    dataset_root: str | Path,
    *,
    split: DatasetSplit | str,
) -> tuple[str, ...]:
    """Return the available shard ids for one split in deterministic order."""
    normalized_split = DatasetSplit(split)
    split_root = Path(dataset_root) / "records" / normalized_split.value
    if not split_root.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in split_root.iterdir()
            if path.is_dir() and path.name not in {".", ".."}
        )
    )


class LazyTokenizedDocumentDataset(IterableDataset[LoadedTokenizedDocument]):
    """Iterate tokenized document rows lazily by split and shard.

    The dataset only discovers shard directories up front. Each Parquet row is loaded
    on demand as the iterator advances, so callers do not need to materialize the full
    ``05_model_input`` corpus before training.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: DatasetSplit | str,
        shard_ids: Sequence[str] | None = None,
        row_loader: Callable[[str | Path], TokenizedDocumentRow] | None = None,
        storage_schema: ModelInputStorageSchema | dict[str, Any] | None = None,
    ) -> None:
        self._dataset_root = Path(dataset_root)
        self._split = DatasetSplit(split)
        self._shard_ids = (
            tuple(_normalize_shard_id(shard_id) for shard_id in shard_ids)
            if shard_ids is not None
            else None
        )
        self._row_loader = (
            load_tokenized_document_row_file if row_loader is None else row_loader
        )
        self._storage_schema = self._load_storage_schema(storage_schema)

    def __iter__(self) -> Iterator[LoadedTokenizedDocument]:
        """Yield tokenized document rows lazily in deterministic split/shard order."""
        for shard_id in self.shard_ids:
            for record_path in self._iter_shard_record_paths(shard_id):
                row = self._row_loader(record_path)
                if row.split is not self._split:
                    raise DatasetError(
                        "Persisted row split does not match the requested split: "
                        f"{record_path.as_posix()}."
                    )
                yield LoadedTokenizedDocument(
                    shard_id=shard_id,
                    record_path=record_path.relative_to(self._dataset_root).as_posix(),
                    row=row,
                )

    @property
    def split(self) -> DatasetSplit:
        """Return the split this lazy dataset is scoped to."""
        return self._split

    @property
    def shard_ids(self) -> tuple[str, ...]:
        """Return the deterministic shard order visible to this dataset."""
        if self._shard_ids is None:
            return discover_model_input_shards(self._dataset_root, split=self._split)
        return self._shard_ids

    def _iter_shard_record_paths(self, shard_id: str) -> Iterator[Path]:
        shard_root = self._dataset_root / "records" / self._split.value / shard_id
        if not shard_root.exists():
            return
        yield from sorted(shard_root.rglob(f"*{self._storage_schema.record_suffix}"))

    def _load_storage_schema(
        self,
        configured_schema: ModelInputStorageSchema | dict[str, Any] | None,
    ) -> ModelInputStorageSchema:
        schema_path = self._dataset_root / MODEL_INPUT_STORAGE_SCHEMA_FILENAME
        if schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as stream:
                persisted_schema = json.load(stream)
            return coerce_model_input_storage_schema(persisted_schema)
        return coerce_model_input_storage_schema(configured_schema)


def _normalize_shard_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("shard_ids must be non-empty.")
    if "/" in normalized or normalized in {".", ".."}:
        raise ValueError("shard_ids must be safe path segments.")
    return normalized


__all__ = [
    "LazyTokenizedDocumentDataset",
    "LoadedTokenizedDocument",
    "discover_model_input_shards",
    "load_tokenized_document_row_file",
]
