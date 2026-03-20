"""Shard-bounded IR corpus dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError

from motifml.datasets.motif_ir_corpus_dataset import (
    MotifIrDocumentRecord,
    ir_artifact_path_for_source,
)
from motifml.ir.serialization import deserialize_document, serialize_document
from motifml.sharding import coerce_partition_index_entries


class MotifIrShardDataset(
    AbstractDataset[list[MotifIrDocumentRecord], list[MotifIrDocumentRecord]]
):
    """Load or save only the IR documents assigned to one shard."""

    def __init__(
        self,
        filepath: str,
        partition_index_filepath: str,
        shard_id: str = "__all__",
    ) -> None:
        self._filepath = Path(filepath)
        self._partition_index_filepath = Path(partition_index_filepath)
        self._shard_id = shard_id.strip() or "__all__"

    def load(self) -> list[MotifIrDocumentRecord]:
        """Load only the IR documents for the configured shard."""
        if not self._filepath.exists():
            raise DatasetError(
                f"Motif IR corpus directory does not exist: {self._filepath.as_posix()}"
            )
        relative_paths = self._relative_paths_for_shard()

        records: list[MotifIrDocumentRecord] = []
        for relative_path in relative_paths:
            artifact_path = self._filepath / ir_artifact_path_for_source(relative_path)
            if not artifact_path.is_file():
                continue

            records.append(
                MotifIrDocumentRecord(
                    relative_path=relative_path,
                    document=deserialize_document(artifact_path.read_bytes()),
                )
            )

        return records

    def save(self, data: list[MotifIrDocumentRecord]) -> None:
        """Persist only the selected shard's IR documents to the shared root."""
        self._filepath.mkdir(parents=True, exist_ok=True)
        allowed_paths = set(self._relative_paths_for_shard())

        for record in sorted(data, key=lambda item: item.relative_path.casefold()):
            if (
                self._shard_id != "__all__"
                and record.relative_path not in allowed_paths
            ):
                raise DatasetError(
                    f"Cannot save '{record.relative_path}' into shard '{self._shard_id}'."
                )

            target_path = self._filepath / ir_artifact_path_for_source(
                record.relative_path
            )
            serialized_bytes = serialize_document(record.document).encode("utf-8")
            if target_path.exists():
                if target_path.stat().st_size == len(serialized_bytes) and (
                    target_path.read_bytes() == serialized_bytes
                ):
                    continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(serialized_bytes)

    def _exists(self) -> bool:
        return self._filepath.exists() and self._partition_index_filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "partition_index_filepath": self._partition_index_filepath.as_posix(),
            "shard_id": self._shard_id,
        }

    def _relative_paths_for_shard(self) -> list[str]:
        if not self._partition_index_filepath.exists():
            raise DatasetError(
                "Shard loading requires a partition index, but it does not exist: "
                f"{self._partition_index_filepath.as_posix()}"
            )

        partition_index = coerce_partition_index_entries(
            json.loads(self._partition_index_filepath.read_text(encoding="utf-8"))
        )
        return [
            entry.relative_path
            for entry in partition_index
            if self._shard_id in {"__all__", entry.shard_id}
        ]
