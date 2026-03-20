"""Shard-bounded raw Motif JSON dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from kedro.io import AbstractDataset, DatasetError

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument, RawMotifScore
from motifml.sharding import coerce_partition_index_entries


class MotifJsonShardDataset(AbstractDataset[None, list[MotifJsonDocument]]):
    """Load only the raw Motif JSON documents assigned to one shard."""

    def __init__(
        self,
        filepath: str,
        partition_index_filepath: str,
        shard_id: str = "__all__",
    ) -> None:
        self._filepath = Path(filepath)
        self._partition_index_filepath = Path(partition_index_filepath)
        self._shard_id = shard_id.strip() or "__all__"

    def load(self) -> list[MotifJsonDocument]:
        """Load the configured shard in deterministic relative-path order."""
        if not self._filepath.exists():
            raise DatasetError(
                f"Raw Motif JSON corpus directory does not exist: {self._filepath.as_posix()}"
            )
        if not self._partition_index_filepath.exists():
            raise DatasetError(
                "Shard loading requires a partition index, but it does not exist: "
                f"{self._partition_index_filepath.as_posix()}"
            )

        partition_index = coerce_partition_index_entries(
            json.loads(self._partition_index_filepath.read_text(encoding="utf-8"))
        )
        relative_paths = [
            entry.relative_path
            for entry in partition_index
            if self._shard_id in {"__all__", entry.shard_id}
        ]

        documents: list[MotifJsonDocument] = []
        for relative_path in relative_paths:
            path = self._filepath / relative_path
            if not path.is_file():
                raise DatasetError(
                    "Partition index referenced a raw Motif JSON file that does not "
                    f"exist: {path.as_posix()}"
                )

            payload = path.read_bytes()
            loaded = json.loads(payload)
            if not isinstance(loaded, dict):
                raise DatasetError(
                    "Motif JSON shard documents must deserialize to objects, "
                    f"but '{path.as_posix()}' did not."
                )

            documents.append(
                MotifJsonDocument(
                    relative_path=relative_path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    file_size_bytes=len(payload),
                    score=cast(RawMotifScore, loaded),
                )
            )

        return documents

    def save(self, data: None) -> None:
        """Reject writes because the raw corpus is source data."""
        raise DatasetError("MotifJsonShardDataset is read-only.")

    def _exists(self) -> bool:
        return self._filepath.exists() and self._partition_index_filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "partition_index_filepath": self._partition_index_filepath.as_posix(),
            "shard_id": self._shard_id,
        }
