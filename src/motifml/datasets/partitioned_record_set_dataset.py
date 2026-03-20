"""Dataset for partitioned record sets keyed by source-relative paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError

from motifml.datasets.json_dataset import to_json_compatible
from motifml.sharding import coerce_partition_index_entries, relative_paths_for_shard


class PartitionedRecordSetDataset(AbstractDataset[Any, Any]):
    """Persist a record set as one JSON file per source-relative record."""

    def __init__(
        self,
        filepath: str,
        *,
        record_suffix: str = ".record.json",
        partition_index_filepath: str | None = None,
        shard_id: str | None = None,
    ) -> None:
        self._filepath = Path(filepath)
        self._record_suffix = record_suffix
        self._partition_index_filepath = (
            Path(partition_index_filepath)
            if partition_index_filepath is not None
            else None
        )
        self._shard_id = _normalize_optional_text(shard_id)

    def load(self) -> Any:
        """Load the partitioned record set from disk."""
        if not self._filepath.exists():
            raise DatasetError(
                f"Partitioned record set directory does not exist: {self._filepath.as_posix()}"
            )

        allowed_relative_paths = self._load_allowed_relative_paths()
        parameters = self._load_parameters()
        records: list[dict[str, Any]] = []
        records_root = self._filepath / "records"
        if records_root.exists():
            for path in sorted(records_root.rglob(f"*{self._record_suffix}")):
                if not path.is_file():
                    continue
                with path.open("r", encoding="utf-8") as stream:
                    record = json.load(stream)
                if not isinstance(record, dict):
                    raise DatasetError(
                        f"Record files must contain JSON objects: {path.as_posix()}"
                    )
                relative_path = str(record.get("relative_path", "")).strip()
                if not relative_path:
                    raise DatasetError(
                        "Partitioned record files must contain a non-empty "
                        f"'relative_path': {path.as_posix()}"
                    )
                if (
                    allowed_relative_paths is not None
                    and relative_path not in allowed_relative_paths
                ):
                    continue
                records.append(record)

        records.sort(key=lambda item: str(item["relative_path"]).casefold())
        return {"parameters": parameters, "records": records}

    def save(self, data: Any) -> None:
        """Persist a record set as one JSON file per record."""
        serializable = to_json_compatible(data)
        if not isinstance(serializable, dict):
            raise DatasetError("Partitioned record sets must serialize to mappings.")

        allowed_relative_paths = self._load_allowed_relative_paths()
        parameters = serializable.get("parameters")
        records = serializable.get("records", [])
        if not isinstance(records, list):
            raise DatasetError(
                "Partitioned record sets require a list-valued 'records'."
            )

        self._filepath.mkdir(parents=True, exist_ok=True)
        self._save_parameters(parameters)
        for record in sorted(
            records,
            key=lambda item: str(item.get("relative_path", "")).casefold(),
        ):
            if not isinstance(record, dict):
                raise DatasetError(
                    "Partitioned record entries must serialize to mappings."
                )
            relative_path = str(record.get("relative_path", "")).strip()
            if not relative_path:
                raise DatasetError(
                    "Partitioned record entries must contain a non-empty 'relative_path'."
                )
            if (
                allowed_relative_paths is not None
                and relative_path not in allowed_relative_paths
            ):
                raise DatasetError(
                    f"Record '{relative_path}' is not assigned to shard '{self._shard_id}'."
                )

            target_path = (
                self._filepath
                / "records"
                / _artifact_path_for_record(
                    relative_path,
                    self._record_suffix,
                )
            )
            payload = json.dumps(
                record,
                indent=2,
                ensure_ascii=True,
                check_circular=False,
            )
            serialized_bytes = f"{payload}\n".encode()
            if target_path.exists():
                if target_path.stat().st_size == len(serialized_bytes) and (
                    target_path.read_bytes() == serialized_bytes
                ):
                    continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(serialized_bytes)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "record_suffix": self._record_suffix,
            "partition_index_filepath": (
                None
                if self._partition_index_filepath is None
                else self._partition_index_filepath.as_posix()
            ),
            "shard_id": self._shard_id,
        }

    def _load_allowed_relative_paths(self) -> frozenset[str] | None:
        if self._partition_index_filepath is None or self._shard_id is None:
            return None
        if not self._partition_index_filepath.exists():
            raise DatasetError(
                "Partition index file does not exist: "
                f"{self._partition_index_filepath.as_posix()}"
            )
        with self._partition_index_filepath.open("r", encoding="utf-8") as stream:
            partition_index = json.load(stream)
        try:
            return relative_paths_for_shard(
                coerce_partition_index_entries(partition_index),
                self._shard_id,
            )
        except ValueError as exc:
            raise DatasetError(str(exc)) from exc

    def _load_parameters(self) -> Any:
        shared_parameters_path = self._filepath / "parameters.json"
        if shared_parameters_path.exists():
            with shared_parameters_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        per_shard_root = self._filepath / "parameters"
        if not per_shard_root.exists():
            return None

        parameter_paths = sorted(per_shard_root.glob("*.json"))
        if not parameter_paths:
            return None

        if self._shard_id is not None:
            shard_path = per_shard_root / f"{self._shard_id}.json"
            if shard_path.exists():
                with shard_path.open("r", encoding="utf-8") as stream:
                    return json.load(stream)

        loaded_values = [
            json.loads(path.read_text(encoding="utf-8")) for path in parameter_paths
        ]
        first_value = loaded_values[0]
        for value in loaded_values[1:]:
            if value != first_value:
                raise DatasetError(
                    "Partitioned record set contains inconsistent shard parameter files."
                )
        return first_value

    def _save_parameters(self, parameters: Any) -> None:
        if parameters is None:
            return

        serializable = to_json_compatible(parameters)
        if self._shard_id is None:
            target_path = self._filepath / "parameters.json"
        else:
            target_path = self._filepath / "parameters" / f"{self._shard_id}.json"

        payload = json.dumps(
            serializable,
            indent=2,
            ensure_ascii=True,
            check_circular=False,
        )
        serialized_bytes = f"{payload}\n".encode()
        if target_path.exists():
            if target_path.stat().st_size == len(serialized_bytes) and (
                target_path.read_bytes() == serialized_bytes
            ):
                return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(serialized_bytes)


def _artifact_path_for_record(relative_path: str, record_suffix: str) -> str:
    path = Path(relative_path)
    if path.is_absolute():
        raise DatasetError("Relative record paths must not be absolute.")
    if any(part == ".." for part in path.parts):
        raise DatasetError("Relative record paths must not escape the dataset root.")
    if str(path) in {"", "."}:
        raise DatasetError("Relative record paths must point to a file-like location.")
    return Path(f"{path.as_posix()}{record_suffix}").as_posix()


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
