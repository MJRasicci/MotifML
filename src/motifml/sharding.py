"""Deterministic sharding helpers for partitioned Kedro execution."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PartitioningParameters:
    """Configuration controlling deterministic corpus sharding."""

    max_documents_per_shard: int = 64
    max_raw_bytes_per_shard: int | None = None

    def __post_init__(self) -> None:
        if self.max_documents_per_shard <= 0:
            raise ValueError("max_documents_per_shard must be positive.")
        if (
            self.max_raw_bytes_per_shard is not None
            and self.max_raw_bytes_per_shard <= 0
        ):
            raise ValueError("max_raw_bytes_per_shard must be positive when provided.")


@dataclass(frozen=True)
class RawPartitionIndexEntry:
    """Stable per-document shard assignment derived during ingestion."""

    document_id: str
    relative_path: str
    sha256: str
    file_size_bytes: int
    shard_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _normalize_text(self.document_id))
        object.__setattr__(self, "relative_path", _normalize_text(self.relative_path))
        object.__setattr__(self, "sha256", _normalize_text(self.sha256))
        if self.file_size_bytes < 0:
            raise ValueError("file_size_bytes must be non-negative.")
        object.__setattr__(self, "shard_id", _normalize_text(self.shard_id))


@dataclass(frozen=True)
class RawShardManifest:
    """One deterministic shard membership manifest."""

    shard_id: str
    document_count: int
    total_file_size_bytes: int
    document_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_id", _normalize_text(self.shard_id))
        if self.document_count < 0:
            raise ValueError("document_count must be non-negative.")
        if self.total_file_size_bytes < 0:
            raise ValueError("total_file_size_bytes must be non-negative.")
        object.__setattr__(
            self,
            "document_relative_paths",
            tuple(
                sorted(
                    (_normalize_text(path) for path in self.document_relative_paths),
                    key=str.casefold,
                )
            ),
        )


def coerce_partitioning_parameters(
    value: PartitioningParameters | Mapping[str, Any] | None,
) -> PartitioningParameters:
    """Coerce Kedro parameter mappings into typed partitioning parameters."""
    if value is None:
        return PartitioningParameters()
    if isinstance(value, PartitioningParameters):
        return value
    return PartitioningParameters(
        max_documents_per_shard=int(value.get("max_documents_per_shard", 64)),
        max_raw_bytes_per_shard=_coerce_optional_positive_int(
            value.get("max_raw_bytes_per_shard")
        ),
    )


def coerce_partition_index_entries(
    value: Sequence[RawPartitionIndexEntry] | Sequence[Mapping[str, Any]] | object,
) -> tuple[RawPartitionIndexEntry, ...]:
    """Normalize JSON-loaded partition index entries into typed records."""
    if not isinstance(value, Sequence):
        raise ValueError("partition index data must be a sequence.")

    entries: list[RawPartitionIndexEntry] = []
    for raw_entry in value:
        if isinstance(raw_entry, RawPartitionIndexEntry):
            entries.append(raw_entry)
            continue
        if not isinstance(raw_entry, Mapping):
            raise ValueError("partition index entries must be mappings.")
        relative_path = _normalize_text(str(raw_entry["relative_path"]))
        entries.append(
            RawPartitionIndexEntry(
                document_id=str(raw_entry.get("document_id", relative_path)),
                relative_path=relative_path,
                sha256=str(raw_entry["sha256"]),
                file_size_bytes=int(raw_entry["file_size_bytes"]),
                shard_id=str(raw_entry["shard_id"]),
            )
        )

    return tuple(sorted(entries, key=lambda item: item.relative_path.casefold()))


def coerce_shard_manifests(
    value: Sequence[RawShardManifest] | Sequence[Mapping[str, Any]] | object,
) -> tuple[RawShardManifest, ...]:
    """Normalize JSON-loaded shard manifests into typed records."""
    if not isinstance(value, Sequence):
        raise ValueError("shard manifest data must be a sequence.")

    manifests: list[RawShardManifest] = []
    for raw_entry in value:
        if isinstance(raw_entry, RawShardManifest):
            manifests.append(raw_entry)
            continue
        if not isinstance(raw_entry, Mapping):
            raise ValueError("shard manifests must be mappings.")
        manifests.append(
            RawShardManifest(
                shard_id=str(raw_entry["shard_id"]),
                document_count=int(raw_entry["document_count"]),
                total_file_size_bytes=int(raw_entry["total_file_size_bytes"]),
                document_relative_paths=tuple(
                    str(path) for path in raw_entry.get("document_relative_paths", ())
                ),
            )
        )

    return tuple(sorted(manifests, key=lambda item: item.shard_id.casefold()))


def build_partition_index(
    *,
    manifest_entries: Sequence[object],
    parameters: PartitioningParameters,
) -> tuple[RawPartitionIndexEntry, ...]:
    """Assign deterministic shard IDs to sorted corpus manifest entries."""
    normalized_entries = tuple(
        sorted(
            (
                _ManifestEntry(
                    relative_path=_normalize_text(
                        str(_manifest_value(entry, "relative_path"))
                    ),
                    sha256=_normalize_text(str(_manifest_value(entry, "sha256"))),
                    file_size_bytes=int(_manifest_value(entry, "file_size_bytes")),
                )
                for entry in manifest_entries
            ),
            key=lambda item: item.relative_path.casefold(),
        )
    )

    results: list[RawPartitionIndexEntry] = []
    shard_index = 0
    current_shard_paths: list[_ManifestEntry] = []
    current_shard_bytes = 0
    for entry in normalized_entries:
        would_exceed_documents = (
            len(current_shard_paths) >= parameters.max_documents_per_shard
        )
        would_exceed_bytes = (
            parameters.max_raw_bytes_per_shard is not None
            and current_shard_paths
            and current_shard_bytes + entry.file_size_bytes
            > parameters.max_raw_bytes_per_shard
        )
        if would_exceed_documents or would_exceed_bytes:
            shard_id = build_shard_id(shard_index)
            results.extend(
                RawPartitionIndexEntry(
                    document_id=item.relative_path,
                    relative_path=item.relative_path,
                    sha256=item.sha256,
                    file_size_bytes=item.file_size_bytes,
                    shard_id=shard_id,
                )
                for item in current_shard_paths
            )
            shard_index += 1
            current_shard_paths = []
            current_shard_bytes = 0

        current_shard_paths.append(entry)
        current_shard_bytes += entry.file_size_bytes

    if current_shard_paths:
        shard_id = build_shard_id(shard_index)
        results.extend(
            RawPartitionIndexEntry(
                document_id=item.relative_path,
                relative_path=item.relative_path,
                sha256=item.sha256,
                file_size_bytes=item.file_size_bytes,
                shard_id=shard_id,
            )
            for item in current_shard_paths
        )

    return tuple(sorted(results, key=lambda item: item.relative_path.casefold()))


def build_shard_manifests(
    partition_index: Sequence[RawPartitionIndexEntry] | Sequence[Mapping[str, Any]],
) -> tuple[RawShardManifest, ...]:
    """Group partition index entries into per-shard manifests."""
    entries = coerce_partition_index_entries(partition_index)
    grouped_paths: dict[str, list[str]] = defaultdict(list)
    grouped_bytes: dict[str, int] = defaultdict(int)
    for entry in entries:
        grouped_paths[entry.shard_id].append(entry.relative_path)
        grouped_bytes[entry.shard_id] += entry.file_size_bytes

    return tuple(
        RawShardManifest(
            shard_id=shard_id,
            document_count=len(grouped_paths[shard_id]),
            total_file_size_bytes=grouped_bytes[shard_id],
            document_relative_paths=tuple(grouped_paths[shard_id]),
        )
        for shard_id in sorted(grouped_paths, key=str.casefold)
    )


def relative_paths_for_shard(
    partition_index: Sequence[RawPartitionIndexEntry] | Sequence[Mapping[str, Any]],
    shard_id: str,
) -> frozenset[str]:
    """Return the relative paths assigned to one shard ID."""
    normalized_shard_id = _normalize_text(shard_id)
    entries = coerce_partition_index_entries(partition_index)
    shard_paths = frozenset(
        entry.relative_path
        for entry in entries
        if entry.shard_id == normalized_shard_id
    )
    if not shard_paths:
        raise ValueError(f"Unknown shard_id '{normalized_shard_id}'.")
    return shard_paths


def shard_ids(
    partition_index: Sequence[RawPartitionIndexEntry] | Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return all shard IDs present in the partition index."""
    entries = coerce_partition_index_entries(partition_index)
    return tuple(sorted({entry.shard_id for entry in entries}, key=str.casefold))


def build_shard_id(index: int) -> str:
    """Format a deterministic shard identifier."""
    if index < 0:
        raise ValueError("shard index must be non-negative.")
    return f"shard-{index:05d}"


@dataclass(frozen=True)
class _ManifestEntry:
    relative_path: str
    sha256: str
    file_size_bytes: int


def _manifest_value(entry: object, field_name: str) -> object:
    if isinstance(entry, Mapping):
        return entry[field_name]
    return getattr(entry, field_name)


def _coerce_optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in {"", "none", "null"}:
        return None
    integer = int(value)
    if integer <= 0:
        raise ValueError("optional positive integer must be greater than zero.")
    return integer


def _normalize_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("text values must be non-empty.")
    return normalized


__all__ = [
    "PartitionIndexEntry",
    "PartitioningParameters",
    "RawPartitionIndexEntry",
    "RawShardManifest",
    "ShardManifest",
    "build_partition_index",
    "build_shard_id",
    "build_shard_manifests",
    "coerce_partition_index_entries",
    "coerce_partitioning_parameters",
    "coerce_shard_manifests",
    "relative_paths_for_shard",
    "shard_ids",
    "shard_ids_from_entries",
]


PartitionIndexEntry = RawPartitionIndexEntry
ShardManifest = RawShardManifest
shard_ids_from_entries = shard_ids
