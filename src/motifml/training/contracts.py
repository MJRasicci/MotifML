"""Typed JSON-backed metadata models for MotifML training artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from motifml.datasets.json_dataset import to_json_compatible


class DatasetSplit(StrEnum):
    """Supported experiment splits for training metadata."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitManifestEntry:
    """One score-level split assignment for deterministic experiment planning."""

    document_id: str
    relative_path: str
    split: DatasetSplit
    group_key: str
    split_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "document_id", _normalize_text(self.document_id, "document_id")
        )
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(self, "split", DatasetSplit(self.split))
        object.__setattr__(
            self, "group_key", _normalize_text(self.group_key, "group_key")
        )
        object.__setattr__(
            self,
            "split_version",
            _normalize_text(self.split_version, "split_version"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the split entry for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> SplitManifestEntry:
        """Deserialize one split entry from a JSON mapping."""
        return cls(
            document_id=str(payload["document_id"]),
            relative_path=str(payload["relative_path"]),
            split=DatasetSplit(str(payload["split"])),
            group_key=str(payload["group_key"]),
            split_version=str(payload["split_version"]),
        )


@dataclass(frozen=True, slots=True)
class SplitStatsEntry:
    """Aggregate counts for one experiment split."""

    split: DatasetSplit
    document_count: int
    group_count: int
    token_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", DatasetSplit(self.split))
        _require_non_negative_int(self.document_count, "document_count")
        _require_non_negative_int(self.group_count, "group_count")
        if self.token_count is not None:
            _require_non_negative_int(self.token_count, "token_count")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize one split-stats entry for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> SplitStatsEntry:
        """Deserialize one split-stats entry from JSON."""
        token_count = payload.get("token_count")
        return cls(
            split=DatasetSplit(str(payload["split"])),
            document_count=int(payload["document_count"]),
            group_count=int(payload["group_count"]),
            token_count=None if token_count is None else int(token_count),
        )


@dataclass(frozen=True, slots=True)
class SplitStatsReport:
    """Aggregate reporting surface for a deterministic split manifest."""

    split_version: str
    total_document_count: int
    total_group_count: int
    splits: tuple[SplitStatsEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_version",
            _normalize_text(self.split_version, "split_version"),
        )
        _require_non_negative_int(self.total_document_count, "total_document_count")
        _require_non_negative_int(self.total_group_count, "total_group_count")
        normalized_splits = tuple(
            sort_split_stats_entries(
                SplitStatsEntry(
                    split=entry["split"],
                    document_count=entry["document_count"],
                    group_count=entry["group_count"],
                    token_count=entry.get("token_count"),
                )
                if isinstance(entry, Mapping)
                else SplitStatsEntry(
                    split=entry.split,
                    document_count=entry.document_count,
                    group_count=entry.group_count,
                    token_count=entry.token_count,
                )
                for entry in self.splits
            )
        )
        if not normalized_splits:
            raise ValueError("splits must contain at least one split summary.")
        object.__setattr__(self, "splits", normalized_splits)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the split-stats report for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> SplitStatsReport:
        """Deserialize one split-stats report from JSON."""
        return cls(
            split_version=str(payload["split_version"]),
            total_document_count=int(payload["total_document_count"]),
            total_group_count=int(payload["total_group_count"]),
            splits=tuple(
                SplitStatsEntry.from_json_dict(entry)
                for entry in payload.get("splits", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class VocabularyMetadata:
    """Summary metadata for a frozen vocabulary artifact."""

    vocabulary_version: str
    feature_version: str
    split_version: str
    token_count: int
    vocabulary_size: int
    construction_parameters: dict[str, Any]
    special_token_policy: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary_version",
            _normalize_text(self.vocabulary_version, "vocabulary_version"),
        )
        object.__setattr__(
            self,
            "feature_version",
            _normalize_text(self.feature_version, "feature_version"),
        )
        object.__setattr__(
            self, "split_version", _normalize_text(self.split_version, "split_version")
        )
        _require_non_negative_int(self.token_count, "token_count")
        _require_non_negative_int(self.vocabulary_size, "vocabulary_size")
        object.__setattr__(
            self,
            "construction_parameters",
            _normalize_json_mapping(
                self.construction_parameters,
                "construction_parameters",
            ),
        )
        object.__setattr__(
            self,
            "special_token_policy",
            _normalize_json_mapping(self.special_token_policy, "special_token_policy"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the vocabulary metadata for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> VocabularyMetadata:
        """Deserialize vocabulary metadata from JSON."""
        return cls(
            vocabulary_version=str(payload["vocabulary_version"]),
            feature_version=str(payload["feature_version"]),
            split_version=str(payload["split_version"]),
            token_count=int(payload["token_count"]),
            vocabulary_size=int(payload["vocabulary_size"]),
            construction_parameters=_mapping_from_payload(
                payload,
                "construction_parameters",
            ),
            special_token_policy=_mapping_from_payload(
                payload,
                "special_token_policy",
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelInputMetadata:
    """Shared metadata for persisted tokenized model-input artifacts."""

    model_input_version: str
    normalized_ir_version: str
    feature_version: str
    vocabulary_version: str
    projection_type: str
    sequence_mode: str
    context_length: int
    stride: int
    padding_strategy: str
    special_token_policy: dict[str, Any]
    storage_backend: str
    storage_schema_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_input_version",
            _normalize_text(self.model_input_version, "model_input_version"),
        )
        object.__setattr__(
            self,
            "normalized_ir_version",
            _normalize_text(self.normalized_ir_version, "normalized_ir_version"),
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
            "projection_type",
            _normalize_text(self.projection_type, "projection_type"),
        )
        object.__setattr__(
            self, "sequence_mode", _normalize_text(self.sequence_mode, "sequence_mode")
        )
        _require_positive_int(self.context_length, "context_length")
        _require_positive_int(self.stride, "stride")
        object.__setattr__(
            self,
            "padding_strategy",
            _normalize_text(self.padding_strategy, "padding_strategy"),
        )
        object.__setattr__(
            self,
            "special_token_policy",
            _normalize_json_mapping(self.special_token_policy, "special_token_policy"),
        )
        object.__setattr__(
            self,
            "storage_backend",
            _normalize_text(self.storage_backend, "storage_backend"),
        )
        object.__setattr__(
            self,
            "storage_schema_version",
            _normalize_text(self.storage_schema_version, "storage_schema_version"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the model-input metadata for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> ModelInputMetadata:
        """Deserialize model-input metadata from JSON."""
        return cls(
            model_input_version=str(payload["model_input_version"]),
            normalized_ir_version=str(payload["normalized_ir_version"]),
            feature_version=str(payload["feature_version"]),
            vocabulary_version=str(payload["vocabulary_version"]),
            projection_type=str(payload["projection_type"]),
            sequence_mode=str(payload["sequence_mode"]),
            context_length=int(payload["context_length"]),
            stride=int(payload["stride"]),
            padding_strategy=str(payload["padding_strategy"]),
            special_token_policy=_mapping_from_payload(
                payload,
                "special_token_policy",
            ),
            storage_backend=str(payload["storage_backend"]),
            storage_schema_version=str(payload["storage_schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class TrainingRunMetadata:
    """Frozen metadata for one training execution."""

    training_run_id: str
    normalized_ir_version: str
    feature_version: str
    vocabulary_version: str
    model_input_version: str
    seed: int
    model_parameters: dict[str, Any]
    training_parameters: dict[str, Any]
    started_at: str
    device: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_run_id",
            _normalize_text(self.training_run_id, "training_run_id"),
        )
        object.__setattr__(
            self,
            "normalized_ir_version",
            _normalize_text(self.normalized_ir_version, "normalized_ir_version"),
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
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "model_parameters",
            _normalize_json_mapping(self.model_parameters, "model_parameters"),
        )
        object.__setattr__(
            self,
            "training_parameters",
            _normalize_json_mapping(self.training_parameters, "training_parameters"),
        )
        object.__setattr__(
            self, "started_at", _normalize_text(self.started_at, "started_at")
        )
        object.__setattr__(self, "device", _normalize_text(self.device, "device"))

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the training run metadata for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> TrainingRunMetadata:
        """Deserialize training run metadata from JSON."""
        return cls(
            training_run_id=str(payload["training_run_id"]),
            normalized_ir_version=str(payload["normalized_ir_version"]),
            feature_version=str(payload["feature_version"]),
            vocabulary_version=str(payload["vocabulary_version"]),
            model_input_version=str(payload["model_input_version"]),
            seed=int(payload["seed"]),
            model_parameters=_mapping_from_payload(payload, "model_parameters"),
            training_parameters=_mapping_from_payload(payload, "training_parameters"),
            started_at=str(payload["started_at"]),
            device=str(payload["device"]),
        )


@dataclass(frozen=True, slots=True)
class EvaluationRunMetadata:
    """Frozen metadata for one evaluation execution."""

    evaluation_run_id: str
    training_run_id: str
    feature_version: str
    vocabulary_version: str
    model_input_version: str
    evaluation_parameters: dict[str, Any]
    evaluated_splits: tuple[DatasetSplit, ...]
    started_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_run_id",
            _normalize_text(self.evaluation_run_id, "evaluation_run_id"),
        )
        object.__setattr__(
            self,
            "training_run_id",
            _normalize_text(self.training_run_id, "training_run_id"),
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
            "evaluation_parameters",
            _normalize_json_mapping(
                self.evaluation_parameters,
                "evaluation_parameters",
            ),
        )
        object.__setattr__(
            self,
            "evaluated_splits",
            tuple(DatasetSplit(split) for split in self.evaluated_splits),
        )
        if not self.evaluated_splits:
            raise ValueError("evaluated_splits must contain at least one split.")
        object.__setattr__(
            self, "started_at", _normalize_text(self.started_at, "started_at")
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the evaluation run metadata for JSON persistence."""
        return _serialize_dataclass(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> EvaluationRunMetadata:
        """Deserialize evaluation run metadata from JSON."""
        return cls(
            evaluation_run_id=str(payload["evaluation_run_id"]),
            training_run_id=str(payload["training_run_id"]),
            feature_version=str(payload["feature_version"]),
            vocabulary_version=str(payload["vocabulary_version"]),
            model_input_version=str(payload["model_input_version"]),
            evaluation_parameters=_mapping_from_payload(
                payload,
                "evaluation_parameters",
            ),
            evaluated_splits=tuple(
                DatasetSplit(str(split)) for split in payload["evaluated_splits"]
            ),
            started_at=str(payload["started_at"]),
        )


TrainingMetadataArtifact = (
    SplitManifestEntry
    | SplitStatsEntry
    | SplitStatsReport
    | VocabularyMetadata
    | ModelInputMetadata
    | TrainingRunMetadata
    | EvaluationRunMetadata
)
TrainingMetadataType = TypeVar("TrainingMetadataType", bound=TrainingMetadataArtifact)


def serialize_metadata_artifact(
    artifact: TrainingMetadataArtifact | Sequence[TrainingMetadataArtifact],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Serialize one metadata artifact or a collection into JSON-compatible data."""
    if isinstance(artifact, Sequence) and not isinstance(artifact, str):
        return [item.to_json_dict() for item in artifact]

    return artifact.to_json_dict()


def deserialize_metadata_artifact(
    payload: Mapping[str, Any],
    artifact_type: type[TrainingMetadataType],
) -> TrainingMetadataType:
    """Deserialize one JSON mapping into the requested metadata model."""
    return artifact_type.from_json_dict(payload)


def _serialize_dataclass(value: TrainingMetadataArtifact) -> dict[str, Any]:
    return to_json_compatible(value)


def coerce_split_manifest_entries(
    value: Sequence[SplitManifestEntry] | Sequence[Mapping[str, Any]] | object,
) -> tuple[SplitManifestEntry, ...]:
    """Normalize split-manifest entries into deterministic relative-path order."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("split manifest data must be a sequence.")

    entries: list[SplitManifestEntry] = []
    for entry in value:
        if isinstance(entry, SplitManifestEntry):
            entries.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise ValueError("split manifest entries must be mappings.")
        entries.append(SplitManifestEntry.from_json_dict(entry))

    return sort_split_manifest_entries(entries)


def sort_split_manifest_entries(
    entries: Sequence[SplitManifestEntry],
) -> tuple[SplitManifestEntry, ...]:
    """Return split-manifest entries in the canonical persisted order."""
    return tuple(sorted(entries, key=_split_manifest_sort_key))


def serialize_split_manifest(
    entries: Sequence[SplitManifestEntry] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize a split manifest in canonical persisted order."""
    return [entry.to_json_dict() for entry in coerce_split_manifest_entries(entries)]


def deserialize_split_manifest(
    payload: Sequence[SplitManifestEntry] | Sequence[Mapping[str, Any]] | object,
) -> tuple[SplitManifestEntry, ...]:
    """Deserialize a split manifest and normalize it into canonical order."""
    return coerce_split_manifest_entries(payload)


def sort_split_stats_entries(
    entries: Sequence[SplitStatsEntry],
) -> tuple[SplitStatsEntry, ...]:
    """Return split-stat entries in the canonical split order."""
    return tuple(sorted(entries, key=_split_stats_sort_key))


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_json_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return _canonicalize_json_value(dict(value))


def _canonicalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, list | tuple):
        return [_canonicalize_json_value(item) for item in value]

    if isinstance(value, set | frozenset):
        return sorted(
            (_canonicalize_json_value(item) for item in value),
            key=lambda item: repr(item),
        )

    return value


def _mapping_from_payload(
    payload: Mapping[str, Any], field_name: str
) -> dict[str, Any]:
    value = payload[field_name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must deserialize from a mapping.")
    return dict(value)


def _require_non_negative_int(value: int, field_name: str) -> None:
    if int(value) < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _require_positive_int(value: int, field_name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be positive.")


def _split_manifest_sort_key(
    entry: SplitManifestEntry,
) -> tuple[str, str, int, str, str]:
    return (
        entry.relative_path.casefold(),
        entry.document_id.casefold(),
        _split_order(entry.split),
        entry.group_key.casefold(),
        entry.split_version.casefold(),
    )


def _split_stats_sort_key(entry: SplitStatsEntry) -> tuple[int, int, int]:
    token_count = -1 if entry.token_count is None else entry.token_count
    return (_split_order(entry.split), entry.document_count, token_count)


def _split_order(split: DatasetSplit) -> int:
    return {
        DatasetSplit.TRAIN: 0,
        DatasetSplit.VALIDATION: 1,
        DatasetSplit.TEST: 2,
    }[DatasetSplit(split)]


__all__ = [
    "SplitStatsEntry",
    "SplitStatsReport",
    "DatasetSplit",
    "EvaluationRunMetadata",
    "ModelInputMetadata",
    "SplitManifestEntry",
    "TrainingRunMetadata",
    "VocabularyMetadata",
    "coerce_split_manifest_entries",
    "deserialize_metadata_artifact",
    "deserialize_split_manifest",
    "serialize_metadata_artifact",
    "serialize_split_manifest",
    "sort_split_manifest_entries",
    "sort_split_stats_entries",
]
