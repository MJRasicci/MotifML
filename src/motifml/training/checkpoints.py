"""Checkpoint manifest helpers for baseline training runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible

CHECKPOINTS_DIRECTORY_NAME = "checkpoints"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"
MODEL_CONFIG_FILENAME = "model_config.json"
TRAINING_CONFIG_FILENAME = "training_config.json"
RUN_METADATA_FILENAME = "run_metadata.json"


@dataclass(frozen=True, slots=True)
class CheckpointManifestEntry:
    """One persisted training checkpoint with its validation score."""

    epoch_index: int
    validation_loss: float
    checkpoint_name: str
    saved_at: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.epoch_index, "epoch_index")
        object.__setattr__(
            self,
            "validation_loss",
            _normalize_non_negative_float(self.validation_loss, "validation_loss"),
        )
        object.__setattr__(
            self,
            "checkpoint_name",
            _normalize_checkpoint_name(self.checkpoint_name),
        )
        object.__setattr__(
            self,
            "saved_at",
            _normalize_non_empty_text(self.saved_at, "saved_at"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the manifest entry into a JSON-compatible mapping."""
        return to_json_compatible(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> CheckpointManifestEntry:
        """Deserialize one checkpoint manifest entry from JSON."""
        return cls(
            epoch_index=int(payload["epoch_index"]),
            validation_loss=float(payload["validation_loss"]),
            checkpoint_name=str(payload["checkpoint_name"]),
            saved_at=str(payload["saved_at"]),
        )


def build_checkpoint_name(epoch_index: int) -> str:
    """Build the canonical checkpoint filename for one epoch."""
    normalized_epoch_index = _require_non_negative_int(epoch_index, "epoch_index")
    return f"epoch-{normalized_epoch_index:04d}.pt"


def coerce_checkpoint_manifest_entries(
    value: Sequence[CheckpointManifestEntry | Mapping[str, Any]],
) -> tuple[CheckpointManifestEntry, ...]:
    """Coerce checkpoint manifest data into stable epoch order."""
    entries = [
        entry
        if isinstance(entry, CheckpointManifestEntry)
        else CheckpointManifestEntry.from_json_dict(entry)
        for entry in value
    ]
    return tuple(
        sorted(entries, key=lambda entry: (entry.epoch_index, entry.checkpoint_name))
    )


def select_best_checkpoint(
    entries: Sequence[CheckpointManifestEntry | Mapping[str, Any]],
) -> CheckpointManifestEntry:
    """Select the best checkpoint by validation loss, then earliest epoch."""
    typed_entries = coerce_checkpoint_manifest_entries(entries)
    if not typed_entries:
        raise ValueError("At least one checkpoint entry is required.")
    return min(
        typed_entries,
        key=lambda entry: (entry.validation_loss, entry.epoch_index),
    )


def serialize_checkpoint_manifest(
    entries: Sequence[CheckpointManifestEntry | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize checkpoint entries into deterministic JSON order."""
    return [
        entry.to_json_dict() for entry in coerce_checkpoint_manifest_entries(entries)
    ]


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_checkpoint_name(value: str) -> str:
    normalized = _normalize_non_empty_text(value, "checkpoint_name")
    if "/" in normalized or normalized in {".", ".."}:
        raise ValueError("checkpoint_name must be a safe relative filename.")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _normalize_non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric.")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return normalized


__all__ = [
    "BEST_CHECKPOINT_FILENAME",
    "CHECKPOINT_MANIFEST_FILENAME",
    "CHECKPOINTS_DIRECTORY_NAME",
    "CheckpointManifestEntry",
    "MODEL_CONFIG_FILENAME",
    "RUN_METADATA_FILENAME",
    "TRAINING_CONFIG_FILENAME",
    "build_checkpoint_name",
    "coerce_checkpoint_manifest_entries",
    "select_best_checkpoint",
    "serialize_checkpoint_manifest",
]
