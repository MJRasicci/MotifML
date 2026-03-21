"""Kedro dataset for persisted baseline training checkpoints and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from kedro.io import AbstractDataset, DatasetError

from motifml.datasets.json_dataset import to_json_compatible
from motifml.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    CHECKPOINT_MANIFEST_FILENAME,
    CHECKPOINTS_DIRECTORY_NAME,
    MODEL_CONFIG_FILENAME,
    RUN_METADATA_FILENAME,
    TRAINING_CONFIG_FILENAME,
    CheckpointManifestEntry,
    build_checkpoint_name,
    coerce_checkpoint_manifest_entries,
    select_best_checkpoint,
)


class TrainingCheckpointDataset(AbstractDataset[Any, Any]):
    """Persist model checkpoints plus JSON metadata under one run directory."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> dict[str, Any]:
        """Load checkpoint metadata and all persisted checkpoint payloads."""
        if not self._filepath.exists():
            raise DatasetError(
                "Training checkpoint directory does not exist: "
                f"{self._filepath.as_posix()}"
            )

        manifest_payload = self._load_json(
            self._filepath / CHECKPOINT_MANIFEST_FILENAME
        )
        if not isinstance(manifest_payload, list):
            raise DatasetError("checkpoint_manifest.json must contain a JSON list.")
        manifest = coerce_checkpoint_manifest_entries(manifest_payload)
        checkpoint_payloads = []
        for entry in manifest:
            checkpoint_path = (
                self._filepath / CHECKPOINTS_DIRECTORY_NAME / entry.checkpoint_name
            )
            if not checkpoint_path.exists():
                raise DatasetError(
                    "Persisted checkpoint manifest entry is missing its checkpoint "
                    f"file: {checkpoint_path.as_posix()}."
                )
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_payloads.append(
                {
                    "epoch_index": entry.epoch_index,
                    "validation_loss": entry.validation_loss,
                    "checkpoint_name": entry.checkpoint_name,
                    "saved_at": entry.saved_at,
                    "state": checkpoint_payload,
                }
            )

        return {
            "model_config": self._load_json(self._filepath / MODEL_CONFIG_FILENAME),
            "training_config": self._load_json(
                self._filepath / TRAINING_CONFIG_FILENAME
            ),
            "run_metadata": self._load_json(self._filepath / RUN_METADATA_FILENAME),
            "checkpoints": checkpoint_payloads,
            "best_checkpoint": self._load_json(
                self._filepath / BEST_CHECKPOINT_FILENAME
            ),
        }

    def save(self, data: Any) -> None:
        """Persist checkpoint states, manifests, and frozen run metadata."""
        if not isinstance(data, dict):
            raise DatasetError(
                "TrainingCheckpointDataset.save expects a mapping payload."
            )

        checkpoints = data.get("checkpoints")
        if not isinstance(checkpoints, list | tuple):
            raise DatasetError("checkpoints must be provided as a sequence.")
        if not checkpoints:
            raise DatasetError("At least one checkpoint is required.")

        manifest_entries: list[CheckpointManifestEntry] = []
        checkpoint_directory = self._filepath / CHECKPOINTS_DIRECTORY_NAME
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                raise DatasetError("Each checkpoint entry must be a mapping.")
            epoch_index = int(checkpoint["epoch_index"])
            checkpoint_name = str(
                checkpoint.get("checkpoint_name", build_checkpoint_name(epoch_index))
            )
            manifest_entry = CheckpointManifestEntry(
                epoch_index=epoch_index,
                validation_loss=float(checkpoint["validation_loss"]),
                checkpoint_name=checkpoint_name,
                saved_at=str(checkpoint["saved_at"]),
            )
            state = checkpoint.get("state")
            if not isinstance(state, dict):
                raise DatasetError(
                    "Each checkpoint must include a mapping state payload."
                )
            torch.save(state, checkpoint_directory / checkpoint_name)
            manifest_entries.append(manifest_entry)

        self._filepath.mkdir(parents=True, exist_ok=True)
        self._save_json(
            self._filepath / CHECKPOINT_MANIFEST_FILENAME,
            [
                entry.to_json_dict()
                for entry in coerce_checkpoint_manifest_entries(manifest_entries)
            ],
        )
        best_checkpoint = data.get("best_checkpoint")
        if best_checkpoint is None:
            best_payload = select_best_checkpoint(manifest_entries).to_json_dict()
        else:
            best_payload = (
                best_checkpoint.to_json_dict()
                if isinstance(best_checkpoint, CheckpointManifestEntry)
                else CheckpointManifestEntry.from_json_dict(
                    best_checkpoint
                ).to_json_dict()
            )
        self._save_json(self._filepath / BEST_CHECKPOINT_FILENAME, best_payload)
        self._save_json(
            self._filepath / MODEL_CONFIG_FILENAME, data.get("model_config")
        )
        self._save_json(
            self._filepath / TRAINING_CONFIG_FILENAME,
            data.get("training_config"),
        )
        self._save_json(
            self._filepath / RUN_METADATA_FILENAME,
            data.get("run_metadata"),
        )

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath.as_posix()}

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
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() == serialized_bytes:
            return
        path.write_bytes(serialized_bytes)


__all__ = ["TrainingCheckpointDataset"]
