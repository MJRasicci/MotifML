"""Tests for the training checkpoint Kedro dataset."""

from __future__ import annotations

from pathlib import Path

import torch

from motifml.datasets.training_checkpoint_dataset import TrainingCheckpointDataset
from motifml.training.checkpoints import CheckpointManifestEntry
from motifml.training.contracts import TrainingRunMetadata


def test_training_checkpoint_dataset_round_trips_checkpoint_payloads(
    tmp_path: Path,
) -> None:
    dataset = TrainingCheckpointDataset(filepath=str(tmp_path / "training_run"))
    payload = {
        "model_config": {"architecture": "decoder_only_transformer", "layers": 2},
        "training_config": {"batch_size": 2, "learning_rate": 0.001},
        "run_metadata": TrainingRunMetadata(
            training_run_id="run-001",
            normalized_ir_version="normalized-v1",
            feature_version="feature-v1",
            vocabulary_version="vocab-v1",
            model_input_version="model-input-v1",
            seed=17,
            model_parameters={"layers": 2},
            training_parameters={"batch_size": 2},
            started_at="2026-03-21T10:00:00-04:00",
            device="cpu",
        ),
        "checkpoints": [
            {
                "epoch_index": 0,
                "validation_loss": 1.25,
                "checkpoint_name": "epoch-0000.pt",
                "saved_at": "2026-03-21T10:01:00-04:00",
                "state": {
                    "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
                    "optimizer_state_dict": {"state": {}, "param_groups": []},
                },
            }
        ],
    }

    dataset.save(payload)
    restored = dataset.load()

    assert restored["model_config"] == payload["model_config"]
    assert restored["training_config"] == payload["training_config"]
    assert restored["run_metadata"]["training_run_id"] == "run-001"
    assert restored["best_checkpoint"]["checkpoint_name"] == "epoch-0000.pt"
    restored_weight = restored["checkpoints"][0]["state"]["model_state_dict"]["weight"]
    assert torch.equal(restored_weight, torch.tensor([1.0, 2.0]))


def test_training_checkpoint_dataset_selects_best_checkpoint_when_missing(
    tmp_path: Path,
) -> None:
    dataset = TrainingCheckpointDataset(filepath=str(tmp_path / "training_run"))
    dataset.save(
        {
            "checkpoints": [
                {
                    "epoch_index": 0,
                    "validation_loss": 1.2,
                    "saved_at": "2026-03-21T10:01:00-04:00",
                    "state": {"model_state_dict": {}, "optimizer_state_dict": {}},
                },
                {
                    "epoch_index": 1,
                    "validation_loss": 0.8,
                    "saved_at": "2026-03-21T10:02:00-04:00",
                    "state": {"model_state_dict": {}, "optimizer_state_dict": {}},
                },
            ]
        }
    )

    restored = dataset.load()
    best_checkpoint = CheckpointManifestEntry.from_json_dict(
        restored["best_checkpoint"]
    )

    assert best_checkpoint.epoch_index == 1
    assert best_checkpoint.checkpoint_name == "epoch-0001.pt"
