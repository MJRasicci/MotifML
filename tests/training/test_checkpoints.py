"""Tests for training checkpoint helpers."""

from __future__ import annotations

from motifml.training.checkpoints import (
    CheckpointManifestEntry,
    build_checkpoint_name,
    select_best_checkpoint,
    serialize_checkpoint_manifest,
)


def test_build_checkpoint_name_uses_stable_epoch_format() -> None:
    assert build_checkpoint_name(7) == "epoch-0007.pt"


def test_select_best_checkpoint_uses_lowest_validation_loss_then_earliest_epoch() -> (
    None
):
    best = select_best_checkpoint(
        [
            CheckpointManifestEntry(
                epoch_index=2,
                validation_loss=1.5,
                checkpoint_name="epoch-0002.pt",
                saved_at="2026-03-21T10:00:00-04:00",
            ),
            CheckpointManifestEntry(
                epoch_index=1,
                validation_loss=1.5,
                checkpoint_name="epoch-0001.pt",
                saved_at="2026-03-21T09:59:00-04:00",
            ),
            CheckpointManifestEntry(
                epoch_index=0,
                validation_loss=2.0,
                checkpoint_name="epoch-0000.pt",
                saved_at="2026-03-21T09:58:00-04:00",
            ),
        ]
    )

    assert best.epoch_index == 1
    assert best.checkpoint_name == "epoch-0001.pt"


def test_serialize_checkpoint_manifest_sorts_by_epoch() -> None:
    serialized = serialize_checkpoint_manifest(
        [
            {
                "epoch_index": 2,
                "validation_loss": 1.5,
                "checkpoint_name": "epoch-0002.pt",
                "saved_at": "2026-03-21T10:00:00-04:00",
            },
            {
                "epoch_index": 1,
                "validation_loss": 1.7,
                "checkpoint_name": "epoch-0001.pt",
                "saved_at": "2026-03-21T09:59:00-04:00",
            },
        ]
    )

    assert [entry["epoch_index"] for entry in serialized] == [1, 2]
