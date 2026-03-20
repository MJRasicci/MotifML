"""Tests for the shard-bounded IR dataset."""

from __future__ import annotations

import json
from pathlib import Path

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.datasets.motif_ir_shard_dataset import MotifIrShardDataset
from motifml.ir.serialization import deserialize_document

GOLDEN_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ir" / "golden"


def test_motif_ir_shard_dataset_saves_and_loads_only_selected_shard_documents(tmp_path):
    document = deserialize_document(
        (GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json").read_text(
            encoding="utf-8"
        )
    )
    partition_index_path = tmp_path / "partition_index.json"
    partition_index_path.write_text(
        json.dumps(
            [
                {
                    "relative_path": "alpha/document.json",
                    "sha256": "alpha",
                    "file_size_bytes": 1,
                    "shard_id": "shard-00000",
                },
                {
                    "relative_path": "beta/document.json",
                    "sha256": "beta",
                    "file_size_bytes": 1,
                    "shard_id": "shard-00001",
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = MotifIrShardDataset(
        filepath=str(tmp_path / "documents"),
        partition_index_filepath=str(partition_index_path),
        shard_id="shard-00000",
    )

    dataset.save(
        [
            MotifIrDocumentRecord(
                relative_path="alpha/document.json",
                document=document,
            )
        ]
    )

    loaded = dataset.load()

    assert [record.relative_path for record in loaded] == ["alpha/document.json"]
