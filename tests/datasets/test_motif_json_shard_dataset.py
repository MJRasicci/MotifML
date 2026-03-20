"""Tests for the shard-bounded raw Motif JSON dataset."""

from __future__ import annotations

import json

from motifml.datasets.motif_json_shard_dataset import MotifJsonShardDataset


def test_motif_json_shard_dataset_loads_only_selected_shard_documents(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "a.json").write_text(json.dumps({"Title": "Alpha"}), encoding="utf-8")
    (raw_root / "b.json").write_text(json.dumps({"Title": "Beta"}), encoding="utf-8")
    (tmp_path / "partition_index.json").write_text(
        json.dumps(
            [
                {
                    "relative_path": "a.json",
                    "sha256": "alpha",
                    "file_size_bytes": 10,
                    "shard_id": "shard-00000",
                },
                {
                    "relative_path": "b.json",
                    "sha256": "beta",
                    "file_size_bytes": 10,
                    "shard_id": "shard-00001",
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = MotifJsonShardDataset(
        filepath=str(raw_root),
        partition_index_filepath=str(tmp_path / "partition_index.json"),
        shard_id="shard-00001",
    )

    loaded = dataset.load()

    assert [document.relative_path for document in loaded] == ["b.json"]
    assert loaded[0].score["Title"] == "Beta"
