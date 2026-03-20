"""Tests for partitioned record-set datasets."""

from __future__ import annotations

import json

from motifml.datasets.partitioned_record_set_dataset import (
    PartitionedRecordSetDataset,
)


def test_partitioned_record_set_dataset_saves_and_loads_records_in_relative_path_order(
    tmp_path,
):
    dataset = PartitionedRecordSetDataset(
        filepath=str(tmp_path / "features"),
        record_suffix=".feature.json",
    )

    dataset.save(
        {
            "parameters": {"projection_type": "sequence"},
            "records": [
                {"relative_path": "b/Beta.json", "projection_type": "sequence"},
                {"relative_path": "a/Alpha.json", "projection_type": "sequence"},
            ],
        }
    )

    loaded = dataset.load()

    assert loaded["parameters"] == {"projection_type": "sequence"}
    assert [record["relative_path"] for record in loaded["records"]] == [
        "a/Alpha.json",
        "b/Beta.json",
    ]


def test_partitioned_record_set_dataset_filters_by_shard_and_merges_parameters(
    tmp_path,
):
    partition_index_path = tmp_path / "raw_partition_index.json"
    partition_index_path.write_text(
        json.dumps(
            [
                {
                    "relative_path": "a/Alpha.json",
                    "sha256": "alpha",
                    "file_size_bytes": 10,
                    "shard_id": "shard-00000",
                },
                {
                    "relative_path": "b/Beta.json",
                    "sha256": "beta",
                    "file_size_bytes": 10,
                    "shard_id": "shard-00001",
                },
            ]
        ),
        encoding="utf-8",
    )

    shard_zero = PartitionedRecordSetDataset(
        filepath=str(tmp_path / "features"),
        record_suffix=".feature.json",
        partition_index_filepath=str(partition_index_path),
        shard_id="shard-00000",
    )
    shard_one = PartitionedRecordSetDataset(
        filepath=str(tmp_path / "features"),
        record_suffix=".feature.json",
        partition_index_filepath=str(partition_index_path),
        shard_id="shard-00001",
    )
    global_dataset = PartitionedRecordSetDataset(
        filepath=str(tmp_path / "features"),
        record_suffix=".feature.json",
    )

    shard_zero.save(
        {
            "parameters": {"projection_type": "sequence"},
            "records": [
                {"relative_path": "a/Alpha.json", "projection_type": "sequence"},
            ],
        }
    )
    shard_one.save(
        {
            "parameters": {"projection_type": "sequence"},
            "records": [
                {"relative_path": "b/Beta.json", "projection_type": "sequence"},
            ],
        }
    )

    loaded = global_dataset.load()

    assert loaded["parameters"] == {"projection_type": "sequence"}
    assert [record["relative_path"] for record in loaded["records"]] == [
        "a/Alpha.json",
        "b/Beta.json",
    ]
