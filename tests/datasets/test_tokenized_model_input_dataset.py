"""Tests for the Parquet-backed tokenized model-input dataset."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kedro.io import DatasetError

from motifml.datasets.model_input_storage import ModelInputStorageSchema
from motifml.datasets.tokenized_model_input_dataset import TokenizedModelInputDataset
from motifml.training.model_input import TokenizedDocumentRow
from motifml.training.special_token_policy import SpecialTokenPolicy

DEFAULT_STORAGE_SCHEMA = ModelInputStorageSchema()


def test_tokenized_model_input_dataset_round_trips_rows_and_layout(
    tmp_path: Path,
) -> None:
    dataset = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        shard_id="shard-00003",
    )

    dataset.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "storage_schema": DEFAULT_STORAGE_SCHEMA.to_json_dict(),
            "records": [
                _build_row(relative_path="fixtures/b.json", split="validation"),
                _build_row(relative_path="fixtures/a.json", split="train"),
            ],
        }
    )

    loaded = dataset.load()

    assert loaded["parameters"] == {"model_input_version": "model-input-v1"}
    assert loaded["storage_schema"] == DEFAULT_STORAGE_SCHEMA.to_json_dict()
    assert [record["relative_path"] for record in loaded["records"]] == [
        "fixtures/a.json",
        "fixtures/b.json",
    ]
    assert (tmp_path / "model_input" / "parameters.json").exists()
    assert (tmp_path / "model_input" / "storage_schema.json").exists()
    assert (
        tmp_path
        / "model_input"
        / "records"
        / "train"
        / "shard-00003"
        / "fixtures"
        / "a.json.model_input.parquet"
    ).exists()
    assert (
        tmp_path
        / "model_input"
        / "records"
        / "validation"
        / "shard-00003"
        / "fixtures"
        / "b.json.model_input.parquet"
    ).exists()


def test_tokenized_model_input_dataset_filters_by_split_and_shard(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "model_input"
    shard_zero = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00000",
    )
    shard_one = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00001",
    )

    shard_zero.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": [_build_row(relative_path="fixtures/a.json", split="train")],
        }
    )
    shard_one.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": [
                _build_row(relative_path="fixtures/b.json", split="validation")
            ],
        }
    )

    global_dataset = TokenizedModelInputDataset(filepath=str(dataset_root))
    train_split_dataset = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        split="train",
    )
    shard_zero_dataset = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00000",
    )

    assert [record["relative_path"] for record in global_dataset.load()["records"]] == [
        "fixtures/a.json",
        "fixtures/b.json",
    ]
    assert [
        record["relative_path"] for record in train_split_dataset.load()["records"]
    ] == ["fixtures/a.json"]
    assert [
        record["relative_path"] for record in shard_zero_dataset.load()["records"]
    ] == ["fixtures/a.json"]


def test_tokenized_model_input_dataset_rejects_incompatible_parquet_schemas(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "model_input"
    dataset = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00000",
    )
    dataset.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": [_build_row(relative_path="fixtures/a.json", split="train")],
        }
    )

    incompatible_path = (
        dataset_root
        / "records"
        / "train"
        / "shard-00000"
        / "fixtures"
        / "b.json.model_input.parquet"
    )
    incompatible_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "relative_path": ["fixtures/b.json"],
                "document_id": ["doc-b"],
                "split": ["train"],
                "split_version": ["split-v1"],
                "projection_type": ["sequence"],
                "sequence_mode": ["baseline_v1"],
                "normalized_ir_version": ["normalized-v1"],
                "feature_version": ["feature-v1"],
                "vocabulary_version": ["vocab-v1"],
                "model_input_version": ["model-input-v1"],
                "storage_schema_version": ["parquet-v1"],
                "token_count": [4],
                "token_ids": [[1, 2, 3, 4]],
                "window_start_offsets": [[0, 2]],
                "context_length": [256],
                "stride": [128],
                "padding_strategy": ["right"],
                "special_token_policy": [{"bos": "document"}],
                "inspection_metadata": [None],
                "extra_column": ["schema-mismatch"],
            }
        ),
        incompatible_path,
    )

    with pytest.raises(DatasetError, match="incompatible Parquet schemas"):
        dataset.load()


def _build_row(*, relative_path: str, split: str) -> TokenizedDocumentRow:
    return TokenizedDocumentRow(
        relative_path=relative_path,
        document_id=relative_path,
        split=split,
        split_version="split-v1",
        projection_type="sequence",
        sequence_mode="baseline_v1",
        normalized_ir_version="normalized-v1",
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_version="model-input-v1",
        storage_schema_version="parquet-v1",
        token_count=4,
        token_ids=(1, 2, 3, 4),
        window_start_offsets=(0, 2),
        context_length=256,
        stride=128,
        padding_strategy="right",
        special_token_policy=SpecialTokenPolicy().to_version_payload(),
    )
