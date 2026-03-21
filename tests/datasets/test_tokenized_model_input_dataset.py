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

EXPECTED_GLOBAL_RECORD_COUNT = 3


def test_tokenized_model_input_dataset_round_trips_rows_and_metadata(
    tmp_path: Path,
) -> None:
    dataset = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        shard_id="shard-00000",
    )
    schema = ModelInputStorageSchema()
    rows = (
        _build_row(relative_path="a/alpha.json", split="train"),
        _build_row(relative_path="b/beta.json", split="validation"),
    )

    dataset.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "storage_schema": schema.to_json_dict(),
            "records": rows,
        }
    )

    loaded = TokenizedModelInputDataset(filepath=str(tmp_path / "model_input")).load()

    assert loaded["parameters"] == {"model_input_version": "model-input-v1"}
    assert loaded["storage_schema"] == schema.to_json_dict()
    assert [record["relative_path"] for record in loaded["records"]] == [
        "a/alpha.json",
        "b/beta.json",
    ]
    assert loaded["records"][0] == rows[0].to_row_dict()
    assert loaded["records"][1] == rows[1].to_row_dict()
    assert (
        tmp_path
        / "model_input"
        / schema.record_path(
            split="train",
            shard_id="shard-00000",
            relative_path="a/alpha.json",
        )
    ).exists()
    assert (
        tmp_path
        / "model_input"
        / schema.record_path(
            split="validation",
            shard_id="shard-00000",
            relative_path="b/beta.json",
        )
    ).exists()


def test_tokenized_model_input_dataset_filters_by_split_and_shard(
    tmp_path: Path,
) -> None:
    shard_zero = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        shard_id="shard-00000",
    )
    shard_one = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        shard_id="shard-00001",
    )
    global_dataset = TokenizedModelInputDataset(filepath=str(tmp_path / "model_input"))
    train_only = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        split="train",
    )
    shard_zero_train = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        shard_id="shard-00000",
        split="train",
    )

    shard_zero.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": (
                _build_row(relative_path="a/alpha.json", split="train"),
                _build_row(relative_path="b/beta.json", split="validation"),
            ),
        }
    )
    shard_one.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": (_build_row(relative_path="c/gamma.json", split="train"),),
        }
    )

    assert len(global_dataset.load()["records"]) == EXPECTED_GLOBAL_RECORD_COUNT
    assert [record["relative_path"] for record in train_only.load()["records"]] == [
        "a/alpha.json",
        "c/gamma.json",
    ]
    assert [
        record["relative_path"] for record in shard_zero_train.load()["records"]
    ] == [
        "a/alpha.json",
    ]


def test_tokenized_model_input_dataset_rejects_incompatible_parquet_schemas(
    tmp_path: Path,
) -> None:
    dataset = TokenizedModelInputDataset(
        filepath=str(tmp_path / "model_input"),
        shard_id="shard-00000",
    )
    dataset.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": (_build_row(relative_path="a/alpha.json", split="train"),),
        }
    )

    incompatible_path = (
        tmp_path
        / "model_input"
        / "records"
        / "train"
        / "shard-00001"
        / "broken.json.model_input.parquet"
    )
    incompatible_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"relative_path": ["broken.json"], "token_ids": ["not-an-array"]}),
        incompatible_path,
    )

    with pytest.raises(DatasetError, match="incompatible Parquet schemas"):
        TokenizedModelInputDataset(filepath=str(tmp_path / "model_input")).load()


def test_tokenized_model_input_dataset_requires_a_concrete_shard_id_for_save(
    tmp_path: Path,
) -> None:
    dataset = TokenizedModelInputDataset(filepath=str(tmp_path / "model_input"))

    with pytest.raises(DatasetError, match="requires a concrete shard_id"):
        dataset.save(
            {"records": (_build_row(relative_path="a/alpha.json", split="train"),)}
        )


def _build_row(relative_path: str, split: str) -> TokenizedDocumentRow:
    return TokenizedDocumentRow(
        relative_path=relative_path,
        document_id=f"doc:{relative_path}",
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
        token_ids=(3, 4, 5, 6),
        window_start_offsets=(0, 2),
        context_length=256,
        stride=128,
        padding_strategy="right",
        special_token_policy=SpecialTokenPolicy().to_version_payload(),
    )
