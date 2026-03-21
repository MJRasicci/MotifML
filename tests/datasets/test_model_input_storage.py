"""Tests for the frozen tokenized model-input storage contract."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from motifml.datasets.model_input_storage import (
    MODEL_INPUT_PARAMETERS_FILENAME,
    MODEL_INPUT_PARTITION_FIELDS,
    MODEL_INPUT_RECORD_SUFFIX,
    MODEL_INPUT_STORAGE_BACKEND,
    MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
    MODEL_INPUT_STORAGE_SCHEMA_VERSION,
    ModelInputStorageSchema,
    coerce_model_input_storage_schema,
)


def test_model_input_storage_schema_defaults_to_the_frozen_parquet_contract() -> None:
    schema = ModelInputStorageSchema()

    assert schema.backend == MODEL_INPUT_STORAGE_BACKEND
    assert schema.storage_schema_version == MODEL_INPUT_STORAGE_SCHEMA_VERSION
    assert schema.record_suffix == MODEL_INPUT_RECORD_SUFFIX
    assert schema.partition_fields == MODEL_INPUT_PARTITION_FIELDS
    assert schema.to_json_dict() == {
        "backend": MODEL_INPUT_STORAGE_BACKEND,
        "storage_schema_version": MODEL_INPUT_STORAGE_SCHEMA_VERSION,
        "record_suffix": MODEL_INPUT_RECORD_SUFFIX,
        "partition_fields": list(MODEL_INPUT_PARTITION_FIELDS),
        "parameters_filename": MODEL_INPUT_PARAMETERS_FILENAME,
        "storage_schema_filename": MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
    }


def test_model_input_storage_schema_builds_split_and_shard_partition_paths() -> None:
    schema = ModelInputStorageSchema()

    assert schema.record_path(
        split="train",
        shard_id="shard-00003",
        relative_path="collection/demo.json",
    ) == ("records/train/shard-00003/" "collection/demo.json.model_input.parquet")


def test_coerce_model_input_storage_schema_accepts_model_input_storage_settings() -> (
    None
):
    schema = coerce_model_input_storage_schema(
        {"backend": "parquet", "schema_version": "parquet-v1"}
    )

    assert schema == ModelInputStorageSchema()


def test_parquet_backend_is_available_in_supported_dev_environments(
    tmp_path: Path,
) -> None:
    table = pa.table({"token_ids": [[1, 2, 3]], "token_count": [3]})
    target_path = tmp_path / "backend_check.parquet"

    pq.write_table(table, target_path)
    loaded = pq.read_table(target_path)

    assert loaded.column_names == ["token_ids", "token_count"]
    assert loaded.to_pylist() == [{"token_ids": [1, 2, 3], "token_count": 3}]
