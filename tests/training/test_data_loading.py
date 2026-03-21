"""Tests for lazy training-time model-input loading utilities."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from motifml.datasets.tokenized_model_input_dataset import TokenizedModelInputDataset
from motifml.training.data_loading import (
    LazyTokenizedDocumentDataset,
    LazyTokenWindowDataset,
    LoadedTokenizedDocument,
    SpecialTokenIds,
    build_token_window_example,
    discover_model_input_shards,
)
from motifml.training.model_input import TokenizedDocumentRow
from motifml.training.special_token_policy import (
    PaddingInteraction,
    SpecialTokenPolicy,
)


def test_discover_model_input_shards_returns_sorted_split_scoped_shards(
    tmp_path: Path,
) -> None:
    dataset_root = _build_model_input_fixture(tmp_path)

    assert discover_model_input_shards(dataset_root, split="train") == (
        "shard-00000",
        "shard-00001",
    )
    assert discover_model_input_shards(dataset_root, split="validation") == (
        "shard-00001",
    )
    assert discover_model_input_shards(dataset_root, split="test") == ()


def test_lazy_tokenized_document_dataset_loads_rows_on_demand(
    tmp_path: Path,
) -> None:
    dataset_root = _build_model_input_fixture(tmp_path)
    loaded_paths: list[str] = []

    def recording_loader(path: str | Path) -> TokenizedDocumentRow:
        loaded_paths.append(Path(path).name)
        return TokenizedDocumentRow.from_row_dict(
            {
                "relative_path": _relative_path_from_filename(Path(path).name),
                "document_id": _relative_path_from_filename(Path(path).name),
                "split": "train",
                "split_version": "split-v1",
                "projection_type": "sequence",
                "sequence_mode": "baseline_v1",
                "normalized_ir_version": "normalized-v1",
                "feature_version": "feature-v1",
                "vocabulary_version": "vocab-v1",
                "model_input_version": "model-input-v1",
                "storage_schema_version": "parquet-v1",
                "token_count": 4,
                "token_ids": [1, 2, 3, 4],
                "window_start_offsets": [0],
                "context_length": 4,
                "stride": 2,
                "padding_strategy": "right",
                "special_token_policy": SpecialTokenPolicy().to_version_payload(),
            }
        )

    dataset = LazyTokenizedDocumentDataset(
        dataset_root,
        split="train",
        row_loader=recording_loader,
    )

    first_document = next(iter(dataset))

    assert first_document.shard_id == "shard-00000"
    assert first_document.row.relative_path == "fixtures/a.json"
    assert loaded_paths == ["a.json.model_input.parquet"]

    remaining = list(islice(iter(dataset), 2))

    assert [document.row.relative_path for document in remaining] == [
        "fixtures/a.json",
        "fixtures/c.json",
    ]
    assert loaded_paths == [
        "a.json.model_input.parquet",
        "a.json.model_input.parquet",
        "c.json.model_input.parquet",
    ]


def test_lazy_tokenized_document_dataset_filters_to_requested_shards(
    tmp_path: Path,
) -> None:
    dataset_root = _build_model_input_fixture(tmp_path)

    dataset = LazyTokenizedDocumentDataset(
        dataset_root,
        split="train",
        shard_ids=("shard-00001",),
    )

    loaded = list(dataset)

    assert [document.shard_id for document in loaded] == ["shard-00001"]
    assert [document.row.relative_path for document in loaded] == ["fixtures/c.json"]


def test_build_token_window_example_shifts_targets_and_right_pads_short_windows() -> (
    None
):
    example = build_token_window_example(
        _build_row(relative_path="fixtures/short.json", split="validation"),
        shard_id="shard-00001",
        window_index=0,
        window_start_offset=0,
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
    )

    assert example.input_ids == (1, 2, 3, 4)
    assert example.target_ids == (2, 3, 4, 0)
    assert example.attention_mask == (1, 1, 1, 0)


def test_build_token_window_example_preserves_inside_boundary_left_padding() -> None:
    row = TokenizedDocumentRow(
        relative_path="fixtures/inside.json",
        document_id="fixtures/inside.json",
        split="train",
        split_version="split-v1",
        projection_type="sequence",
        sequence_mode="baseline_v1",
        normalized_ir_version="normalized-v1",
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_version="model-input-v1",
        storage_schema_version="parquet-v1",
        token_count=4,
        token_ids=(1, 5, 6, 2),
        window_start_offsets=(0,),
        context_length=5,
        stride=2,
        padding_strategy="left",
        special_token_policy=SpecialTokenPolicy(
            padding_interaction=PaddingInteraction.INSIDE_BOUNDARIES
        ).to_version_payload(),
    )

    example = build_token_window_example(
        row,
        shard_id="shard-00000",
        window_index=0,
        window_start_offset=0,
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
    )

    assert example.input_ids == (1, 0, 5, 6, 2)
    assert example.target_ids == (5, 0, 6, 2, 0)
    assert example.attention_mask == (1, 0, 1, 1, 0)


def test_lazy_token_window_dataset_reconstructs_persisted_windows_deterministically() -> (
    None
):
    documents = (
        LoadedTokenizedDocument(
            shard_id="shard-00000",
            record_path="records/train/shard-00000/fixtures/example.json.model_input.parquet",
            row=TokenizedDocumentRow(
                relative_path="fixtures/example.json",
                document_id="fixture-doc",
                split="train",
                split_version="split-v1",
                projection_type="sequence",
                sequence_mode="baseline_v1",
                normalized_ir_version="normalized-v1",
                feature_version="feature-v1",
                vocabulary_version="vocab-v1",
                model_input_version="model-input-v1",
                storage_schema_version="parquet-v1",
                token_count=8,
                token_ids=(1, 4, 6, 5, 8, 7, 5, 2),
                window_start_offsets=(0, 3, 4),
                context_length=4,
                stride=3,
                padding_strategy="right",
                special_token_policy=SpecialTokenPolicy().to_version_payload(),
            ),
        ),
    )
    dataset = LazyTokenWindowDataset(
        documents,
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
    )

    first = list(dataset)
    repeated = list(dataset)

    assert repeated == first
    assert [example.window_start_offset for example in first] == [0, 3, 4]
    assert first[0].input_ids == (1, 4, 6, 5)
    assert first[0].target_ids == (4, 6, 5, 8)
    assert first[0].attention_mask == (1, 1, 1, 1)
    assert first[-1].input_ids == (8, 7, 5, 2)
    assert first[-1].target_ids == (7, 5, 2, 0)
    assert first[-1].attention_mask == (1, 1, 1, 0)


def _build_model_input_fixture(tmp_path: Path) -> Path:
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
                _build_row(relative_path="fixtures/b.json", split="validation"),
                _build_row(relative_path="fixtures/c.json", split="train"),
            ],
        }
    )
    return dataset_root


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
        window_start_offsets=(0,),
        context_length=4,
        stride=2,
        padding_strategy="right",
        special_token_policy=SpecialTokenPolicy().to_version_payload(),
    )


def _relative_path_from_filename(filename: str) -> str:
    return f"fixtures/{filename.removesuffix('.model_input.parquet')}"
