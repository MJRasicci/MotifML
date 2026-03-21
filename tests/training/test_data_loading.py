"""Tests for lazy training-time model-input loading utilities."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import torch

from motifml.datasets.tokenized_model_input_dataset import TokenizedModelInputDataset
from motifml.training.data_loading import (
    LazyTokenizedDocumentDataset,
    LazyTokenWindowDataset,
    LoadedTokenizedDocument,
    LoaderIterationOptions,
    SpecialTokenIds,
    TokenWindowBatchCollator,
    TokenWindowExample,
    build_token_window_data_loader,
    build_token_window_example,
    discover_model_input_shards,
)
from motifml.training.model_input import TokenizedDocumentRow
from motifml.training.special_token_policy import (
    PaddingInteraction,
    SpecialTokenPolicy,
)

EXPECTED_BATCH_COUNT = 2


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
        iteration_options=LoaderIterationOptions(shuffle_windows=False),
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


def test_lazy_tokenized_document_dataset_reproducibly_shuffles_train_order_by_epoch(
    tmp_path: Path,
) -> None:
    dataset_root = _build_train_iteration_fixture(tmp_path)
    first = LazyTokenizedDocumentDataset(
        dataset_root,
        split="train",
        iteration_options=LoaderIterationOptions(seed=17),
    )
    repeated = LazyTokenizedDocumentDataset(
        dataset_root,
        split="train",
        iteration_options=LoaderIterationOptions(seed=17),
    )
    next_epoch = first.with_epoch(1)
    validation = LazyTokenizedDocumentDataset(
        dataset_root,
        split="validation",
        iteration_options=LoaderIterationOptions(seed=17),
    )

    first_order = [
        (document.shard_id, document.row.relative_path) for document in first
    ]
    repeated_order = [
        (document.shard_id, document.row.relative_path) for document in repeated
    ]
    next_epoch_order = [
        (document.shard_id, document.row.relative_path) for document in next_epoch
    ]
    validation_order = [
        (document.shard_id, document.row.relative_path) for document in validation
    ]

    assert repeated_order == first_order
    assert next_epoch_order != first_order
    assert validation_order == [
        ("shard-00001", "fixtures/validation_a.json"),
        ("shard-00002", "fixtures/validation_b.json"),
    ]


def test_lazy_token_window_dataset_reproducibly_shuffles_train_windows_only() -> None:
    document = LoadedTokenizedDocument(
        shard_id="shard-00000",
        record_path="records/train/shard-00000/fixtures/train_windows.json.model_input.parquet",
        row=TokenizedDocumentRow(
            relative_path="fixtures/train_windows.json",
            document_id="train-windows-doc",
            split="train",
            split_version="split-v1",
            projection_type="sequence",
            sequence_mode="baseline_v1",
            normalized_ir_version="normalized-v1",
            feature_version="feature-v1",
            vocabulary_version="vocab-v1",
            model_input_version="model-input-v1",
            storage_schema_version="parquet-v1",
            token_count=14,
            token_ids=(1, 4, 6, 5, 8, 7, 5, 9, 10, 11, 12, 13, 14, 2),
            window_start_offsets=(0, 2, 4, 6, 8, 10),
            context_length=4,
            stride=2,
            padding_strategy="right",
            special_token_policy=SpecialTokenPolicy().to_version_payload(),
        ),
    )
    first = LazyTokenWindowDataset(
        (document,),
        split="train",
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
        iteration_options=LoaderIterationOptions(seed=17),
    )
    repeated = LazyTokenWindowDataset(
        (document,),
        split="train",
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
        iteration_options=LoaderIterationOptions(seed=17),
    )
    next_epoch = first.with_epoch(1)
    validation = LazyTokenWindowDataset(
        (document,),
        split="validation",
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
        iteration_options=LoaderIterationOptions(seed=17),
    )

    first_offsets = [example.window_start_offset for example in first]
    repeated_offsets = [example.window_start_offset for example in repeated]
    next_epoch_offsets = [example.window_start_offset for example in next_epoch]
    validation_offsets = [example.window_start_offset for example in validation]

    assert repeated_offsets == first_offsets
    assert next_epoch_offsets != first_offsets
    assert validation_offsets == [0, 2, 4, 6, 8, 10]


def test_token_window_batch_collator_stacks_examples_into_tensors() -> None:
    collator = TokenWindowBatchCollator(pad_token_id=0)

    batch = collator(
        (
            TokenWindowExample(
                split="train",
                shard_id="shard-00000",
                relative_path="fixtures/a.json",
                document_id="doc-a",
                window_index=0,
                window_start_offset=0,
                input_ids=(1, 2, 3, 4),
                target_ids=(2, 3, 4, 0),
                attention_mask=(1, 1, 1, 0),
            ),
            TokenWindowExample(
                split="validation",
                shard_id="shard-00001",
                relative_path="fixtures/b.json",
                document_id="doc-b",
                window_index=1,
                window_start_offset=4,
                input_ids=(5, 6, 7, 8),
                target_ids=(6, 7, 8, 0),
                attention_mask=(1, 1, 1, 0),
            ),
        )
    )

    assert batch.input_ids.shape == (2, 4)
    assert batch.target_ids.shape == (2, 4)
    assert batch.attention_mask.dtype == torch.bool
    assert batch.input_ids.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert batch.splits == ("train", "validation")
    assert batch.window_start_offsets == (0, 4)


def test_token_window_data_loader_only_pads_the_current_batch() -> None:
    dataset = (
        TokenWindowExample(
            split="train",
            shard_id="shard-00000",
            relative_path="fixtures/short_a.json",
            document_id="short-a",
            window_index=0,
            window_start_offset=0,
            input_ids=(1, 2),
            target_ids=(2, 0),
            attention_mask=(1, 0),
        ),
        TokenWindowExample(
            split="train",
            shard_id="shard-00000",
            relative_path="fixtures/short_b.json",
            document_id="short-b",
            window_index=1,
            window_start_offset=2,
            input_ids=(3, 4, 5),
            target_ids=(4, 5, 0),
            attention_mask=(1, 1, 0),
        ),
        TokenWindowExample(
            split="train",
            shard_id="shard-00001",
            relative_path="fixtures/long.json",
            document_id="long-doc",
            window_index=0,
            window_start_offset=0,
            input_ids=(6, 7, 8, 9, 10, 11),
            target_ids=(7, 8, 9, 10, 11, 0),
            attention_mask=(1, 1, 1, 1, 1, 0),
        ),
    )
    loader = build_token_window_data_loader(
        dataset,
        batch_size=2,
        pad_token_id=0,
    )

    batches = list(loader)

    assert len(batches) == EXPECTED_BATCH_COUNT
    assert batches[0].input_ids.shape == (2, 3)
    assert batches[0].input_ids.tolist() == [[1, 2, 0], [3, 4, 5]]
    assert batches[0].attention_mask.tolist() == [
        [True, False, False],
        [True, True, False],
    ]
    assert batches[1].input_ids.shape == (1, 6)
    assert batches[1].input_ids.tolist() == [[6, 7, 8, 9, 10, 11]]


def test_lazy_token_window_dataset_streams_long_documents_before_later_documents() -> (
    None
):
    visited_documents: list[str] = []

    def document_source() -> object:
        visited_documents.append("long-doc")
        yield LoadedTokenizedDocument(
            shard_id="shard-00000",
            record_path="records/train/shard-00000/fixtures/long.json.model_input.parquet",
            row=TokenizedDocumentRow(
                relative_path="fixtures/long.json",
                document_id="long-doc",
                split="train",
                split_version="split-v1",
                projection_type="sequence",
                sequence_mode="baseline_v1",
                normalized_ir_version="normalized-v1",
                feature_version="feature-v1",
                vocabulary_version="vocab-v1",
                model_input_version="model-input-v1",
                storage_schema_version="parquet-v1",
                token_count=16,
                token_ids=tuple(range(16)),
                window_start_offsets=(0, 2, 4, 6, 8, 10, 12),
                context_length=4,
                stride=2,
                padding_strategy="right",
                special_token_policy=SpecialTokenPolicy().to_version_payload(),
            ),
        )
        visited_documents.append("later-doc")
        yield LoadedTokenizedDocument(
            shard_id="shard-00001",
            record_path="records/train/shard-00001/fixtures/later.json.model_input.parquet",
            row=_build_row(relative_path="fixtures/later.json", split="train"),
        )

    dataset = LazyTokenWindowDataset(
        document_source(),
        split="train",
        special_token_ids=SpecialTokenIds(
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
        iteration_options=LoaderIterationOptions(shuffle_windows=False),
    )

    first_windows = list(islice(iter(dataset), 3))

    assert [example.document_id for example in first_windows] == [
        "long-doc",
        "long-doc",
        "long-doc",
    ]
    assert visited_documents == ["long-doc"]


def test_build_token_window_data_loader_streams_the_first_batch_lazily() -> None:
    visited_documents: list[str] = []

    def document_source() -> object:
        visited_documents.append("long-doc")
        yield LoadedTokenizedDocument(
            shard_id="shard-00000",
            record_path="records/train/shard-00000/fixtures/long.json.model_input.parquet",
            row=TokenizedDocumentRow(
                relative_path="fixtures/long.json",
                document_id="long-doc",
                split="train",
                split_version="split-v1",
                projection_type="sequence",
                sequence_mode="baseline_v1",
                normalized_ir_version="normalized-v1",
                feature_version="feature-v1",
                vocabulary_version="vocab-v1",
                model_input_version="model-input-v1",
                storage_schema_version="parquet-v1",
                token_count=16,
                token_ids=tuple(range(16)),
                window_start_offsets=(0, 2, 4, 6, 8, 10, 12),
                context_length=4,
                stride=2,
                padding_strategy="right",
                special_token_policy=SpecialTokenPolicy().to_version_payload(),
            ),
        )
        visited_documents.append("later-doc")
        raise AssertionError(
            "Later documents should not be touched to assemble the first batch."
        )

    loader = build_token_window_data_loader(
        LazyTokenWindowDataset(
            document_source(),
            split="train",
            special_token_ids=SpecialTokenIds(
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                unk_token_id=3,
            ),
            iteration_options=LoaderIterationOptions(shuffle_windows=False),
        ),
        batch_size=2,
        pad_token_id=0,
    )

    first_batch = next(iter(loader))

    assert list(first_batch.document_ids) == ["long-doc", "long-doc"]
    assert list(first_batch.window_start_offsets) == [0, 2]
    assert visited_documents == ["long-doc"]


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


def _build_train_iteration_fixture(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "train_iteration_model_input"
    shard_zero = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00000",
    )
    shard_one = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00001",
    )
    shard_two = TokenizedModelInputDataset(
        filepath=str(dataset_root),
        shard_id="shard-00002",
    )

    shard_zero.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": [
                _build_row(relative_path="fixtures/train_a.json", split="train"),
                _build_row(relative_path="fixtures/train_b.json", split="train"),
            ],
        }
    )
    shard_one.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": [
                _build_row(relative_path="fixtures/train_c.json", split="train"),
                _build_row(relative_path="fixtures/train_d.json", split="train"),
                _build_row(
                    relative_path="fixtures/validation_a.json", split="validation"
                ),
            ],
        }
    )
    shard_two.save(
        {
            "parameters": {"model_input_version": "model-input-v1"},
            "records": [
                _build_row(relative_path="fixtures/train_e.json", split="train"),
                _build_row(relative_path="fixtures/train_f.json", split="train"),
                _build_row(
                    relative_path="fixtures/validation_b.json", split="validation"
                ),
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
