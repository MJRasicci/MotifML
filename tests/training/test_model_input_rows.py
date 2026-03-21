"""Tests for typed tokenized-document row contracts."""

from __future__ import annotations

import pytest

from motifml.training.model_input import (
    TokenizedDocumentRow,
    build_window_start_offsets,
    sort_tokenized_document_rows,
)
from motifml.training.special_token_policy import SpecialTokenPolicy


def test_tokenized_document_row_normalizes_inputs_into_stable_equality() -> None:
    row = TokenizedDocumentRow(
        relative_path="fixtures/example.json",
        document_id="doc-001",
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
        token_ids=[3, 4, 5, 6],
        window_start_offsets=[0, 2],
        context_length=256,
        stride=128,
        padding_strategy="RIGHT",
        special_token_policy={
            "unknown_token_mapping": "map_to_unk",
            "eos": "document",
            "bos": "document",
        },
        inspection_metadata={"z": 2, "a": {"y": True, "x": False}},
    )
    repeated = TokenizedDocumentRow.from_row_dict(row.to_row_dict())

    assert repeated == row
    assert repeated.padding_strategy == "right"
    assert repeated.special_token_policy == SpecialTokenPolicy().to_version_payload()
    assert repeated.inspection_metadata == {"a": {"x": False, "y": True}, "z": 2}
    assert repeated.to_row_dict()["token_ids"] == [3, 4, 5, 6]


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"token_count": 3}, "token_count must match the number of token_ids"),
        (
            {"window_start_offsets": (2, 0)},
            "window_start_offsets must be sorted in ascending order",
        ),
        (
            {"window_start_offsets": (0, 0)},
            "window_start_offsets must be unique",
        ),
        (
            {"window_start_offsets": (0, 4)},
            "window_start_offsets must point inside the token_ids sequence",
        ),
        (
            {"token_count": 0, "token_ids": (), "window_start_offsets": (0,)},
            "window_start_offsets must be empty when token_ids are empty",
        ),
        ({"padding_strategy": "diagonal"}, "padding_strategy must be one of"),
    ],
)
def test_tokenized_document_row_rejects_invalid_contract_values(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    payload = {
        "relative_path": "fixtures/example.json",
        "document_id": "doc-001",
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
        "token_ids": (3, 4, 5, 6),
        "window_start_offsets": (0, 2),
        "context_length": 256,
        "stride": 128,
        "padding_strategy": "right",
        "special_token_policy": SpecialTokenPolicy().to_version_payload(),
        **overrides,
    }

    with pytest.raises(ValueError, match=expected_message):
        TokenizedDocumentRow(**payload)


def test_sort_tokenized_document_rows_orders_records_by_split_then_path() -> None:
    validation_row = TokenizedDocumentRow(
        relative_path="fixtures/b.json",
        document_id="doc-b",
        split="validation",
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
    train_row = TokenizedDocumentRow(
        relative_path="fixtures/a.json",
        document_id="doc-a",
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
        token_ids=(3, 4, 5, 6),
        window_start_offsets=(0, 2),
        context_length=256,
        stride=128,
        padding_strategy="right",
        special_token_policy=SpecialTokenPolicy().to_version_payload(),
    )

    sorted_rows = sort_tokenized_document_rows([validation_row, train_row])

    assert [row.relative_path for row in sorted_rows] == [
        "fixtures/a.json",
        "fixtures/b.json",
    ]


def test_build_window_start_offsets_covers_document_tail_deterministically() -> None:
    offsets = build_window_start_offsets(
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        context_length=4,
        stride=3,
    )

    assert offsets == (0, 3, 5)


def test_build_window_start_offsets_emits_one_window_for_short_documents() -> None:
    offsets = build_window_start_offsets(
        (0, 1, 2),
        context_length=8,
        stride=4,
    )

    assert offsets == (0,)
