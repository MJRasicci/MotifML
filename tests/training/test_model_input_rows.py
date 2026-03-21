"""Tests for typed tokenized-document row contracts."""

from __future__ import annotations

import pytest

from motifml.training.model_input import TokenizedDocumentRow
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
