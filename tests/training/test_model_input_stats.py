"""Tests for tokenized model-input reporting helpers."""

from __future__ import annotations

import json

from motifml.training.model_input import TokenizedDocumentRow
from motifml.training.model_input_stats import (
    ModelInputReportingParameters,
    build_model_input_shard_stats,
    reduce_model_input_stats_shards,
)
from motifml.training.special_token_policy import SpecialTokenPolicy

EXPECTED_FIRST_SHARD_DOCUMENT_COUNT = 3
EXPECTED_FIRST_SHARD_TOTAL_TOKEN_COUNT = 180
EXPECTED_FIRST_SHARD_P95_TOKEN_COUNT = 110
EXPECTED_REDUCED_TOTAL_DOCUMENT_COUNT = 5


def test_build_model_input_shard_stats_reports_oversized_documents_without_token_ids() -> (
    None
):
    stats = build_model_input_shard_stats(
        (
            _build_row("train/a.json", split="train", token_count=20),
            _build_row("train/b.json", split="train", token_count=50),
            _build_row("validation/c.json", split="validation", token_count=110),
        ),
        shard_id="shard-00000",
        reporting_parameters=ModelInputReportingParameters(
            worst_document_limit=2,
            oversized_token_count_threshold=100,
        ),
    )

    assert stats.document_count == EXPECTED_FIRST_SHARD_DOCUMENT_COUNT
    assert stats.total_token_count == EXPECTED_FIRST_SHARD_TOTAL_TOKEN_COUNT
    assert stats.max_token_count == EXPECTED_FIRST_SHARD_P95_TOKEN_COUNT
    assert stats.p95_token_count == EXPECTED_FIRST_SHARD_P95_TOKEN_COUNT
    assert stats.split_token_counts == {
        "train": (20, 50),
        "validation": (110,),
    }
    assert [entry.relative_path for entry in stats.top_documents] == [
        "validation/c.json",
        "train/b.json",
    ]
    assert stats.top_documents[0].exceeds_oversized_threshold is True
    assert stats.top_documents[1].exceeds_oversized_threshold is False
    assert "token_ids" not in json.dumps(stats.to_json_dict(), sort_keys=True)


def test_reduce_model_input_stats_shards_is_deterministic_and_aggregates_exact_counts() -> (
    None
):
    shard_zero = build_model_input_shard_stats(
        (
            _build_row("train/a.json", split="train", token_count=20),
            _build_row("train/b.json", split="train", token_count=50),
            _build_row("validation/c.json", split="validation", token_count=110),
        ),
        shard_id="shard-00000",
        reporting_parameters={
            "worst_document_limit": 2,
            "oversized_token_count_threshold": 100,
        },
    )
    shard_one = build_model_input_shard_stats(
        (
            _build_row("train/d.json", split="train", token_count=40),
            _build_row("test/e.json", split="test", token_count=70),
        ),
        shard_id="shard-00001",
        reporting_parameters={
            "worst_document_limit": 2,
            "oversized_token_count_threshold": 100,
        },
    )

    first = reduce_model_input_stats_shards((shard_zero, shard_one))
    repeated = reduce_model_input_stats_shards((shard_one, shard_zero))

    assert first == repeated
    assert first.total_document_count == EXPECTED_REDUCED_TOTAL_DOCUMENT_COUNT
    assert [entry.shard_id for entry in first.shard_summaries] == [
        "shard-00000",
        "shard-00001",
    ]
    assert {
        entry.split.value: entry.total_token_count for entry in first.split_summaries
    } == {
        "train": 110,
        "validation": 110,
        "test": 70,
    }
    assert {
        entry.split.value: entry.p95_token_count for entry in first.split_summaries
    } == {
        "train": 50,
        "validation": 110,
        "test": 70,
    }
    assert [entry.relative_path for entry in first.worst_offending_documents] == [
        "validation/c.json",
        "test/e.json",
    ]


def _build_row(
    relative_path: str, *, split: str, token_count: int
) -> TokenizedDocumentRow:
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
        token_count=token_count,
        token_ids=tuple(range(token_count)),
        window_start_offsets=(0,) if token_count > 0 else (),
        context_length=256,
        stride=128,
        padding_strategy="right",
        special_token_policy=SpecialTokenPolicy().to_version_payload(),
    )
