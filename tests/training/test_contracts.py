"""Tests for typed training metadata contracts."""

from __future__ import annotations

import pytest

from motifml.training.contracts import (
    DatasetSplit,
    EvaluationRunMetadata,
    ModelInputMetadata,
    SplitManifestEntry,
    SplitStatsEntry,
    SplitStatsReport,
    TrainingRunMetadata,
    VocabularyMetadata,
    deserialize_metadata_artifact,
    deserialize_split_manifest,
    serialize_metadata_artifact,
    serialize_split_manifest,
    sort_split_manifest_entries,
)


def test_split_manifest_entry_round_trips_through_json() -> None:
    entry = SplitManifestEntry(
        document_id="doc-1",
        relative_path="fixtures/example.json",
        split="train",
        group_key="doc-1",
        split_version="split-v1",
    )

    payload = entry.to_json_dict()
    restored = SplitManifestEntry.from_json_dict(payload)

    assert payload["split"] == "train"
    assert restored == entry


def test_split_manifest_helpers_sort_entries_in_stable_relative_path_order() -> None:
    entries = [
        SplitManifestEntry(
            document_id="Doc-B",
            relative_path="fixtures/B.json",
            split="validation",
            group_key="group-b",
            split_version="split-v1",
        ),
        SplitManifestEntry(
            document_id="doc-a",
            relative_path="fixtures/a.json",
            split="train",
            group_key="group-a",
            split_version="split-v1",
        ),
    ]

    serialized = serialize_split_manifest(entries)
    restored = deserialize_split_manifest(serialized)

    assert [entry.relative_path for entry in sort_split_manifest_entries(entries)] == [
        "fixtures/a.json",
        "fixtures/B.json",
    ]
    assert [entry["relative_path"] for entry in serialized] == [
        "fixtures/a.json",
        "fixtures/B.json",
    ]
    assert [entry.relative_path for entry in restored] == [
        "fixtures/a.json",
        "fixtures/B.json",
    ]


def test_split_stats_report_round_trips_and_sorts_split_entries() -> None:
    report = SplitStatsReport(
        split_version="split-v1",
        total_document_count=10,
        total_group_count=9,
        splits=(
            SplitStatsEntry(
                split=DatasetSplit.TEST,
                document_count=1,
                group_count=1,
                token_count=None,
            ),
            SplitStatsEntry(
                split=DatasetSplit.TRAIN,
                document_count=8,
                group_count=7,
                token_count=1024,
            ),
            SplitStatsEntry(
                split=DatasetSplit.VALIDATION,
                document_count=1,
                group_count=1,
                token_count=128,
            ),
        ),
    )

    payload = report.to_json_dict()
    restored = SplitStatsReport.from_json_dict(payload)

    assert [entry["split"] for entry in payload["splits"]] == [
        "train",
        "validation",
        "test",
    ]
    assert restored == report


def test_split_stats_entry_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="group_count"):
        SplitStatsEntry(
            split="train",
            document_count=1,
            group_count=-1,
            token_count=None,
        )


def test_vocabulary_metadata_validates_non_negative_counts() -> None:
    with pytest.raises(ValueError, match="token_count"):
        VocabularyMetadata(
            vocabulary_version="vocab-v1",
            feature_version="feature-v1",
            split_version="split-v1",
            token_count=-1,
            vocabulary_size=32,
            construction_parameters={"minimum_frequency": 2},
            special_token_policy={"bos": "document"},
        )


def test_vocabulary_metadata_round_trips_with_canonicalized_snapshots() -> None:
    metadata = VocabularyMetadata(
        vocabulary_version="vocab-v1",
        feature_version="feature-v1",
        split_version="split-v1",
        token_count=128,
        vocabulary_size=32,
        construction_parameters={"z": 2, "a": {"y": True, "x": False}},
        special_token_policy={"eos": "document", "bos": "document"},
    )

    payload = metadata.to_json_dict()
    restored = VocabularyMetadata.from_json_dict(payload)

    assert list(payload["construction_parameters"]) == ["a", "z"]
    assert restored == metadata


def test_model_input_metadata_requires_positive_window_settings() -> None:
    with pytest.raises(ValueError, match="context_length"):
        ModelInputMetadata(
            model_input_version="model-input-v1",
            normalized_ir_version="normalized-v1",
            feature_version="feature-v1",
            vocabulary_version="vocab-v1",
            projection_type="sequence",
            sequence_mode="baseline",
            context_length=0,
            stride=128,
            padding_strategy="right",
            special_token_policy={"bos": "document"},
            storage_backend="parquet",
            storage_schema_version="parquet-v1",
        )


def test_training_run_metadata_round_trips_through_generic_helpers() -> None:
    metadata = TrainingRunMetadata(
        training_run_id="run-001",
        normalized_ir_version="normalized-v1",
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_version="model-input-v1",
        seed=17,
        model_parameters={"layers": 4, "hidden_size": 256},
        training_parameters={"batch_size": 16, "learning_rate": 0.001},
        started_at="2026-03-20T12:00:00-04:00",
        device="cpu",
    )

    payload = serialize_metadata_artifact(metadata)
    restored = deserialize_metadata_artifact(payload, TrainingRunMetadata)

    assert restored == metadata


def test_evaluation_run_metadata_normalizes_splits_and_round_trips() -> None:
    metadata = EvaluationRunMetadata(
        evaluation_run_id="eval-001",
        training_run_id="run-001",
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_version="model-input-v1",
        evaluation_parameters={"top_k": 5, "splits": ["validation", "test"]},
        evaluated_splits=("validation", DatasetSplit.TEST),
        started_at="2026-03-20T13:00:00-04:00",
    )

    payload = metadata.to_json_dict()
    restored = EvaluationRunMetadata.from_json_dict(payload)

    assert payload["evaluated_splits"] == ["validation", "test"]
    assert restored.evaluated_splits == (
        DatasetSplit.VALIDATION,
        DatasetSplit.TEST,
    )


def test_serialize_metadata_artifact_handles_collections() -> None:
    serialized = serialize_metadata_artifact(
        [
            SplitManifestEntry(
                document_id="doc-1",
                relative_path="fixtures/a.json",
                split="train",
                group_key="doc-1",
                split_version="split-v1",
            ),
            SplitManifestEntry(
                document_id="doc-2",
                relative_path="fixtures/b.json",
                split="validation",
                group_key="doc-2",
                split_version="split-v1",
            ),
        ]
    )

    assert serialized == [
        {
            "document_id": "doc-1",
            "relative_path": "fixtures/a.json",
            "split": "train",
            "group_key": "doc-1",
            "split_version": "split-v1",
        },
        {
            "document_id": "doc-2",
            "relative_path": "fixtures/b.json",
            "split": "validation",
            "group_key": "doc-2",
            "split_version": "split-v1",
        },
    ]
