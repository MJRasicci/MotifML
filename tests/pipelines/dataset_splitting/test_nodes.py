"""Unit tests for deterministic dataset splitting."""

from __future__ import annotations

from typing import Final

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.models import IrDocumentMetadata, MotifMlIrDocument
from motifml.pipelines.dataset_splitting.models import (
    DataSplitParameters,
    GroupingStrategy,
    SplitRatios,
)
from motifml.pipelines.dataset_splitting.nodes import (
    assign_dataset_splits,
    build_split_statistics,
)
from motifml.training.contracts import DatasetSplit, SplitManifestEntry

EXPECTED_DOCUMENT_COUNT: Final = 3
EXPECTED_GROUP_COUNT: Final = 2
EXPECTED_TRAIN_DOCUMENT_COUNT: Final = 2


def test_assign_dataset_splits_is_stable_and_sorts_by_relative_path() -> None:
    records = [
        _build_record("fixtures/b.json"),
        _build_record("fixtures/a.json"),
        _build_record("fixtures/c.json"),
    ]

    first = assign_dataset_splits(records, {"hash_seed": 17})
    second = assign_dataset_splits(
        records,
        DataSplitParameters(
            ratios=SplitRatios(train=0.8, validation=0.1, test=0.1),
            hash_seed="17",
        ),
    )

    assert first == second
    assert [entry.relative_path for entry in first] == [
        "fixtures/a.json",
        "fixtures/b.json",
        "fixtures/c.json",
    ]
    assert len({entry.split_version for entry in first}) == 1


def test_assign_dataset_splits_can_force_all_documents_into_one_ratio_bucket() -> None:
    manifest = assign_dataset_splits(
        [_build_record("fixtures/a.json"), _build_record("fixtures/b.json")],
        {
            "ratios": {"train": 0.0, "validation": 1.0, "test": 0.0},
            "hash_seed": 17,
        },
    )

    assert {entry.split for entry in manifest} == {DatasetSplit.VALIDATION}


def test_assign_dataset_splits_keeps_parent_directory_groups_together() -> None:
    manifest = assign_dataset_splits(
        [
            _build_record("collection_a/song_one.json"),
            _build_record("collection_a/song_two.json"),
            _build_record("collection_b/song_three.json"),
        ],
        {
            "grouping_strategy": GroupingStrategy.PARENT_DIRECTORY,
            "grouping_key_fallback": GroupingStrategy.RELATIVE_PATH,
            "hash_seed": 17,
        },
    )

    by_path = {entry.relative_path: entry for entry in manifest}

    assert by_path["collection_a/song_one.json"].group_key == "collection_a"
    assert by_path["collection_a/song_two.json"].group_key == "collection_a"
    assert (
        by_path["collection_a/song_one.json"].split
        == by_path["collection_a/song_two.json"].split
    )


def test_build_split_statistics_reports_per_split_counts_and_token_placeholders() -> (
    None
):
    report = build_split_statistics(
        (
            SplitManifestEntry(
                document_id="collection_a/song_one.json",
                relative_path="collection_a/song_one.json",
                split=DatasetSplit.TRAIN,
                group_key="collection_a",
                split_version="split-v1",
            ),
            SplitManifestEntry(
                document_id="collection_a/song_two.json",
                relative_path="collection_a/song_two.json",
                split=DatasetSplit.TRAIN,
                group_key="collection_a",
                split_version="split-v1",
            ),
            SplitManifestEntry(
                document_id="collection_b/song_three.json",
                relative_path="collection_b/song_three.json",
                split=DatasetSplit.TEST,
                group_key="collection_b",
                split_version="split-v1",
            ),
        )
    )

    assert report.total_document_count == EXPECTED_DOCUMENT_COUNT
    assert report.total_group_count == EXPECTED_GROUP_COUNT
    assert [entry.split for entry in report.splits] == [
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
        DatasetSplit.TEST,
    ]
    assert report.splits[0].document_count == EXPECTED_TRAIN_DOCUMENT_COUNT
    assert report.splits[0].group_count == 1
    assert report.splits[0].token_count is None
    assert report.splits[1].document_count == 0
    assert report.splits[2].document_count == 1


def _build_record(relative_path: str) -> MotifIrDocumentRecord:
    return MotifIrDocumentRecord(
        relative_path=relative_path,
        document=MotifMlIrDocument(
            metadata=IrDocumentMetadata(
                ir_schema_version="1.0.0",
                corpus_build_version="ir-build-v1",
                generator_version="tests",
                source_document_hash=f"hash:{relative_path}",
            )
        ),
    )
