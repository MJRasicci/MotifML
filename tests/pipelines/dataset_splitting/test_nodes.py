"""Unit tests for deterministic dataset splitting."""

from __future__ import annotations

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.models import IrDocumentMetadata, MotifMlIrDocument
from motifml.pipelines.dataset_splitting.models import (
    DataSplitParameters,
    GroupingStrategy,
    SplitRatios,
)
from motifml.pipelines.dataset_splitting.nodes import assign_dataset_splits
from motifml.training.contracts import DatasetSplit


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
