"""Unit tests for the normalization pipeline skeleton."""

from __future__ import annotations

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.models import IrDocumentMetadata, MotifMlIrDocument
from motifml.pipelines.normalization.models import NormalizationParameters
from motifml.pipelines.normalization.nodes import (
    build_normalized_ir_version,
    merge_normalized_ir_version_fragments,
    normalize_ir_corpus,
)


def test_normalize_ir_corpus_returns_the_original_document_list() -> None:
    records = [_build_record("examples/demo.json")]

    normalized = normalize_ir_corpus(records)

    assert normalized is records


def test_build_normalized_ir_version_is_deterministic_for_unchanged_inputs() -> None:
    records = [
        _build_record("examples/b.json"),
        _build_record("examples/a.json"),
    ]

    first = build_normalized_ir_version(records, NormalizationParameters())
    second = build_normalized_ir_version(records, NormalizationParameters())

    assert first == second
    assert first.normalized_ir_version


def test_merge_normalized_ir_version_fragments_requires_identical_shards() -> None:
    fragment = build_normalized_ir_version(
        [_build_record("examples/demo.json")],
        NormalizationParameters(),
    )

    merged = merge_normalized_ir_version_fragments(
        [
            fragment,
            {
                "normalized_ir_version": fragment.normalized_ir_version,
                "contract_name": fragment.contract_name,
                "contract_version": fragment.contract_version,
                "serialized_document_format": fragment.serialized_document_format,
                "normalization_strategy": fragment.normalization_strategy,
                "upstream_ir_schema_version": fragment.upstream_ir_schema_version,
                "task_agnostic_guarantees": list(fragment.task_agnostic_guarantees),
            },
        ]
    )

    assert merged == fragment


def _build_record(relative_path: str) -> MotifIrDocumentRecord:
    return MotifIrDocumentRecord(
        relative_path=relative_path,
        document=MotifMlIrDocument(
            metadata=IrDocumentMetadata(
                ir_schema_version="1.0.0",
                corpus_build_version="build-1",
                generator_version="0.1.0",
                source_document_hash="abc123",
            )
        ),
    )
