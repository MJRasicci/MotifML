"""Unit tests for the normalization pipeline skeleton."""

from __future__ import annotations

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.models import IrDocumentMetadata, MotifMlIrDocument
from motifml.pipelines.normalization.nodes import normalize_ir_corpus


def test_normalize_ir_corpus_returns_the_original_document_list() -> None:
    records = [
        MotifIrDocumentRecord(
            relative_path="examples/demo.json",
            document=MotifMlIrDocument(
                metadata=IrDocumentMetadata(
                    ir_schema_version="1.0.0",
                    corpus_build_version="build-1",
                    generator_version="0.1.0",
                    source_document_hash="abc123",
                )
            ),
        )
    ]

    normalized = normalize_ir_corpus(records)

    assert normalized is records
