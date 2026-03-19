"""Nodes for validating canonical IR documents."""

from __future__ import annotations

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.validation import (
    IrDocumentValidationReport,
    build_document_validation_report,
)


def validate_ir_documents(
    documents: list[MotifIrDocumentRecord],
) -> list[IrDocumentValidationReport]:
    """Validate IR documents in deterministic path order."""
    return [
        build_document_validation_report(
            relative_path=record.relative_path,
            source_hash=record.document.metadata.source_document_hash,
            document=record.document,
        )
        for record in sorted(documents, key=lambda item: item.relative_path.casefold())
    ]
