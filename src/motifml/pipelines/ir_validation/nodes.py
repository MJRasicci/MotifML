"""Nodes for validating canonical IR documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.validation import (
    IrDocumentValidationReport,
    build_document_validation_report,
)


def validate_ir_documents(
    documents: list[MotifIrDocumentRecord],
    ir_validation: Mapping[str, Any] | None = None,
) -> list[IrDocumentValidationReport]:
    """Validate IR documents in deterministic path order."""
    rule_severities = _resolve_rule_severities(ir_validation)
    return [
        build_document_validation_report(
            relative_path=record.relative_path,
            source_hash=record.document.metadata.source_document_hash,
            document=record.document,
            rule_severities=rule_severities,
        )
        for record in sorted(documents, key=lambda item: item.relative_path.casefold())
    ]


def _resolve_rule_severities(
    ir_validation: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if ir_validation is None:
        return None
    if not isinstance(ir_validation, Mapping):
        raise ValueError("ir_validation parameters must be a mapping.")

    rule_severities = ir_validation.get("rule_severities")
    if rule_severities is None:
        return None
    if not isinstance(rule_severities, Mapping):
        raise ValueError("ir_validation.rule_severities must be a mapping.")

    return rule_severities
