"""Nodes for the IR normalization pipeline."""

from __future__ import annotations

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord


def normalize_ir_corpus(
    documents: list[MotifIrDocumentRecord],
) -> list[MotifIrDocumentRecord]:
    """Return the canonical IR corpus unchanged.

    The current normalization baseline is a typed passthrough so downstream pipelines can
    operate without changing the canonical IR surface.
    """
    return documents
