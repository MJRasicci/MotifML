"""Kedro pipeline registry for MotifML."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.pipelines.feature_extraction.pipeline import (
    create_pipeline as create_feature_extraction,
)
from motifml.pipelines.ingestion.pipeline import create_pipeline as create_ingestion
from motifml.pipelines.ir_build.pipeline import create_pipeline as create_ir_build
from motifml.pipelines.ir_validation.pipeline import (
    create_pipeline as create_ir_validation,
)
from motifml.pipelines.normalization.pipeline import (
    create_pipeline as create_normalization,
)
from motifml.pipelines.tokenization.pipeline import (
    create_pipeline as create_tokenization,
)


def _stage_raw_corpus_for_ir_build(
    documents: list[MotifJsonDocument],
    raw_corpus_summary: object,
) -> list[MotifJsonDocument]:
    """Gate IR build on completed ingestion without mutating the raw corpus."""
    del raw_corpus_summary
    return documents


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    ingestion = create_ingestion()
    ir_build = create_ir_build()
    ir_validation = create_ir_validation()
    normalization = create_normalization()
    feature_extraction = create_feature_extraction()
    tokenization = create_tokenization()
    staged_ir_build = pipeline(
        [
            node(
                func=_stage_raw_corpus_for_ir_build,
                inputs=["raw_motif_json_corpus", "raw_motif_json_summary"],
                outputs="ingested_raw_motif_json_corpus",
                name="stage_raw_corpus_for_ir_build",
            )
        ]
    ) + pipeline(
        ir_build,
        inputs={"raw_motif_json_corpus": "ingested_raw_motif_json_corpus"},
    )

    return {
        "ingestion": ingestion,
        "ir_build": ir_build,
        "ir_validation": ir_validation,
        "normalization": normalization,
        "feature_extraction": feature_extraction,
        "tokenization": tokenization,
        "__default__": (
            ingestion
            + staged_ir_build
            + ir_validation
            + normalization
            + feature_extraction
            + tokenization
        ),
    }
