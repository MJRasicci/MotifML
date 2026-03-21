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
from motifml.pipelines.partitioned.pipeline import create_reduce_pipeline
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
    """Register the project's pipelines."""
    ingestion = create_ingestion()
    ir_build = create_ir_build()
    ir_validation = create_ir_validation()
    normalization = create_normalization()
    feature_extraction = create_feature_extraction()
    tokenization = create_tokenization()
    partitioned_reduce = create_reduce_pipeline()

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

    ir_build_shard = pipeline(
        ir_build,
        inputs={"raw_motif_json_corpus": "raw_motif_json_corpus_shard"},
        outputs={
            "motif_ir_corpus": "motif_ir_corpus_shard",
            "motif_ir_manifest": "motif_ir_manifest_shard",
        },
    )
    normalization_shard = pipeline(
        normalization,
        inputs={"motif_ir_corpus": "motif_ir_corpus_shard"},
        outputs={
            "normalized_ir_corpus": "normalized_ir_corpus_shard",
            "normalized_ir_version": "normalized_ir_version_shard",
        },
    )
    ir_validation_shard = pipeline(
        ir_validation,
        inputs={
            "motif_ir_corpus": "motif_ir_corpus_shard",
            "motif_ir_manifest": "motif_ir_manifest_shard",
        },
        outputs={
            "motif_ir_validation_report": "motif_ir_validation_report_shard",
            "motif_ir_summary": "motif_ir_summary_shard",
        },
    )
    feature_extraction_shard = pipeline(
        feature_extraction,
        inputs={
            "normalized_ir_corpus": "normalized_ir_corpus_shard",
            "normalized_ir_version": "normalized_ir_version_shard",
        },
        outputs={"ir_features": "ir_features_shard"},
    )
    tokenization_shard = pipeline(
        tokenization,
        inputs={"ir_features": "ir_features_shard"},
        outputs={"model_input": "model_input_shard"},
    )
    shard_processing = (
        ir_build_shard
        + normalization_shard
        + ir_validation_shard
        + feature_extraction_shard
        + tokenization_shard
    )

    return {
        "ingestion": ingestion,
        "partition_ingestion": ingestion,
        "partitioned_ingestion": ingestion,
        "ir_build": ir_build,
        "ir_validation": ir_validation,
        "normalization": normalization,
        "feature_extraction": feature_extraction,
        "tokenization": tokenization,
        "ir_build_shard": ir_build_shard,
        "ir_validation_shard": ir_validation_shard,
        "normalization_shard": normalization_shard,
        "feature_extraction_shard": feature_extraction_shard,
        "tokenization_shard": tokenization_shard,
        "partitioned_reduce": partitioned_reduce,
        "shard_reduce": partitioned_reduce,
        "shard_processing": shard_processing,
        "__default__": (
            ingestion
            + staged_ir_build
            + ir_validation
            + normalization
            + feature_extraction
            + tokenization
        ),
    }
