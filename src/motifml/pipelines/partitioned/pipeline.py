"""Pipelines for partitioned reducer stages."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ir_build.nodes import merge_ir_manifest_fragments
from motifml.pipelines.ir_validation.nodes import (
    merge_ir_shard_summaries,
    merge_ir_validation_report_fragments,
    report_ir_scale_metrics,
)
from motifml.pipelines.normalization.nodes import merge_normalized_ir_version_fragments
from motifml.pipelines.tokenization.nodes import (
    reduce_vocabulary_from_data_split_parameters,
)


def create_reduce_pipeline(**kwargs: object) -> Pipeline:
    """Create reducers for shard-level IR artifacts."""
    del kwargs

    return pipeline(
        [
            node(
                func=merge_ir_manifest_fragments,
                inputs="motif_ir_manifest_shard_collection",
                outputs="motif_ir_manifest",
                name="merge_ir_manifest_fragments",
            ),
            node(
                func=merge_ir_validation_report_fragments,
                inputs="motif_ir_validation_report_shard_collection",
                outputs="motif_ir_validation_report",
                name="merge_ir_validation_report_fragments",
            ),
            node(
                func=merge_ir_shard_summaries,
                inputs="motif_ir_summary_shard_collection",
                outputs="ir_corpus_summary_model",
                name="merge_ir_shard_summaries",
            ),
            node(
                func=report_ir_scale_metrics,
                inputs="ir_corpus_summary_model",
                outputs="motif_ir_summary",
                name="report_ir_scale_metrics",
            ),
            node(
                func=merge_normalized_ir_version_fragments,
                inputs="normalized_ir_version_shard_collection",
                outputs="normalized_ir_version",
                name="merge_normalized_ir_version_fragments",
            ),
            node(
                func=reduce_vocabulary_from_data_split_parameters,
                inputs=[
                    "token_count_shard_collection",
                    "params:vocabulary",
                    "params:data_split",
                ],
                outputs=["vocabulary", "vocab_stats", "vocabulary_version"],
                name="reduce_vocabulary_from_shard_counts",
            ),
        ],
        tags=["partitioned", "reduce"],
    )
