"""Pipeline definition for raw Motif JSON ingestion."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ingestion.nodes import (
    build_raw_corpus_manifest,
    build_raw_partition_index,
    build_raw_shard_manifests,
    summarize_raw_corpus,
)


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the raw corpus ingestion pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=build_raw_corpus_manifest,
                inputs="raw_motif_json_corpus",
                outputs="raw_motif_json_manifest",
                name="build_raw_corpus_manifest",
            ),
            node(
                func=summarize_raw_corpus,
                inputs="raw_motif_json_manifest",
                outputs="raw_motif_json_summary",
                name="summarize_raw_corpus",
            ),
            node(
                func=build_raw_partition_index,
                inputs=["raw_motif_json_manifest", "params:partitioning"],
                outputs="raw_partition_index",
                name="build_raw_partition_index",
            ),
            node(
                func=build_raw_shard_manifests,
                inputs="raw_partition_index",
                outputs="raw_shard_manifests",
                name="build_raw_shard_manifests",
            ),
        ],
        tags=["ingestion"],
    )
