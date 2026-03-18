"""Pipeline definition for raw Motif JSON ingestion."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.ingestion.nodes import (
    build_raw_corpus_manifest,
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
        ],
        tags=["ingestion"],
    )
