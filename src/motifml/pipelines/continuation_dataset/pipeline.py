"""Pipeline definition for V1 continuation-example extraction."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.continuation_dataset.nodes import extract_continuation_examples


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the continuation dataset extraction pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=extract_continuation_examples,
                inputs=[
                    "normalized_ir_corpus",
                    "split_manifest",
                    "params:continuation_dataset",
                    "normalized_ir_version",
                ],
                outputs=["v1_continuation_examples", "v1_continuation_summary"],
                name="extract_continuation_examples",
            )
        ],
        tags=["continuation_dataset", "training_prep"],
    )


__all__ = ["create_pipeline"]
