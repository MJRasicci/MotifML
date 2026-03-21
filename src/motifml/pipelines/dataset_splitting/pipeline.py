"""Pipeline definition for deterministic experiment splitting."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from motifml.pipelines.dataset_splitting.nodes import assign_dataset_splits


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the dataset splitting pipeline."""
    del kwargs

    return pipeline(
        [
            node(
                func=assign_dataset_splits,
                inputs=["normalized_ir_corpus", "params:data_split"],
                outputs="split_manifest",
                name="assign_dataset_splits",
            )
        ],
        tags=["dataset_splitting", "training_prep"],
    )
