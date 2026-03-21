"""Shared training-contract utilities for MotifML."""

from motifml.training.contracts import (
    DatasetSplit,
    EvaluationRunMetadata,
    ModelInputMetadata,
    SplitManifestEntry,
    TrainingRunMetadata,
    VocabularyMetadata,
    deserialize_metadata_artifact,
    serialize_metadata_artifact,
)
from motifml.training.versioning import (
    build_contract_version,
    build_feature_version,
    build_model_input_version,
    build_normalized_ir_version,
    build_split_version,
    build_vocabulary_version,
)

__all__ = [
    "DatasetSplit",
    "EvaluationRunMetadata",
    "ModelInputMetadata",
    "SplitManifestEntry",
    "TrainingRunMetadata",
    "VocabularyMetadata",
    "build_contract_version",
    "build_feature_version",
    "build_model_input_version",
    "build_normalized_ir_version",
    "build_split_version",
    "build_vocabulary_version",
    "deserialize_metadata_artifact",
    "serialize_metadata_artifact",
]
