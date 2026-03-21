"""Tests for deterministic training version-key utilities."""

from __future__ import annotations

from motifml.training.special_token_policy import SpecialTokenPolicy
from motifml.training.versioning import (
    build_feature_version,
    build_model_input_version,
    build_normalized_ir_version,
    build_split_version,
    build_vocabulary_version,
)


def test_normalized_ir_version_is_stable_and_dict_order_independent() -> None:
    first = build_normalized_ir_version(
        normalized_ir_contract={
            "contract_name": "motifml.normalized_ir",
            "contract_version": "1.0.0",
        },
        normalization_rules={"allow_optional_views": False, "strategy": "passthrough"},
    )
    second = build_normalized_ir_version(
        normalized_ir_contract={
            "contract_version": "1.0.0",
            "contract_name": "motifml.normalized_ir",
        },
        normalization_rules={"strategy": "passthrough", "allow_optional_views": False},
    )

    assert first == second


def test_normalized_ir_version_changes_when_contract_inputs_change() -> None:
    baseline = build_normalized_ir_version(
        normalized_ir_contract={"contract_name": "motifml.normalized_ir"},
        normalization_rules={"strategy": "passthrough"},
    )
    updated = build_normalized_ir_version(
        normalized_ir_contract={"contract_name": "motifml.normalized_ir"},
        normalization_rules={"strategy": "task_agnostic_cleanup_v2"},
    )

    assert updated != baseline


def test_feature_version_changes_only_for_feature_dependencies() -> None:
    baseline = build_feature_version(
        normalized_ir_version="normalized-v1",
        projection_config={"projection_type": "sequence", "sequence_mode": "baseline"},
        sequence_schema_version="sequence-schema-v1",
    )
    same = build_feature_version(
        normalized_ir_version="normalized-v1",
        projection_config={"sequence_mode": "baseline", "projection_type": "sequence"},
        sequence_schema_version="sequence-schema-v1",
    )
    updated = build_feature_version(
        normalized_ir_version="normalized-v2",
        projection_config={"projection_type": "sequence", "sequence_mode": "baseline"},
        sequence_schema_version="sequence-schema-v1",
    )

    assert baseline == same
    assert updated != baseline


def test_split_version_is_order_independent_for_corpus_membership() -> None:
    first = build_split_version(
        corpus_membership=("fixtures/b.json", "fixtures/a.json"),
        split_config={"ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}},
        split_seed=17,
    )
    second = build_split_version(
        corpus_membership=("fixtures/a.json", "fixtures/b.json"),
        split_config={"ratios": {"test": 0.1, "train": 0.8, "validation": 0.1}},
        split_seed=17,
    )
    changed = build_split_version(
        corpus_membership=("fixtures/a.json", "fixtures/c.json"),
        split_config={"ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}},
        split_seed=17,
    )

    assert first == second
    assert changed != first


def test_vocabulary_version_changes_only_for_vocabulary_dependencies() -> None:
    policy = SpecialTokenPolicy().to_version_payload()
    baseline = build_vocabulary_version(
        feature_version="feature-v1",
        tokenization_config={"minimum_frequency": 2, "time_resolution": 96},
        split_version="split-v1",
        split_seed=17,
        special_token_policy=policy,
    )
    repeated = build_vocabulary_version(
        feature_version="feature-v1",
        tokenization_config={"time_resolution": 96, "minimum_frequency": 2},
        split_version="split-v1",
        split_seed=17,
        special_token_policy={
            "unknown_token_mapping": "map_to_unk",
            "eos": "document",
            "policy_mode": "baseline_v1",
            "bos": "document",
            "padding_interaction": "outside_boundaries",
            "policy_name": "baseline_special_tokens",
        },
    )
    changed = build_vocabulary_version(
        feature_version="feature-v1",
        tokenization_config={"minimum_frequency": 3, "time_resolution": 96},
        split_version="split-v1",
        split_seed=17,
        special_token_policy=policy,
    )

    assert baseline == repeated
    assert changed != baseline


def test_model_input_version_changes_only_for_model_input_dependencies() -> None:
    policy = SpecialTokenPolicy().to_version_payload()
    baseline = build_model_input_version(
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_config={
            "context_length": 256,
            "stride": 128,
            "padding_strategy": "right",
        },
        special_token_policy=policy,
        storage_schema_version="parquet-v1",
    )
    repeated = build_model_input_version(
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_config={
            "stride": 128,
            "context_length": 256,
            "padding_strategy": "right",
        },
        special_token_policy={
            "unknown_token_mapping": "map_to_unk",
            "eos": "document",
            "policy_mode": "baseline_v1",
            "bos": "document",
            "padding_interaction": "outside_boundaries",
            "policy_name": "baseline_special_tokens",
        },
        storage_schema_version="parquet-v1",
    )
    changed = build_model_input_version(
        feature_version="feature-v1",
        vocabulary_version="vocab-v1",
        model_input_config={
            "context_length": 512,
            "stride": 128,
            "padding_strategy": "right",
        },
        special_token_policy=policy,
        storage_schema_version="parquet-v1",
    )

    assert baseline == repeated
    assert changed != baseline


def test_versions_do_not_cross_contaminate_dependency_surfaces() -> None:
    policy = SpecialTokenPolicy().to_version_payload()
    feature_baseline = build_feature_version(
        normalized_ir_version="normalized-v1",
        projection_config={"projection_type": "sequence"},
        sequence_schema_version="schema-v1",
    )
    feature_repeated = build_feature_version(
        normalized_ir_version="normalized-v1",
        projection_config={"projection_type": "sequence"},
        sequence_schema_version="schema-v1",
    )
    vocabulary_baseline = build_vocabulary_version(
        feature_version=feature_baseline,
        tokenization_config={"minimum_frequency": 2},
        split_version="split-v1",
        split_seed=7,
        special_token_policy=policy,
    )
    vocabulary_changed = build_vocabulary_version(
        feature_version=feature_repeated,
        tokenization_config={"minimum_frequency": 3},
        split_version="split-v1",
        split_seed=7,
        special_token_policy=policy,
    )

    assert feature_baseline == feature_repeated
    assert vocabulary_baseline != vocabulary_changed
