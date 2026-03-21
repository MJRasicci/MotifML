"""Tests for training configuration surfaces and frozen parameter snapshots."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from motifml.training.config import (
    build_parameter_snapshots,
    freeze_parameter_snapshot,
)
from motifml.training.special_token_policy import (
    SpecialTokenPolicy,
    coerce_special_token_policy,
)

PARAMETERS_PATH = "conf/base/parameters.yml"
OVERRIDE_BATCH_SIZE = 16
OVERRIDE_TOP_K = 10
OVERRIDE_CONTEXT_LENGTH = 384
EXPECTED_MINIMUM_VOCABULARY_SIZE = 7


def test_parameters_yaml_defines_all_training_parameter_families() -> None:
    parameters = _load_parameters()

    assert "data_split" in parameters
    assert "sequence_schema" in parameters
    assert "vocabulary" in parameters
    assert "model_input" in parameters
    assert "model" in parameters
    assert "training" in parameters
    assert "evaluation" in parameters
    assert "seed" in parameters


def test_freeze_parameter_snapshot_serializes_in_stable_key_order() -> None:
    snapshot = freeze_parameter_snapshot(
        {
            "z": 3,
            "a": {"y": True, "x": False},
            "list": [{"b": 2, "a": 1}],
        }
    )

    assert list(snapshot) == ["a", "list", "z"]
    assert list(snapshot["a"]) == ["x", "y"]
    assert list(snapshot["list"][0]) == ["a", "b"]


def test_build_parameter_snapshots_reflects_overrides_without_cross_contamination() -> (
    None
):
    baseline_parameters = _load_parameters()
    overridden_parameters = deepcopy(baseline_parameters)
    overridden_parameters["training"]["batch_size"] = OVERRIDE_BATCH_SIZE
    overridden_parameters["evaluation"]["top_k"] = OVERRIDE_TOP_K
    overridden_parameters["model_input"]["context_length"] = OVERRIDE_CONTEXT_LENGTH

    baseline = build_parameter_snapshots(baseline_parameters)
    overridden = build_parameter_snapshots(overridden_parameters)

    assert overridden.training_run["training"]["batch_size"] == OVERRIDE_BATCH_SIZE
    assert overridden.evaluation_run["evaluation"]["top_k"] == OVERRIDE_TOP_K
    assert (
        overridden.model_input_persistence["model_input"]["context_length"]
        == OVERRIDE_CONTEXT_LENGTH
    )
    assert overridden.split_generation == baseline.split_generation
    assert overridden.vocabulary_construction == baseline.vocabulary_construction


def test_build_parameter_snapshots_rejects_missing_baseline_families() -> None:
    parameters = _load_parameters()
    del parameters["vocabulary"]

    with pytest.raises(ValueError, match="Missing: vocabulary"):
        build_parameter_snapshots(parameters)


def test_parameters_yaml_uses_the_canonical_special_token_policy_shape() -> None:
    parameters = _load_parameters()

    assert (
        coerce_special_token_policy(parameters["model_input"]["special_token_policy"])
        == SpecialTokenPolicy()
    )


def test_parameters_yaml_defines_vocabulary_guardrails() -> None:
    parameters = _load_parameters()
    guardrails = parameters["vocabulary"]["guardrails"]

    assert guardrails["minimum_vocabulary_size"] == EXPECTED_MINIMUM_VOCABULARY_SIZE
    assert guardrails["required_token_families"] == [
        "NOTE_DURATION",
        "NOTE_PITCH",
        "STRUCTURE",
    ]


def _load_parameters() -> dict[str, object]:
    with open(PARAMETERS_PATH, encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    return dict(loaded)
