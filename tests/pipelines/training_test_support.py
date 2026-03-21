"""Shared helpers for training fixture generation and regression tests."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from tests.pipelines.ir_test_support import (
    load_json,
    load_text,
    materialize_raw_fixture_subset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "training"
TRAINING_FIXTURE_CATALOG_PATH = TRAINING_FIXTURE_ROOT / "training_fixture_catalog.json"
SMOKE_BUNDLE_ROOT = TRAINING_FIXTURE_ROOT / "smoke_bundle"
NORMALIZED_TIMESTAMP = "<normalized>"
FLOAT_COMPARISON_TOLERANCE = 1e-6


@lru_cache(maxsize=1)
def load_training_fixture_catalog() -> dict[str, Any]:
    """Load the tracked training fixture catalog."""
    return json.loads(TRAINING_FIXTURE_CATALOG_PATH.read_text(encoding="utf-8"))


def training_fixture_entries() -> list[dict[str, Any]]:
    """Return the approved tiny training fixture slice entries."""
    return list(load_training_fixture_catalog()["fixtures"])


def training_fixture_raw_paths() -> tuple[str, ...]:
    """Return the raw Motif JSON paths for the approved training fixture slice."""
    return tuple(
        str(entry["raw_motif_json_path"]) for entry in training_fixture_entries()
    )


def materialize_training_fixture_corpus(destination: Path) -> Path:
    """Copy the approved training fixture slice into a temporary raw corpus root."""
    return materialize_raw_fixture_subset(
        destination, list(training_fixture_raw_paths())
    )


def baseline_training_runtime_overrides() -> dict[str, Any]:
    """Return the tiny CPU training overrides used by training fixture tests."""
    return copy.deepcopy(
        {
            "data_split": {
                "ratios": {
                    "train": 0.5,
                    "validation": 0.25,
                    "test": 0.25,
                },
                "hash_seed": 17,
                "grouping_strategy": "document_id",
                "grouping_key_fallback": "relative_path",
            },
            "model_input": {
                "projection_type": "sequence",
                "sequence_mode": "baseline_v1",
                "context_length": 64,
                "stride": 32,
                "padding_strategy": "right",
                "special_token_policy": {
                    "bos": "document",
                    "eos": "document",
                    "padding_interaction": "outside_boundaries",
                    "unknown_token_mapping": "map_to_unk",
                },
                "storage": {
                    "backend": "parquet",
                    "schema_version": "parquet-v1",
                },
                "reporting": {
                    "worst_document_limit": 10,
                    "oversized_token_count_threshold": 8192,
                },
            },
            "model": {
                "architecture": "decoder_only_transformer",
                "embedding_dim": 32,
                "hidden_size": 64,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
                "positional_encoding": "learned",
            },
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "num_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "optimizer": "adamw",
                "lr_scheduler": {
                    "name": "constant",
                    "warmup_steps": 0,
                },
            },
        }
    )


def baseline_evaluation_runtime_overrides() -> dict[str, Any]:
    """Return the tiny CPU training-plus-evaluation overrides used by fixtures."""
    overrides = baseline_training_runtime_overrides()
    overrides["evaluation"] = {
        "device": "cpu",
        "batch_size": 2,
        "top_k": 3,
        "decode_max_tokens": 4,
        "splits": ["validation"],
        "qualitative": {
            "samples_per_split": 1,
            "prompt_token_count": 4,
            "summary_token_limit": 3,
        },
        "guardrails": {
            "maximum_split_unk_rate": 0.5,
            "maximum_generated_unk_rate": 0.5,
        },
    }
    return overrides


def build_normalized_smoke_bundle(
    output_root: Path,
    *,
    runtime_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Collect one normalized training/evaluation smoke bundle from pipeline outputs."""
    training_run_metadata = _normalize_timestamp_field(
        load_json(output_root / "training_run_metadata.json"),
        "started_at",
    )
    evaluation_run_metadata = _normalize_timestamp_field(
        load_json(output_root / "evaluation_run_metadata.json"),
        "started_at",
    )
    metrics = _normalize_metrics_payload(load_json(output_root / "metrics.json"))

    return {
        "frozen_config.json": {
            "seed": training_run_metadata["seed"],
            "data_split": copy.deepcopy(runtime_overrides["data_split"]),
            "model_input": copy.deepcopy(runtime_overrides["model_input"]),
            "model": copy.deepcopy(training_run_metadata["model_parameters"]),
            "training": copy.deepcopy(runtime_overrides["training"]),
            "evaluation": copy.deepcopy(
                evaluation_run_metadata["evaluation_parameters"]
            ),
        },
        "training_history.json": load_json(output_root / "training_history.json"),
        "training_run_metadata.json": training_run_metadata,
        "evaluation_run_metadata.json": evaluation_run_metadata,
        "metrics.json": metrics,
        "evaluation/qualitative_samples.json": load_json(
            output_root / "evaluation" / "qualitative_samples.json"
        ),
        "qualitative_report.md": load_text(output_root / "qualitative_report.md"),
    }


def load_tracked_smoke_bundle() -> dict[str, Any]:
    """Load the tracked normalized smoke bundle from disk."""
    return {
        "frozen_config.json": load_json(SMOKE_BUNDLE_ROOT / "frozen_config.json"),
        "training_history.json": load_json(SMOKE_BUNDLE_ROOT / "training_history.json"),
        "training_run_metadata.json": load_json(
            SMOKE_BUNDLE_ROOT / "training_run_metadata.json"
        ),
        "evaluation_run_metadata.json": load_json(
            SMOKE_BUNDLE_ROOT / "evaluation_run_metadata.json"
        ),
        "metrics.json": load_json(SMOKE_BUNDLE_ROOT / "metrics.json"),
        "evaluation/qualitative_samples.json": load_json(
            SMOKE_BUNDLE_ROOT / "evaluation" / "qualitative_samples.json"
        ),
        "qualitative_report.md": load_text(SMOKE_BUNDLE_ROOT / "qualitative_report.md"),
    }


def assert_nested_close(actual: Any, expected: Any) -> None:
    """Compare nested JSON-compatible values using tolerant float assertions."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            assert_nested_close(actual[key], expected[key])
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_nested_close(actual_item, expected_item)
        return
    if isinstance(expected, float):
        assert isinstance(actual, int | float)
        assert abs(float(actual) - expected) <= FLOAT_COMPARISON_TOLERANCE
        return
    assert actual == expected


def _normalize_timestamp_field(
    payload: dict[str, Any], field_name: str
) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    if field_name in normalized:
        normalized[field_name] = NORMALIZED_TIMESTAMP
    return normalized


def _normalize_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    best_checkpoint = normalized.get("best_checkpoint")
    if isinstance(best_checkpoint, dict) and "saved_at" in best_checkpoint:
        best_checkpoint["saved_at"] = NORMALIZED_TIMESTAMP
    return normalized


__all__ = [
    "NORMALIZED_TIMESTAMP",
    "SMOKE_BUNDLE_ROOT",
    "assert_nested_close",
    "build_normalized_smoke_bundle",
    "TRAINING_FIXTURE_CATALOG_PATH",
    "TRAINING_FIXTURE_ROOT",
    "baseline_evaluation_runtime_overrides",
    "baseline_training_runtime_overrides",
    "load_tracked_smoke_bundle",
    "load_training_fixture_catalog",
    "materialize_training_fixture_corpus",
    "training_fixture_entries",
    "training_fixture_raw_paths",
]
