"""Deterministic version-key helpers for MotifML training artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def build_normalized_ir_version(
    *,
    normalized_ir_contract: Mapping[str, Any],
    normalization_rules: Mapping[str, Any] | None = None,
) -> str:
    """Build the normalized-IR contract version key."""
    return _hash_version_payload(
        "normalized_ir_version",
        {
            "normalized_ir_contract": normalized_ir_contract,
            "normalization_rules": normalization_rules or {},
        },
    )


def build_feature_version(
    *,
    normalized_ir_version: str,
    projection_config: Mapping[str, Any],
    sequence_schema_version: str,
) -> str:
    """Build the feature projection version key."""
    return _hash_version_payload(
        "feature_version",
        {
            "normalized_ir_version": normalized_ir_version,
            "projection_config": projection_config,
            "sequence_schema_version": sequence_schema_version,
        },
    )


def build_split_version(
    *,
    corpus_membership: Sequence[str],
    split_config: Mapping[str, Any],
    split_seed: int | str,
) -> str:
    """Build the experiment split manifest version key."""
    return _hash_version_payload(
        "split_version",
        {
            "corpus_membership": tuple(
                sorted(str(value) for value in corpus_membership)
            ),
            "split_config": split_config,
            "split_seed": str(split_seed),
        },
    )


def build_vocabulary_version(
    *,
    feature_version: str,
    tokenization_config: Mapping[str, Any],
    split_version: str,
    split_seed: int | str,
    special_token_policy: Mapping[str, Any],
) -> str:
    """Build the frozen vocabulary version key."""
    return _hash_version_payload(
        "vocabulary_version",
        {
            "feature_version": feature_version,
            "tokenization_config": tokenization_config,
            "split_version": split_version,
            "split_seed": str(split_seed),
            "special_token_policy": special_token_policy,
        },
    )


def build_model_input_version(
    *,
    feature_version: str,
    vocabulary_version: str,
    model_input_config: Mapping[str, Any],
    special_token_policy: Mapping[str, Any],
    storage_schema_version: str,
) -> str:
    """Build the model-input artifact version key."""
    return _hash_version_payload(
        "model_input_version",
        {
            "feature_version": feature_version,
            "vocabulary_version": vocabulary_version,
            "model_input_config": model_input_config,
            "special_token_policy": special_token_policy,
            "storage_schema_version": storage_schema_version,
        },
    )


def _hash_version_payload(namespace: str, payload: Mapping[str, Any]) -> str:
    serialized_payload = json.dumps(
        _canonicalize_for_hash({"namespace": namespace, "payload": payload}),
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def _canonicalize_for_hash(value: Any) -> Any:  # noqa: PLR0911
    if is_dataclass(value):
        return _canonicalize_for_hash(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_for_hash(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, set | frozenset):
        canonical_items = [_canonicalize_for_hash(item) for item in value]
        return sorted(canonical_items, key=_sort_key)

    if isinstance(value, list | tuple):
        return [_canonicalize_for_hash(item) for item in value]

    return value


def _sort_key(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, sort_keys=True)


__all__ = [
    "build_feature_version",
    "build_model_input_version",
    "build_normalized_ir_version",
    "build_split_version",
    "build_vocabulary_version",
]
