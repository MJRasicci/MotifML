"""Nodes for deterministic experiment split assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.pipelines.dataset_splitting.models import (
    DataSplitParameters,
    GroupingStrategy,
    coerce_data_split_parameters,
)
from motifml.training.contracts import (
    DatasetSplit,
    SplitManifestEntry,
    sort_split_manifest_entries,
)
from motifml.training.versioning import build_split_version


@dataclass(frozen=True, slots=True)
class _SplitCandidate:
    document_id: str
    relative_path: str
    group_key: str


def assign_dataset_splits(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    parameters: DataSplitParameters | Mapping[str, Any],
) -> tuple[SplitManifestEntry, ...]:
    """Assign train/validation/test labels independently of execution shards."""
    typed_parameters = coerce_data_split_parameters(parameters)
    candidates = _build_candidates(normalized_ir_corpus, typed_parameters)
    split_version = build_split_version(
        corpus_membership=tuple(candidate.relative_path for candidate in candidates),
        split_config=_split_config_payload(typed_parameters),
        split_seed=typed_parameters.hash_seed,
    )
    split_by_group = {
        group_key: _assign_group_split(group_key, typed_parameters)
        for group_key in sorted(
            {candidate.group_key for candidate in candidates}, key=str.casefold
        )
    }
    return sort_split_manifest_entries(
        tuple(
            SplitManifestEntry(
                document_id=candidate.document_id,
                relative_path=candidate.relative_path,
                split=split_by_group[candidate.group_key],
                group_key=candidate.group_key,
                split_version=split_version,
            )
            for candidate in candidates
        )
    )


def _build_candidates(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    parameters: DataSplitParameters,
) -> tuple[_SplitCandidate, ...]:
    seen_relative_paths: set[str] = set()
    candidates: list[_SplitCandidate] = []
    for record in sorted(
        normalized_ir_corpus, key=lambda item: item.relative_path.casefold()
    ):
        relative_path = _normalize_text(record.relative_path, "relative_path")
        if relative_path in seen_relative_paths:
            raise ValueError(
                "normalized_ir_corpus contains duplicate relative_path entries: "
                f"{relative_path}"
            )
        seen_relative_paths.add(relative_path)

        document_id = relative_path
        group_key = _group_key_for_candidate(
            document_id=document_id,
            relative_path=relative_path,
            parameters=parameters,
        )
        candidates.append(
            _SplitCandidate(
                document_id=document_id,
                relative_path=relative_path,
                group_key=group_key,
            )
        )
    return tuple(candidates)


def _assign_group_split(
    group_key: str,
    parameters: DataSplitParameters,
) -> DatasetSplit:
    normalized = parameters.ratios.normalized()
    score = _hash_fraction(f"{parameters.hash_seed}:{group_key}")
    if score < normalized.train:
        return DatasetSplit.TRAIN
    if score < normalized.train + normalized.validation:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def _group_key_for_candidate(
    *,
    document_id: str,
    relative_path: str,
    parameters: DataSplitParameters,
) -> str:
    primary = _strategy_value(
        strategy=parameters.grouping_strategy,
        document_id=document_id,
        relative_path=relative_path,
    )
    if primary:
        return primary

    fallback = _strategy_value(
        strategy=parameters.grouping_key_fallback,
        document_id=document_id,
        relative_path=relative_path,
    )
    if fallback:
        return fallback

    raise ValueError(
        "Unable to derive a grouping key from the configured grouping strategy "
        f"'{parameters.grouping_strategy.value}' or fallback "
        f"'{parameters.grouping_key_fallback.value}'."
    )


def _strategy_value(
    *,
    strategy: GroupingStrategy,
    document_id: str,
    relative_path: str,
) -> str:
    if strategy is GroupingStrategy.DOCUMENT_ID:
        return _normalize_optional_text(document_id)
    if strategy is GroupingStrategy.RELATIVE_PATH:
        return _normalize_optional_text(relative_path)
    if strategy is GroupingStrategy.PARENT_DIRECTORY:
        parent = Path(relative_path).parent.as_posix()
        if parent in {"", "."}:
            return ""
        return _normalize_optional_text(parent)
    raise ValueError(f"Unsupported grouping strategy: {strategy.value}")


def _split_config_payload(parameters: DataSplitParameters) -> dict[str, Any]:
    return {
        "ratios": {
            "train": parameters.ratios.train,
            "validation": parameters.ratios.validation,
            "test": parameters.ratios.test,
        },
        "grouping_strategy": parameters.grouping_strategy.value,
        "grouping_key_fallback": parameters.grouping_key_fallback.value,
    }


def _hash_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16) / float(2**256)


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_optional_text(value: str) -> str:
    return str(value).strip()


__all__ = ["assign_dataset_splits"]
