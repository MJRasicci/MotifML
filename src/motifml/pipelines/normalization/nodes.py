"""Nodes for the IR normalization pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.serialization import serialize_document
from motifml.pipelines.normalization.models import (
    NormalizationParameters,
    NormalizedIrVersionMetadata,
    coerce_normalization_parameters,
    coerce_normalized_ir_version_metadata,
)
from motifml.training.versioning import (
    build_normalized_ir_version as build_normalized_ir_contract_version,
)


def normalize_ir_corpus(
    documents: list[MotifIrDocumentRecord],
) -> list[MotifIrDocumentRecord]:
    """Return the canonical IR corpus unchanged.

    The current normalization baseline is a typed passthrough so downstream pipelines can
    operate without changing the canonical IR surface.
    """
    return documents


def build_normalized_ir_version(
    documents: list[MotifIrDocumentRecord],
    parameters: NormalizationParameters | Mapping[str, Any],
) -> NormalizedIrVersionMetadata:
    """Build the persisted normalized-IR version metadata artifact."""
    typed_parameters = coerce_normalization_parameters(parameters)
    validate_normalized_ir_contract(documents, typed_parameters)
    upstream_schema_versions = sorted(
        {record.document.metadata.ir_schema_version for record in documents}
    )
    if not upstream_schema_versions:
        raise ValueError("normalized_ir_corpus must contain at least one document.")
    if len(upstream_schema_versions) != 1:
        joined_versions = ", ".join(upstream_schema_versions)
        raise ValueError(
            "normalized_ir_corpus must agree on one upstream ir_schema_version, "
            f"but received: {joined_versions}."
        )

    upstream_ir_schema_version = upstream_schema_versions[0]
    normalized_ir_version = build_normalized_ir_version_key(
        parameters=typed_parameters,
        upstream_ir_schema_version=upstream_ir_schema_version,
    )
    return NormalizedIrVersionMetadata(
        normalized_ir_version=normalized_ir_version,
        contract_name=typed_parameters.contract_name,
        contract_version=typed_parameters.contract_version,
        serialized_document_format=typed_parameters.serialized_document_format,
        normalization_strategy=typed_parameters.normalization_strategy,
        upstream_ir_schema_version=upstream_ir_schema_version,
        task_agnostic_guarantees=typed_parameters.task_agnostic_guarantees,
    )


def validate_normalized_ir_contract(
    documents: list[MotifIrDocumentRecord],
    parameters: NormalizationParameters | Mapping[str, Any],
) -> None:
    """Fail fast if normalized IR artifacts leak training-specific fields."""
    typed_parameters = coerce_normalization_parameters(parameters)
    forbidden_fields = frozenset(typed_parameters.forbidden_model_fields)

    for record in documents:
        serialized_payload = json.loads(serialize_document(record.document))
        for field_path, field_name in _iter_field_paths(serialized_payload):
            if field_name in forbidden_fields:
                raise ValueError(
                    "Normalized IR contract violation for "
                    f"'{record.relative_path}': forbidden field '{field_name}' at "
                    f"'{field_path}'."
                )


def merge_normalized_ir_version_fragments(
    fragments: list[NormalizedIrVersionMetadata] | list[Mapping[str, Any]],
) -> NormalizedIrVersionMetadata:
    """Merge shard-local normalized-IR version fragments into one global artifact."""
    typed_fragments = [
        coerce_normalized_ir_version_metadata(item) for item in fragments
    ]
    if not typed_fragments:
        raise ValueError("No normalized_ir_version fragments were provided.")

    first_fragment = typed_fragments[0]
    for fragment in typed_fragments[1:]:
        if fragment != first_fragment:
            raise ValueError(
                "All normalized_ir_version fragments must be identical across shards."
            )

    return first_fragment


def build_normalized_ir_version_key(
    *,
    parameters: NormalizationParameters,
    upstream_ir_schema_version: str,
) -> str:
    """Build the normalized-IR version key from the frozen contract inputs."""
    return build_normalized_ir_contract_version(
        normalized_ir_contract={
            "contract_name": parameters.contract_name,
            "contract_version": parameters.contract_version,
            "serialized_document_format": parameters.serialized_document_format,
            "upstream_ir_schema_version": upstream_ir_schema_version,
        },
        normalization_rules={
            "normalization_strategy": parameters.normalization_strategy,
            "allow_optional_overlays": parameters.allow_optional_overlays,
            "allow_optional_views": parameters.allow_optional_views,
            "task_agnostic_guarantees": parameters.task_agnostic_guarantees,
        },
    )


def _iter_field_paths(
    value: object,
    *,
    path: str = "$",
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, nested_value in value.items():
            field_name = str(key)
            current_path = f"{path}.{field_name}"
            matches.append((current_path, field_name))
            matches.extend(_iter_field_paths(nested_value, path=current_path))
        return matches

    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            current_path = f"{path}[{index}]"
            matches.extend(_iter_field_paths(nested_value, path=current_path))

    return matches
