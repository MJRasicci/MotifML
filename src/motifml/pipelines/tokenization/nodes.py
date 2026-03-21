"""Nodes for the tokenization pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from motifml.pipelines.feature_extraction.models import (
    IrFeatureRecord,
    IrFeatureSet,
    ProjectionType,
)
from motifml.pipelines.tokenization.models import (
    ModelInputRecord,
    ModelInputSet,
    PaddingStrategy,
    TokenizationParameters,
    coerce_tokenization_parameters,
)
from motifml.training.token_families import PAD_TOKEN


@dataclass(frozen=True)
class _FeatureRecordInput:
    relative_path: str
    projection_type: ProjectionType
    projection: object


def tokenize_features(
    ir_features: IrFeatureSet | Mapping[str, Any],
    parameters: TokenizationParameters | Mapping[str, Any],
) -> ModelInputSet:
    """Convert projected features into deterministic model-input records."""
    typed_parameters = coerce_tokenization_parameters(parameters)
    records = tuple(
        _tokenize_record(record, typed_parameters)
        for record in _iter_feature_records(ir_features)
    )
    return ModelInputSet(parameters=typed_parameters, records=records)


def merge_model_input_shards(
    model_input_shards: list[ModelInputSet] | list[Mapping[str, Any]],
) -> ModelInputSet:
    """Merge shard-local model-input sets into one global model-input set."""
    typed_shards = [_coerce_model_input_set(shard) for shard in model_input_shards]
    if not typed_shards:
        return ModelInputSet(parameters=TokenizationParameters())

    parameters = typed_shards[0].parameters
    for shard in typed_shards[1:]:
        if shard.parameters != parameters:
            raise ValueError("All model-input shards must use identical parameters.")

    return ModelInputSet(
        parameters=parameters,
        records=tuple(record for shard in typed_shards for record in shard.records),
    )


def _iter_feature_records(
    ir_features: IrFeatureSet | Mapping[str, Any],
) -> tuple[_FeatureRecordInput, ...]:
    if isinstance(ir_features, IrFeatureSet):
        source_records = ir_features.records
    else:
        source_records = ir_features.get("records", ())

    normalized_records: list[_FeatureRecordInput] = []
    for record in source_records:
        if isinstance(record, IrFeatureRecord):
            normalized_records.append(
                _FeatureRecordInput(
                    relative_path=record.relative_path,
                    projection_type=record.projection_type,
                    projection=record.projection,
                )
            )
            continue

        normalized_records.append(
            _FeatureRecordInput(
                relative_path=str(record["relative_path"]),
                projection_type=ProjectionType(record["projection_type"]),
                projection=record.get("projection", {}),
            )
        )

    return tuple(normalized_records)


def _tokenize_record(
    record: _FeatureRecordInput,
    parameters: TokenizationParameters,
) -> ModelInputRecord:
    base_tokens = (
        f"projection:{record.projection_type.value}",
        f"vocabulary:{parameters.vocabulary_strategy}",
        f"time_resolution:{parameters.time_resolution}",
        *_projection_summary_tokens(record),
    )
    tokens, attention_mask = _apply_sequence_constraints(base_tokens, parameters)
    return ModelInputRecord(
        relative_path=record.relative_path,
        projection_type=record.projection_type,
        vocabulary_strategy=parameters.vocabulary_strategy,
        time_resolution=parameters.time_resolution,
        original_token_count=len(base_tokens),
        tokens=tokens,
        attention_mask=attention_mask,
    )


def _coerce_model_input_set(
    value: ModelInputSet | Mapping[str, Any],
) -> ModelInputSet:
    if isinstance(value, ModelInputSet):
        return value

    return ModelInputSet(
        parameters=coerce_tokenization_parameters(value.get("parameters", {})),
        records=tuple(
            ModelInputRecord(
                relative_path=str(record["relative_path"]),
                projection_type=ProjectionType(record["projection_type"]),
                vocabulary_strategy=str(record["vocabulary_strategy"]),
                time_resolution=int(record["time_resolution"]),
                original_token_count=int(record["original_token_count"]),
                tokens=tuple(str(token) for token in record.get("tokens", ())),
                attention_mask=tuple(
                    int(mask_value) for mask_value in record.get("attention_mask", ())
                ),
            )
            for record in value.get("records", ())
        ),
    )


def _projection_summary_tokens(record: _FeatureRecordInput) -> tuple[str, ...]:
    if record.projection_type is ProjectionType.SEQUENCE:
        return (f"events:{_projection_item_count(record.projection, 'events')}",)

    if record.projection_type is ProjectionType.GRAPH:
        return (
            f"nodes:{_projection_item_count(record.projection, 'nodes')}",
            f"edges:{_projection_item_count(record.projection, 'edges')}",
        )

    return (
        f"parts:{_projection_item_count(record.projection, 'parts')}",
        f"bars:{_projection_item_count(record.projection, 'bars')}",
    )


def _projection_item_count(projection: object, field_name: str) -> int:
    if isinstance(projection, Mapping):
        items = projection.get(field_name, ())
    else:
        items = getattr(projection, field_name)

    return len(items)


def _apply_sequence_constraints(
    tokens: tuple[str, ...],
    parameters: TokenizationParameters,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    truncated_tokens = tuple(tokens[: parameters.max_sequence_length])
    attention_mask = tuple(1 for _ in truncated_tokens)
    padding_needed = parameters.max_sequence_length - len(truncated_tokens)
    if padding_needed <= 0 or parameters.padding_strategy is PaddingStrategy.NONE:
        return truncated_tokens, attention_mask

    pad_tokens = tuple(PAD_TOKEN for _ in range(padding_needed))
    pad_mask = tuple(0 for _ in range(padding_needed))
    if parameters.padding_strategy is PaddingStrategy.LEFT:
        return pad_tokens + truncated_tokens, pad_mask + attention_mask

    return truncated_tokens + pad_tokens, attention_mask + pad_mask
