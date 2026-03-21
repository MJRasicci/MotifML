"""Nodes for the tokenization pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from motifml.ir.projections.sequence import SequenceProjection
from motifml.pipelines.feature_extraction.models import (
    IrFeatureRecord,
    IrFeatureSet,
    ProjectionType,
    coerce_feature_extraction_parameters,
)
from motifml.pipelines.tokenization.models import (
    ModelInputRecord,
    ModelInputSet,
    PaddingStrategy,
    ShardTokenCounts,
    TokenCountEntry,
    TokenFamilyCoverageEntry,
    TokenizationParameters,
    VocabularyArtifact,
    VocabularyParameters,
    VocabularyStatsReport,
    coerce_shard_token_counts,
    coerce_tokenization_parameters,
    coerce_vocabulary_parameters,
)
from motifml.training.contracts import (
    DatasetSplit,
    SplitManifestEntry,
    VocabularyMetadata,
    coerce_split_manifest_entries,
)
from motifml.training.sequence_schema import (
    SequenceSchemaContract,
    coerce_sequence_schema_contract,
)
from motifml.training.special_token_policy import coerce_special_token_policy
from motifml.training.token_codec import encode_projected_events_to_tokens
from motifml.training.token_families import PAD_TOKEN, SPECIAL_TOKENS
from motifml.training.versioning import build_vocabulary_version


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


def count_training_split_tokens(
    ir_features: IrFeatureSet | Mapping[str, Any],
    split_manifest: tuple[SplitManifestEntry, ...] | list[Mapping[str, Any]],
    sequence_schema: SequenceSchemaContract | Mapping[str, Any],
    parameters: VocabularyParameters | Mapping[str, Any],
    special_token_policy: Mapping[str, Any],
) -> ShardTokenCounts:
    """Count training-split token frequencies in a stable reducer-friendly shape."""
    typed_parameters = coerce_vocabulary_parameters(parameters)
    typed_sequence_schema = coerce_sequence_schema_contract(sequence_schema)
    typed_special_token_policy = coerce_special_token_policy(special_token_policy)
    manifest_entries = coerce_split_manifest_entries(split_manifest)
    split_version = _split_version_for_manifest(manifest_entries)
    split_by_path = {entry.relative_path: entry for entry in manifest_entries}
    feature_set = _coerce_feature_set(ir_features)
    if feature_set.parameters.feature_version is None:
        raise ValueError("ir_features.parameters.feature_version must be populated.")

    counted_relative_paths: list[str] = []
    token_counter: Counter[str] = Counter()
    for record in feature_set.records:
        split_entry = split_by_path.get(record.relative_path)
        if split_entry is None or split_entry.split is not DatasetSplit.TRAIN:
            continue
        if record.projection_type is not ProjectionType.SEQUENCE:
            raise ValueError(
                "count_training_split_tokens only supports sequence projections."
            )
        if not isinstance(record.projection, SequenceProjection):
            raise ValueError(
                "Sequence feature records must contain typed SequenceProjection data "
                "for token counting."
            )
        counted_relative_paths.append(record.relative_path)
        token_counter.update(
            encode_projected_events_to_tokens(
                record.projection.events,
                time_resolution=typed_parameters.time_resolution,
                note_payload_fields=typed_sequence_schema.note_payload_fields,
                special_token_policy=typed_special_token_policy,
            )
        )

    return ShardTokenCounts(
        feature_version=feature_set.parameters.feature_version,
        split_version=split_version,
        time_resolution=typed_parameters.time_resolution,
        special_token_policy=typed_special_token_policy.to_version_payload(),
        counted_document_count=len(counted_relative_paths),
        total_token_count=sum(token_counter.values()),
        counted_relative_paths=tuple(counted_relative_paths),
        token_counts=tuple(
            TokenCountEntry(token=token, count=count)
            for token, count in sorted(token_counter.items())
        ),
    )


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


def merge_token_count_shards(
    token_count_shards: list[ShardTokenCounts] | list[Mapping[str, Any]],
) -> ShardTokenCounts:
    """Merge shard-local token-count artifacts while preserving deterministic order."""
    typed_shards = [coerce_shard_token_counts(shard) for shard in token_count_shards]
    if not typed_shards:
        raise ValueError("token_count_shards must contain at least one shard artifact.")

    baseline = typed_shards[0]
    token_counter: Counter[str] = Counter()
    counted_relative_paths: list[str] = []
    for shard in typed_shards:
        _validate_shard_count_contract(shard, baseline)
        token_counter.update({entry.token: entry.count for entry in shard.token_counts})
        counted_relative_paths.extend(shard.counted_relative_paths)

    return ShardTokenCounts(
        feature_version=baseline.feature_version,
        split_version=baseline.split_version,
        time_resolution=baseline.time_resolution,
        special_token_policy=dict(baseline.special_token_policy),
        counted_document_count=len(counted_relative_paths),
        total_token_count=sum(token_counter.values()),
        counted_relative_paths=tuple(counted_relative_paths),
        token_counts=tuple(
            TokenCountEntry(token=token, count=count)
            for token, count in sorted(token_counter.items())
        ),
    )


def reduce_vocabulary(
    token_count_shards: list[ShardTokenCounts] | list[Mapping[str, Any]],
    parameters: VocabularyParameters | Mapping[str, Any],
    split_seed: int | str,
) -> tuple[VocabularyArtifact, VocabularyStatsReport, VocabularyMetadata]:
    """Reduce shard-local counts into one deterministic frozen vocabulary surface."""
    merged_counts = merge_token_count_shards(token_count_shards)
    typed_parameters = coerce_vocabulary_parameters(parameters)
    retained_counts = _retained_token_counts(
        merged_counts,
        typed_parameters=typed_parameters,
    )
    vocabulary_version_payload = _vocabulary_version_payload(typed_parameters)
    vocabulary_version = build_vocabulary_version(
        feature_version=merged_counts.feature_version,
        tokenization_config=vocabulary_version_payload,
        split_version=merged_counts.split_version,
        split_seed=split_seed,
        special_token_policy=merged_counts.special_token_policy,
    )
    vocabulary = VocabularyArtifact(
        vocabulary_version=vocabulary_version,
        feature_version=merged_counts.feature_version,
        split_version=merged_counts.split_version,
        token_count=sum(entry.count for entry in retained_counts),
        vocabulary_size=len(retained_counts),
        token_to_id=_build_token_to_id(
            retained_counts,
            special_tokens=typed_parameters.special_tokens or SPECIAL_TOKENS,
        ),
        token_counts=retained_counts,
        construction_parameters=vocabulary_version_payload,
        special_token_policy=dict(merged_counts.special_token_policy),
    )
    stats = VocabularyStatsReport(
        vocabulary_version=vocabulary_version,
        feature_version=merged_counts.feature_version,
        split_version=merged_counts.split_version,
        token_count=vocabulary.token_count,
        vocabulary_size=vocabulary.vocabulary_size,
        token_family_coverage=_build_token_family_coverage(retained_counts),
        top_tokens=tuple(
            sorted(retained_counts, key=lambda item: (-item.count, item.token))[:10]
        ),
        construction_parameters=vocabulary_version_payload,
    )
    metadata = VocabularyMetadata(
        vocabulary_version=vocabulary_version,
        feature_version=merged_counts.feature_version,
        split_version=merged_counts.split_version,
        token_count=vocabulary.token_count,
        vocabulary_size=vocabulary.vocabulary_size,
        construction_parameters=vocabulary_version_payload,
        special_token_policy=dict(merged_counts.special_token_policy),
    )
    return vocabulary, stats, metadata


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


def _coerce_feature_set(value: IrFeatureSet | Mapping[str, Any]) -> IrFeatureSet:
    if isinstance(value, IrFeatureSet):
        return value

    return IrFeatureSet(
        parameters=coerce_feature_extraction_parameters(value.get("parameters", {})),
        records=tuple(
            IrFeatureRecord(
                relative_path=str(record["relative_path"]),
                projection_type=ProjectionType(record["projection_type"]),
                projection=record.get("projection", {}),
            )
            for record in value.get("records", ())
        ),
    )


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


def _split_version_for_manifest(entries: tuple[SplitManifestEntry, ...]) -> str:
    if not entries:
        raise ValueError("split_manifest must contain at least one entry.")
    split_version = entries[0].split_version
    for entry in entries[1:]:
        if entry.split_version != split_version:
            raise ValueError("split_manifest entries must share one split_version.")
    return split_version


def _validate_shard_count_contract(
    shard: ShardTokenCounts,
    baseline: ShardTokenCounts,
) -> None:
    if shard.feature_version != baseline.feature_version:
        raise ValueError("All token-count shards must share one feature_version.")
    if shard.split_version != baseline.split_version:
        raise ValueError("All token-count shards must share one split_version.")
    if shard.time_resolution != baseline.time_resolution:
        raise ValueError("All token-count shards must share one time_resolution.")
    if shard.special_token_policy != baseline.special_token_policy:
        raise ValueError("All token-count shards must share one special_token_policy.")


def _retained_token_counts(
    merged_counts: ShardTokenCounts,
    *,
    typed_parameters: VocabularyParameters,
) -> tuple[TokenCountEntry, ...]:
    special_tokens = typed_parameters.special_tokens or SPECIAL_TOKENS
    special_token_values = tuple(special_tokens.values())
    counted_by_token = {
        entry.token: entry.count for entry in merged_counts.token_counts
    }
    retained: list[TokenCountEntry] = [
        TokenCountEntry(
            token=token,
            count=counted_by_token.get(token, 0),
        )
        for token in special_token_values
    ]
    ordinary_tokens = [
        entry
        for entry in merged_counts.token_counts
        if entry.token not in set(special_token_values)
        and entry.count >= typed_parameters.minimum_frequency
    ]
    ordinary_tokens.sort(key=lambda item: (-item.count, item.token))
    remaining_capacity = max(typed_parameters.maximum_size - len(retained), 0)
    retained.extend(ordinary_tokens[:remaining_capacity])
    return tuple(sorted(retained, key=lambda item: item.token))


def _build_token_to_id(
    retained_counts: tuple[TokenCountEntry, ...],
    *,
    special_tokens: dict[str, str],
) -> dict[str, int]:
    ordered_tokens = [
        special_tokens[key]
        for key in ("pad", "bos", "eos", "unk")
        if key in special_tokens
    ]
    ordered_tokens.extend(
        entry.token
        for entry in sorted(retained_counts, key=lambda item: (-item.count, item.token))
        if entry.token not in set(ordered_tokens)
    )
    return {token: index for index, token in enumerate(ordered_tokens)}


def _build_token_family_coverage(
    retained_counts: tuple[TokenCountEntry, ...],
) -> tuple[TokenFamilyCoverageEntry, ...]:
    coverage: dict[str, tuple[int, int]] = {}
    for entry in retained_counts:
        family = _token_family(entry.token)
        vocabulary_size, token_count = coverage.get(family, (0, 0))
        coverage[family] = (vocabulary_size + 1, token_count + entry.count)
    return tuple(
        TokenFamilyCoverageEntry(
            family=family,
            vocabulary_size=vocabulary_size,
            token_count=token_count,
        )
        for family, (vocabulary_size, token_count) in sorted(coverage.items())
    )


def _token_family(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        return "SPECIAL"
    return token.split(":", maxsplit=1)[0]


def _vocabulary_version_payload(parameters: VocabularyParameters) -> dict[str, Any]:
    return {
        "time_resolution": parameters.time_resolution,
        "minimum_frequency": parameters.minimum_frequency,
        "maximum_size": parameters.maximum_size,
        "special_tokens": dict(parameters.special_tokens or SPECIAL_TOKENS),
    }


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
