"""Nodes for the tokenization pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from motifml.datasets.model_input_storage import coerce_model_input_storage_schema
from motifml.ir.models import (
    ControlScope,
    DynamicChangeValue,
    FermataValue,
    HairpinDirection,
    HairpinValue,
    NoteEvent,
    OttavaValue,
    Pitch,
    PitchStep,
    PointControlEvent,
    PointControlKind,
    SpanControlEvent,
    SpanControlKind,
    TempoChangeValue,
)
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    SequenceProjection,
    SequenceProjectionMode,
    SpanControlSequenceEvent,
    StructureMarkerKind,
    StructureMarkerSequenceEvent,
)
from motifml.ir.time import ScoreTime
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
    VocabularyGuardrailParameters,
    VocabularyGuardrailReport,
    VocabularyParameters,
    VocabularyStatsReport,
    coerce_shard_token_counts,
    coerce_tokenization_parameters,
    coerce_vocabulary_artifact,
    coerce_vocabulary_parameters,
)
from motifml.training.contracts import (
    DatasetSplit,
    ModelInputMetadata,
    SplitManifestEntry,
    VocabularyMetadata,
    coerce_split_manifest_entries,
)
from motifml.training.model_input import TokenizedDocumentRow
from motifml.training.sequence_schema import (
    SequenceSchemaContract,
    coerce_sequence_schema_contract,
)
from motifml.training.special_token_policy import coerce_special_token_policy
from motifml.training.token_codec import (
    encode_projected_events_to_tokens,
    encode_token_strings_to_ids,
)
from motifml.training.token_families import PAD_TOKEN, SPECIAL_TOKENS
from motifml.training.versioning import (
    build_model_input_version,
    build_vocabulary_version,
)


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


def count_training_split_tokens_from_model_input_parameters(
    ir_features: IrFeatureSet | Mapping[str, Any],
    split_manifest: tuple[SplitManifestEntry, ...] | list[Mapping[str, Any]],
    sequence_schema: SequenceSchemaContract | Mapping[str, Any],
    vocabulary_parameters: VocabularyParameters | Mapping[str, Any],
    model_input_parameters: Mapping[str, Any],
) -> ShardTokenCounts:
    """Pipeline-friendly wrapper that reads special-token policy from model input."""
    return count_training_split_tokens(
        ir_features,
        split_manifest,
        sequence_schema,
        vocabulary_parameters,
        special_token_policy=_special_token_policy_from_model_input(
            model_input_parameters
        ),
    )


def tokenize_features_with_vocabulary(
    ir_features: IrFeatureSet | Mapping[str, Any],
    split_manifest: tuple[SplitManifestEntry, ...] | list[Mapping[str, Any]],
    sequence_schema: SequenceSchemaContract | Mapping[str, Any],
    vocabulary: VocabularyArtifact | Mapping[str, Any],
    model_input_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Tokenize sequence features into typed per-document model-input rows."""
    feature_set = _coerce_feature_set(ir_features)
    typed_sequence_schema = coerce_sequence_schema_contract(sequence_schema)
    manifest_entries = coerce_split_manifest_entries(split_manifest)
    typed_vocabulary = coerce_vocabulary_artifact(vocabulary)
    typed_special_token_policy = coerce_special_token_policy(
        _special_token_policy_from_model_input(model_input_parameters)
    )
    split_version = _split_version_for_manifest(manifest_entries)
    split_by_path = {entry.relative_path: entry for entry in manifest_entries}
    if feature_set.parameters.feature_version is None:
        raise ValueError("ir_features.parameters.feature_version must be populated.")
    if feature_set.parameters.normalized_ir_version is None:
        raise ValueError(
            "ir_features.parameters.normalized_ir_version must be populated."
        )
    if typed_vocabulary.feature_version != feature_set.parameters.feature_version:
        raise ValueError(
            "vocabulary.feature_version must match ir_features.parameters.feature_version."
        )
    if typed_vocabulary.split_version != split_version:
        raise ValueError("vocabulary.split_version must match split_manifest.")

    storage_parameters = _storage_parameters_from_model_input(model_input_parameters)
    storage_schema = coerce_model_input_storage_schema(storage_parameters)
    model_input_version = build_model_input_version(
        feature_version=feature_set.parameters.feature_version,
        vocabulary_version=typed_vocabulary.vocabulary_version,
        model_input_config=_model_input_version_payload(model_input_parameters),
        special_token_policy=typed_special_token_policy.to_version_payload(),
        storage_schema_version=storage_schema.storage_schema_version,
    )
    metadata = ModelInputMetadata(
        model_input_version=model_input_version,
        normalized_ir_version=feature_set.parameters.normalized_ir_version,
        feature_version=feature_set.parameters.feature_version,
        vocabulary_version=typed_vocabulary.vocabulary_version,
        projection_type=str(model_input_parameters["projection_type"]),
        sequence_mode=str(model_input_parameters["sequence_mode"]),
        context_length=int(model_input_parameters["context_length"]),
        stride=int(model_input_parameters["stride"]),
        padding_strategy=str(model_input_parameters["padding_strategy"]),
        special_token_policy=typed_special_token_policy.to_version_payload(),
        storage_backend=storage_schema.backend,
        storage_schema_version=storage_schema.storage_schema_version,
    )

    records: list[TokenizedDocumentRow] = []
    time_resolution = int(typed_vocabulary.construction_parameters["time_resolution"])
    for record in feature_set.records:
        split_entry = split_by_path.get(record.relative_path)
        if split_entry is None:
            raise ValueError(
                "split_manifest is missing an entry for feature record "
                f"{record.relative_path!r}."
            )
        if record.projection_type is not ProjectionType.SEQUENCE:
            raise ValueError(
                "tokenize_features_with_vocabulary only supports sequence projections."
            )
        if not isinstance(record.projection, SequenceProjection):
            raise ValueError(
                "Sequence feature records must contain typed SequenceProjection data "
                "for vocabulary-driven tokenization."
            )
        token_strings = encode_projected_events_to_tokens(
            record.projection.events,
            time_resolution=time_resolution,
            note_payload_fields=typed_sequence_schema.note_payload_fields,
            special_token_policy=typed_special_token_policy,
        )
        token_ids = encode_token_strings_to_ids(
            token_strings,
            vocabulary=typed_vocabulary.token_to_id,
            special_token_policy=typed_special_token_policy,
        )
        records.append(
            TokenizedDocumentRow(
                relative_path=record.relative_path,
                document_id=split_entry.document_id,
                split=split_entry.split,
                split_version=split_entry.split_version,
                projection_type=str(model_input_parameters["projection_type"]),
                sequence_mode=str(model_input_parameters["sequence_mode"]),
                normalized_ir_version=feature_set.parameters.normalized_ir_version,
                feature_version=feature_set.parameters.feature_version,
                vocabulary_version=typed_vocabulary.vocabulary_version,
                model_input_version=model_input_version,
                storage_schema_version=storage_schema.storage_schema_version,
                token_count=len(token_ids),
                token_ids=token_ids,
                window_start_offsets=(),
                context_length=int(model_input_parameters["context_length"]),
                stride=int(model_input_parameters["stride"]),
                padding_strategy=str(model_input_parameters["padding_strategy"]),
                special_token_policy=typed_special_token_policy.to_version_payload(),
            )
        )

    return {
        "parameters": metadata,
        "storage_schema": storage_schema,
        "records": tuple(records),
    }


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
    guardrail_report = _build_vocabulary_guardrail_report(
        merged_counts,
        retained_counts,
        typed_parameters.guardrails,
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
        guardrails=guardrail_report,
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
    _raise_if_vocabulary_degenerate(guardrail_report)
    return vocabulary, stats, metadata


def reduce_vocabulary_from_data_split_parameters(
    token_count_shards: list[ShardTokenCounts] | list[Mapping[str, Any]],
    parameters: VocabularyParameters | Mapping[str, Any],
    data_split_parameters: Mapping[str, Any],
) -> tuple[VocabularyArtifact, VocabularyStatsReport, VocabularyMetadata]:
    """Pipeline-friendly wrapper that derives the split seed from data-split params."""
    return reduce_vocabulary(
        token_count_shards,
        parameters,
        split_seed=data_split_parameters["hash_seed"],
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


def _coerce_feature_set(value: IrFeatureSet | Mapping[str, Any]) -> IrFeatureSet:
    if isinstance(value, IrFeatureSet):
        return value

    return IrFeatureSet(
        parameters=coerce_feature_extraction_parameters(value.get("parameters", {})),
        records=tuple(
            _coerce_feature_record(record) for record in value.get("records", ())
        ),
    )


def _coerce_feature_record(
    record: IrFeatureRecord | Mapping[str, Any],
) -> IrFeatureRecord:
    if isinstance(record, IrFeatureRecord):
        return record

    projection_type = ProjectionType(record["projection_type"])
    projection = record.get("projection", {})
    if projection_type is ProjectionType.SEQUENCE:
        projection = _coerce_sequence_projection(projection)

    return IrFeatureRecord(
        relative_path=str(record["relative_path"]),
        projection_type=projection_type,
        projection=projection,
    )


def _coerce_sequence_projection(
    value: SequenceProjection | Mapping[str, Any],
) -> SequenceProjection:
    if isinstance(value, SequenceProjection):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Sequence projections must be mappings or SequenceProjection.")

    return SequenceProjection(
        mode=SequenceProjectionMode(str(value["mode"])),
        events=tuple(
            _coerce_sequence_event(event) for event in value.get("events", ())
        ),
    )


def _coerce_sequence_event(
    value: object,
) -> (
    StructureMarkerSequenceEvent
    | PointControlSequenceEvent
    | SpanControlSequenceEvent
    | NoteSequenceEvent
):
    if isinstance(
        value,
        StructureMarkerSequenceEvent
        | PointControlSequenceEvent
        | SpanControlSequenceEvent
        | NoteSequenceEvent,
    ):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Sequence events must be mappings or typed event objects.")

    if "note" in value:
        return NoteSequenceEvent(
            time=_coerce_score_time(value["time"]),
            note=_coerce_note_event(value["note"]),
            part_id=str(value["part_id"]),
            staff_id=str(value["staff_id"]),
            bar_id=str(value["bar_id"]),
            voice_lane_id=str(value["voice_lane_id"]),
            onset_id=str(value["onset_id"]),
        )

    if "marker_kind" in value:
        return StructureMarkerSequenceEvent(
            time=_coerce_score_time(value["time"]),
            marker_kind=StructureMarkerKind(str(value["marker_kind"])),
            entity_id=str(value["entity_id"]),
            part_id=_optional_text(value.get("part_id")),
            staff_id=_optional_text(value.get("staff_id")),
            bar_id=_optional_text(value.get("bar_id")),
            voice_lane_id=_optional_text(value.get("voice_lane_id")),
        )

    control = value.get("control")
    if isinstance(control, Mapping) and "start_time" in control:
        return SpanControlSequenceEvent(
            time=_coerce_score_time(value["time"]),
            control=_coerce_span_control_event(control),
            part_id=_optional_text(value.get("part_id")),
            staff_id=_optional_text(value.get("staff_id")),
            bar_id=_optional_text(value.get("bar_id")),
            voice_lane_id=_optional_text(value.get("voice_lane_id")),
        )

    if isinstance(control, Mapping):
        return PointControlSequenceEvent(
            time=_coerce_score_time(value["time"]),
            control=_coerce_point_control_event(control),
            part_id=_optional_text(value.get("part_id")),
            staff_id=_optional_text(value.get("staff_id")),
            bar_id=_optional_text(value.get("bar_id")),
            voice_lane_id=_optional_text(value.get("voice_lane_id")),
        )

    raise ValueError("Unable to determine the sequence event type from persisted data.")


def _coerce_note_event(value: NoteEvent | Mapping[str, Any]) -> NoteEvent:
    if isinstance(value, NoteEvent):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Note events must be mappings or NoteEvent instances.")

    pitch = value.get("pitch")
    return NoteEvent(
        note_id=str(value["note_id"]),
        onset_id=str(value["onset_id"]),
        part_id=str(value["part_id"]),
        staff_id=str(value["staff_id"]),
        time=_coerce_score_time(value["time"]),
        attack_duration=_coerce_score_time(value["attack_duration"]),
        sounding_duration=_coerce_score_time(value["sounding_duration"]),
        pitch=None if pitch is None else _coerce_pitch(pitch),
        velocity=None if value.get("velocity") is None else int(value["velocity"]),
        string_number=(
            None if value.get("string_number") is None else int(value["string_number"])
        ),
        show_string_number=(
            None
            if value.get("show_string_number") is None
            else bool(value["show_string_number"])
        ),
        techniques=value.get("techniques"),
    )


def _coerce_pitch(value: Pitch | Mapping[str, Any]) -> Pitch:
    if isinstance(value, Pitch):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Pitch values must be mappings or Pitch instances.")

    return Pitch(
        step=PitchStep(str(value["step"])),
        octave=int(value["octave"]),
        accidental=_optional_text(value.get("accidental")),
    )


def _coerce_point_control_event(
    value: PointControlEvent | Mapping[str, Any],
) -> PointControlEvent:
    if isinstance(value, PointControlEvent):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "Point control events must be mappings or PointControlEvent instances."
        )

    kind = PointControlKind(str(value["kind"]))
    return PointControlEvent(
        control_id=str(value["control_id"]),
        kind=kind,
        scope=ControlScope(str(value["scope"])),
        target_ref=str(value["target_ref"]),
        time=_coerce_score_time(value["time"]),
        value=_coerce_point_control_value(kind, value["value"]),
    )


def _coerce_span_control_event(
    value: SpanControlEvent | Mapping[str, Any],
) -> SpanControlEvent:
    if isinstance(value, SpanControlEvent):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "Span control events must be mappings or SpanControlEvent instances."
        )

    kind = SpanControlKind(str(value["kind"]))
    return SpanControlEvent(
        control_id=str(value["control_id"]),
        kind=kind,
        scope=ControlScope(str(value["scope"])),
        target_ref=str(value["target_ref"]),
        start_time=_coerce_score_time(value["start_time"]),
        end_time=_coerce_score_time(value["end_time"]),
        value=_coerce_span_control_value(kind, value["value"]),
        start_anchor_ref=_optional_text(value.get("start_anchor_ref")),
        end_anchor_ref=_optional_text(value.get("end_anchor_ref")),
    )


def _coerce_point_control_value(
    kind: PointControlKind,
    value: object,
) -> TempoChangeValue | DynamicChangeValue | FermataValue:
    if kind is PointControlKind.TEMPO_CHANGE:
        payload = _mapping(value, "tempo change value")
        return TempoChangeValue(beats_per_minute=float(payload["beats_per_minute"]))
    if kind is PointControlKind.DYNAMIC_CHANGE:
        payload = _mapping(value, "dynamic change value")
        return DynamicChangeValue(marking=str(payload["marking"]))
    payload = _mapping(value, "fermata value")
    return FermataValue(
        fermata_type=_optional_text(payload.get("fermata_type")),
        length_scale=(
            None
            if payload.get("length_scale") is None
            else float(payload["length_scale"])
        ),
    )


def _coerce_span_control_value(
    kind: SpanControlKind,
    value: object,
) -> HairpinValue | OttavaValue:
    payload = _mapping(value, "span control value")
    if kind is SpanControlKind.HAIRPIN:
        return HairpinValue(
            direction=HairpinDirection(str(payload["direction"])),
            niente=bool(payload.get("niente", False)),
        )
    return OttavaValue(octave_shift=int(payload["octave_shift"]))


def _coerce_score_time(value: ScoreTime | Mapping[str, Any]) -> ScoreTime:
    if isinstance(value, ScoreTime):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("ScoreTime values must be mappings or ScoreTime instances.")
    return ScoreTime(int(value["numerator"]), int(value["denominator"]))


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


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


def _build_vocabulary_guardrail_report(
    merged_counts: ShardTokenCounts,
    retained_counts: tuple[TokenCountEntry, ...],
    guardrails: VocabularyGuardrailParameters,
) -> VocabularyGuardrailReport:
    token_family_coverage = _build_token_family_coverage(retained_counts)
    observed_families = {entry.family for entry in token_family_coverage}
    missing_required_families = tuple(
        family
        for family in guardrails.required_token_families
        if family not in observed_families
    )

    retained_token_count = sum(entry.count for entry in retained_counts)
    dropped_token_count = max(merged_counts.total_token_count - retained_token_count, 0)
    estimated_unk_fraction = (
        dropped_token_count / merged_counts.total_token_count
        if merged_counts.total_token_count > 0
        else 0.0
    )

    counted_tokens = tuple(
        entry
        for entry in sorted(retained_counts, key=lambda item: (-item.count, item.token))
        if entry.count > 0
    )
    top_token = counted_tokens[0] if counted_tokens else None
    top_token_fraction = (
        top_token.count / retained_token_count
        if top_token is not None and retained_token_count > 0
        else 0.0
    )
    passed = (
        len(retained_counts) >= guardrails.minimum_vocabulary_size
        and not missing_required_families
        and top_token_fraction <= guardrails.maximum_top_token_fraction
        and estimated_unk_fraction <= guardrails.maximum_unk_fraction
    )
    return VocabularyGuardrailReport(
        observed_vocabulary_size=len(retained_counts),
        minimum_vocabulary_size=guardrails.minimum_vocabulary_size,
        required_token_families=guardrails.required_token_families,
        missing_required_token_families=missing_required_families,
        top_token=None if top_token is None else top_token.token,
        top_token_count=0 if top_token is None else top_token.count,
        top_token_fraction=top_token_fraction,
        maximum_top_token_fraction=guardrails.maximum_top_token_fraction,
        estimated_unk_token_count=dropped_token_count,
        estimated_unk_fraction=estimated_unk_fraction,
        maximum_unk_fraction=guardrails.maximum_unk_fraction,
        passed=passed,
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


def _raise_if_vocabulary_degenerate(report: VocabularyGuardrailReport) -> None:
    if report.passed:
        return

    failures: list[str] = []
    if report.observed_vocabulary_size < report.minimum_vocabulary_size:
        failures.append(
            "vocabulary_size="
            f"{report.observed_vocabulary_size} is below minimum_vocabulary_size="
            f"{report.minimum_vocabulary_size}"
        )
    if report.missing_required_token_families:
        failures.append(
            "missing required token families: "
            + ", ".join(report.missing_required_token_families)
        )
    if report.top_token_fraction > report.maximum_top_token_fraction:
        failures.append(
            "top token concentration exceeds threshold: "
            f"{report.top_token!r} accounts for "
            f"{report.top_token_fraction:.4f} of retained tokens "
            f"(max {report.maximum_top_token_fraction:.4f})"
        )
    if report.estimated_unk_fraction > report.maximum_unk_fraction:
        failures.append(
            "estimated <unk> rate exceeds threshold: "
            f"{report.estimated_unk_fraction:.4f} "
            f"({report.estimated_unk_token_count} dropped tokens, "
            f"max {report.maximum_unk_fraction:.4f})"
        )
    raise ValueError("Vocabulary guardrails failed: " + "; ".join(failures) + ".")


def _special_token_policy_from_model_input(
    model_input_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "special_token_policy" not in model_input_parameters:
        raise ValueError("model_input parameters must include special_token_policy.")
    special_token_policy = model_input_parameters["special_token_policy"]
    if not isinstance(special_token_policy, Mapping):
        raise ValueError("special_token_policy must be a mapping.")
    return special_token_policy


def _storage_parameters_from_model_input(
    model_input_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    storage_parameters = model_input_parameters.get("storage")
    if not isinstance(storage_parameters, Mapping):
        raise ValueError("model_input parameters must include a storage mapping.")
    return storage_parameters


def _model_input_version_payload(
    model_input_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "projection_type": str(model_input_parameters["projection_type"]),
        "sequence_mode": str(model_input_parameters["sequence_mode"]),
        "context_length": int(model_input_parameters["context_length"]),
        "stride": int(model_input_parameters["stride"]),
        "padding_strategy": str(model_input_parameters["padding_strategy"]),
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
