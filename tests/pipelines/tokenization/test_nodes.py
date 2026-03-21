"""Unit tests for tokenization nodes and reducer helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.models import NoteEvent, Pitch
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    SequenceProjection,
    SequenceProjectionMode,
    StructureMarkerKind,
    StructureMarkerSequenceEvent,
)
from motifml.ir.serialization import deserialize_document
from motifml.ir.time import ScoreTime
from motifml.pipelines.feature_extraction.models import (
    FeatureExtractionParameters,
    IrFeatureRecord,
    IrFeatureSet,
)
from motifml.pipelines.feature_extraction.nodes import extract_features
from motifml.pipelines.tokenization.models import (
    PaddingStrategy,
    TokenizationParameters,
    VocabularyParameters,
)
from motifml.pipelines.tokenization.nodes import (
    count_training_split_tokens,
    merge_model_input_shards,
    reduce_vocabulary,
    tokenize_features,
    tokenize_features_with_vocabulary,
)
from motifml.training.contracts import (
    DatasetSplit,
    ModelInputMetadata,
    SplitManifestEntry,
    VocabularyMetadata,
)
from motifml.training.model_input import (
    TokenizedDocumentRow,
    build_window_start_offsets,
)
from motifml.training.sequence_schema import SequenceSchemaContract
from motifml.training.token_codec import (
    decode_token_ids_to_strings,
    encode_projected_events_to_tokens,
)
from motifml.training.token_families import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures"
REPRESENTATIVE_DOCUMENT_FIXTURE = (
    FIXTURE_ROOT / "ir" / "representative_document.ir.json"
)

EXPECTED_TIME_RESOLUTION = 48
EXPECTED_TRAIN_TOKEN_COUNT = 6
EXPECTED_REDUCED_VOCABULARY_SIZE = 7
EXPECTED_TOP_TOKEN_COUNT = 4
EXPECTED_DROPPED_TOKEN_COUNT = 3
EXPECTED_EOS_TOKEN_ID = 2


def test_tokenize_features_consumes_typed_parameters() -> None:
    model_input = tokenize_features(
        IrFeatureSet(
            parameters=FeatureExtractionParameters(),
            records=(
                IrFeatureRecord(
                    relative_path="fixtures/example.json",
                    projection_type="sequence",
                    projection=SequenceProjection(
                        mode=SequenceProjectionMode.NOTES_ONLY,
                        events=(),
                    ),
                ),
            ),
        ),
        TokenizationParameters(
            vocabulary_strategy="shared_placeholder",
            max_sequence_length=6,
            padding_strategy=PaddingStrategy.RIGHT,
            time_resolution=24,
        ),
    )

    assert model_input.parameters.vocabulary_strategy == "shared_placeholder"
    assert model_input.records[0].tokens == (
        "projection:sequence",
        "vocabulary:shared_placeholder",
        "time_resolution:24",
        "events:0",
        PAD_TOKEN,
        PAD_TOKEN,
    )
    assert model_input.records[0].attention_mask == (1, 1, 1, 1, 0, 0)


def test_tokenize_features_accepts_json_loaded_feature_sets() -> None:
    model_input = tokenize_features(
        {
            "parameters": {"projection_type": "graph"},
            "records": [
                {
                    "relative_path": "fixtures/b.json",
                    "projection_type": "graph",
                    "projection": {"nodes": [{}, {}], "edges": [{}]},
                },
                {
                    "relative_path": "fixtures/a.json",
                    "projection_type": "hierarchical",
                    "projection": {"parts": [{}], "bars": [{}, {}]},
                },
            ],
        },
        {
            "vocabulary_strategy": "projection_native",
            "max_sequence_length": 5,
            "padding_strategy": "left",
            "time_resolution": EXPECTED_TIME_RESOLUTION,
        },
    )

    assert [record.relative_path for record in model_input.records] == [
        "fixtures/a.json",
        "fixtures/b.json",
    ]
    assert model_input.records[0].tokens == (
        "projection:hierarchical",
        "vocabulary:projection_native",
        "time_resolution:48",
        "parts:1",
        "bars:2",
    )
    assert model_input.records[1].tokens == (
        "projection:graph",
        "vocabulary:projection_native",
        "time_resolution:48",
        "nodes:2",
        "edges:1",
    )
    assert all(
        record.attention_mask == (1, 1, 1, 1, 1) for record in model_input.records
    )


def test_merge_model_input_shards_preserves_parameter_contract_and_order() -> None:
    merged = merge_model_input_shards(
        [
            {
                "parameters": {
                    "vocabulary_strategy": "projection_native",
                    "max_sequence_length": 5,
                    "padding_strategy": "right",
                    "time_resolution": EXPECTED_TIME_RESOLUTION,
                },
                "records": [
                    {
                        "relative_path": "fixtures/b.json",
                        "projection_type": "sequence",
                        "vocabulary_strategy": "projection_native",
                        "time_resolution": EXPECTED_TIME_RESOLUTION,
                        "original_token_count": 1,
                        "tokens": ["x"],
                        "attention_mask": [1],
                    }
                ],
            },
            {
                "parameters": {
                    "vocabulary_strategy": "projection_native",
                    "max_sequence_length": 5,
                    "padding_strategy": "right",
                    "time_resolution": EXPECTED_TIME_RESOLUTION,
                },
                "records": [
                    {
                        "relative_path": "fixtures/a.json",
                        "projection_type": "graph",
                        "vocabulary_strategy": "projection_native",
                        "time_resolution": EXPECTED_TIME_RESOLUTION,
                        "original_token_count": 1,
                        "tokens": ["y"],
                        "attention_mask": [1],
                    }
                ],
            },
        ]
    )

    assert merged.parameters.time_resolution == EXPECTED_TIME_RESOLUTION
    assert [record.relative_path for record in merged.records] == [
        "fixtures/a.json",
        "fixtures/b.json",
    ]


def test_count_training_split_tokens_is_deterministic_and_filters_to_train() -> None:
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            feature_version="feature-v1",
        ),
        records=(
            IrFeatureRecord(
                relative_path="fixtures/validation.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_ONLY,
                    events=(_build_note_event("D", ScoreTime(1, 4)),),
                ),
            ),
            IrFeatureRecord(
                relative_path="fixtures/train.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_ONLY,
                    events=(
                        StructureMarkerSequenceEvent(
                            time=ScoreTime(0, 1),
                            marker_kind=StructureMarkerKind.BAR,
                            entity_id="bar:1",
                        ),
                        _build_note_event("C", ScoreTime(1, 4)),
                    ),
                ),
            ),
        ),
    )
    split_manifest = (
        SplitManifestEntry(
            document_id="fixtures/train.json",
            relative_path="fixtures/train.json",
            split=DatasetSplit.TRAIN,
            group_key="fixtures/train.json",
            split_version="split-v1",
        ),
        SplitManifestEntry(
            document_id="fixtures/validation.json",
            relative_path="fixtures/validation.json",
            split=DatasetSplit.VALIDATION,
            group_key="fixtures/validation.json",
            split_version="split-v1",
        ),
    )
    vocabulary_parameters = VocabularyParameters(time_resolution=96)
    sequence_schema = SequenceSchemaContract()
    policy = {
        "bos": "document",
        "eos": "document",
        "padding_interaction": "outside_boundaries",
        "unknown_token_mapping": "map_to_unk",
    }

    first = count_training_split_tokens(
        feature_set,
        split_manifest,
        sequence_schema,
        vocabulary_parameters,
        policy,
    )
    repeated = count_training_split_tokens(
        feature_set,
        split_manifest,
        sequence_schema,
        vocabulary_parameters,
        policy,
    )

    assert first == repeated
    assert first.feature_version == "feature-v1"
    assert first.split_version == "split-v1"
    assert first.counted_relative_paths == ("fixtures/train.json",)
    assert first.counted_document_count == 1
    assert first.total_token_count == EXPECTED_TRAIN_TOKEN_COUNT
    assert {entry.token: entry.count for entry in first.token_counts} == {
        BOS_TOKEN: 1,
        EOS_TOKEN: 1,
        "STRUCTURE:BAR": 1,
        "TIME_SHIFT:96": 1,
        "NOTE_PITCH:C4": 1,
        "NOTE_DURATION:96": 1,
    }
    assert "NOTE_PITCH:D4" not in {entry.token for entry in first.token_counts}


def test_reduce_vocabulary_is_deterministic_and_assigns_stable_ids() -> None:
    shard_counts = _build_reduction_shard_counts()
    parameters = VocabularyParameters(
        time_resolution=96,
        minimum_frequency=2,
        maximum_size=7,
    )

    first_vocabulary, first_stats, first_metadata = reduce_vocabulary(
        shard_counts,
        parameters,
        split_seed=17,
    )
    repeated_vocabulary, repeated_stats, repeated_metadata = reduce_vocabulary(
        list(reversed(shard_counts)),
        parameters,
        split_seed=17,
    )

    assert first_vocabulary == repeated_vocabulary
    assert first_stats == repeated_stats
    assert first_metadata == repeated_metadata
    assert isinstance(first_metadata, VocabularyMetadata)
    assert first_vocabulary.token_to_id == {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "NOTE_DURATION:96": 4,
        "NOTE_PITCH:C4": 5,
        "STRUCTURE:BAR": 6,
    }
    assert first_vocabulary.id_to_token == (
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "NOTE_DURATION:96",
        "NOTE_PITCH:C4",
        "STRUCTURE:BAR",
    )
    assert {entry.token: entry.count for entry in first_vocabulary.token_counts} == {
        "<pad>": 0,
        "<bos>": 2,
        "<eos>": 2,
        "<unk>": 0,
        "NOTE_DURATION:96": 4,
        "NOTE_PITCH:C4": 3,
        "STRUCTURE:BAR": 3,
    }
    assert first_vocabulary.vocabulary_size == EXPECTED_REDUCED_VOCABULARY_SIZE
    assert first_stats.token_family_coverage[0].family == "NOTE_DURATION"
    assert first_stats.token_family_coverage[-1].family == "STRUCTURE"
    assert first_stats.guardrails.passed is True
    assert first_stats.guardrails.missing_required_token_families == ()
    assert first_stats.guardrails.top_token == "NOTE_DURATION:96"
    assert first_stats.guardrails.top_token_count == EXPECTED_TOP_TOKEN_COUNT
    assert first_stats.guardrails.top_token_fraction == pytest.approx(4 / 14)
    assert first_stats.unk_token_count == 0
    assert first_stats.unk_token_fraction == pytest.approx(0.0)
    assert (
        first_stats.guardrails.estimated_unk_token_count == EXPECTED_DROPPED_TOKEN_COUNT
    )
    assert first_stats.guardrails.estimated_unk_fraction == pytest.approx(3 / 17)


@pytest.mark.parametrize(
    ("guardrails", "expected_message"),
    [
        (
            {"minimum_vocabulary_size": 8},
            "minimum_vocabulary_size=8",
        ),
        (
            {"required_token_families": ("NOTE_DURATION", "NOTE_VELOCITY")},
            "NOTE_VELOCITY",
        ),
        (
            {"maximum_top_token_fraction": 0.2},
            "top token concentration exceeds threshold",
        ),
        (
            {"maximum_unk_fraction": 0.1},
            "estimated <unk> rate exceeds threshold",
        ),
    ],
)
def test_reduce_vocabulary_fails_fast_for_degenerate_guardrail_conditions(
    guardrails: dict[str, object],
    expected_message: str,
) -> None:
    parameters = VocabularyParameters(
        time_resolution=96,
        minimum_frequency=2,
        maximum_size=7,
        guardrails={
            "minimum_vocabulary_size": 7,
            "required_token_families": (
                "NOTE_DURATION",
                "NOTE_PITCH",
                "STRUCTURE",
            ),
            "maximum_top_token_fraction": 0.6,
            "maximum_unk_fraction": 0.25,
            **guardrails,
        },
    )

    with pytest.raises(ValueError, match=expected_message):
        reduce_vocabulary(
            _build_reduction_shard_counts(),
            parameters,
            split_seed=17,
        )


def test_tokenize_features_with_vocabulary_emits_token_ids_for_one_document() -> None:
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            feature_version="feature-v1",
            sequence_schema_version="sequence-schema-v1",
            normalized_ir_version="normalized-v1",
        ),
        records=(
            IrFeatureRecord(
                relative_path="fixtures/example.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS,
                    events=(
                        StructureMarkerSequenceEvent(
                            time=ScoreTime(0, 1),
                            marker_kind=StructureMarkerKind.BAR,
                            entity_id="bar:1",
                            part_id="part:1",
                            staff_id="staff:part:1:1",
                            bar_id="bar:1",
                            voice_lane_id=None,
                        ),
                        _build_note_event("C", ScoreTime(0, 1)),
                        _build_note_event("D", ScoreTime(1, 4)),
                    ),
                ),
            ),
        ),
    )
    model_input = tokenize_features_with_vocabulary(
        feature_set,
        split_manifest=(
            SplitManifestEntry(
                document_id="doc-1",
                relative_path="fixtures/example.json",
                split=DatasetSplit.TRAIN,
                group_key="doc-1",
                split_version="split-v1",
            ),
        ),
        sequence_schema=SequenceSchemaContract(),
        vocabulary={
            "vocabulary_version": "vocab-v1",
            "feature_version": "feature-v1",
            "split_version": "split-v1",
            "token_count": 8,
            "vocabulary_size": 9,
            "token_to_id": {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "STRUCTURE:BAR": 4,
                "NOTE_DURATION:96": 5,
                "NOTE_PITCH:C4": 6,
                "NOTE_PITCH:D4": 7,
                "TIME_SHIFT:96": 8,
            },
            "token_counts": [
                {"token": "<pad>", "count": 0},
                {"token": "<bos>", "count": 1},
                {"token": "<eos>", "count": 1},
                {"token": "<unk>", "count": 0},
                {"token": "STRUCTURE:BAR", "count": 1},
                {"token": "NOTE_DURATION:96", "count": 2},
                {"token": "NOTE_PITCH:C4", "count": 1},
                {"token": "NOTE_PITCH:D4", "count": 1},
                {"token": "TIME_SHIFT:96", "count": 1},
            ],
            "construction_parameters": {
                "time_resolution": 96,
                "minimum_frequency": 1,
                "maximum_size": 65536,
                "special_tokens": {
                    "pad": "<pad>",
                    "bos": "<bos>",
                    "eos": "<eos>",
                    "unk": "<unk>",
                },
            },
            "special_token_policy": {
                "policy_name": "baseline_special_tokens",
                "policy_mode": "baseline_v1",
                "bos": "document",
                "eos": "document",
                "padding_interaction": "outside_boundaries",
                "unknown_token_mapping": "map_to_unk",
            },
        },
        model_input_parameters={
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
            "context_length": 4,
            "stride": 3,
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
        },
    )

    assert isinstance(model_input["parameters"], ModelInputMetadata)
    assert model_input["parameters"].vocabulary_version == "vocab-v1"
    assert model_input["storage_schema"].storage_schema_version == "parquet-v1"
    assert len(model_input["records"]) == 1
    record = model_input["records"][0]
    assert isinstance(record, TokenizedDocumentRow)
    assert record.relative_path == "fixtures/example.json"
    assert record.document_id == "doc-1"
    assert record.split is DatasetSplit.TRAIN
    assert record.projection_type == "sequence"
    assert record.sequence_mode == "baseline_v1"
    assert record.feature_version == "feature-v1"
    assert record.normalized_ir_version == "normalized-v1"
    assert record.vocabulary_version == "vocab-v1"
    assert record.model_input_version == model_input["parameters"].model_input_version
    assert record.token_ids == (1, 4, 6, 5, 8, 7, 5, 2)
    assert record.token_count == len(record.token_ids)
    assert record.window_start_offsets == (0, 3, 4)
    assert record.token_ids[record.window_start_offsets[0]] == 1
    assert record.token_ids[-1] == EXPECTED_EOS_TOKEN_ID


def test_tokenize_features_with_vocabulary_round_trips_one_fixture_backed_document() -> (
    None
):
    document = deserialize_document(
        REPRESENTATIVE_DOCUMENT_FIXTURE.read_text(encoding="utf-8")
    )
    sequence_schema = SequenceSchemaContract()
    feature_set = extract_features(
        normalized_ir_corpus=[
            MotifIrDocumentRecord(
                relative_path="fixtures/representative_document.json",
                document=document,
            )
        ],
        normalized_ir_version={
            "normalized_ir_version": "normalized-v1",
            "contract_name": "motifml.normalized_ir",
            "contract_version": "1.0.0",
            "serialized_document_format": "motifml.ir.document",
            "normalization_strategy": "passthrough_v1",
            "upstream_ir_schema_version": document.metadata.ir_schema_version,
            "task_agnostic_guarantees": (
                "stable_source_relative_identity",
                "task_agnostic_domain_truth",
                "no_model_specific_flattening",
                "no_model_specific_windowing",
            ),
        },
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            sequence_mode="baseline_v1",
        ),
        sequence_schema=sequence_schema,
    )
    projection = feature_set.records[0].projection
    assert isinstance(projection, SequenceProjection)
    token_strings = encode_projected_events_to_tokens(
        projection.events,
        time_resolution=96,
        note_payload_fields=sequence_schema.note_payload_fields,
        special_token_policy={
            "bos": "document",
            "eos": "document",
            "padding_interaction": "outside_boundaries",
            "unknown_token_mapping": "map_to_unk",
        },
    )
    assert feature_set.parameters.feature_version is not None
    vocabulary = _build_vocabulary_artifact(
        token_strings,
        feature_version=feature_set.parameters.feature_version,
    )

    model_input = tokenize_features_with_vocabulary(
        feature_set,
        split_manifest=(
            SplitManifestEntry(
                document_id="fixture-doc",
                relative_path="fixtures/representative_document.json",
                split=DatasetSplit.TRAIN,
                group_key="fixture-doc",
                split_version="split-v1",
            ),
        ),
        sequence_schema=sequence_schema,
        vocabulary=vocabulary,
        model_input_parameters={
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
            "context_length": 256,
            "stride": 128,
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
        },
    )

    record = model_input["records"][0]
    assert decode_token_ids_to_strings(record.token_ids, vocabulary=vocabulary) == (
        token_strings
    )
    assert record.token_count == len(token_strings)
    assert record.window_start_offsets == build_window_start_offsets(
        record.token_ids,
        context_length=256,
        stride=128,
    )
    assert (
        model_input["parameters"].feature_version
        == feature_set.parameters.feature_version
    )


def test_tokenize_features_with_vocabulary_emits_one_window_for_short_documents() -> (
    None
):
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            feature_version="feature-v1",
            sequence_schema_version="sequence-schema-v1",
            normalized_ir_version="normalized-v1",
        ),
        records=(
            IrFeatureRecord(
                relative_path="fixtures/short.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_ONLY,
                    events=(_build_note_event("C", ScoreTime(0, 1)),),
                ),
            ),
        ),
    )

    model_input = tokenize_features_with_vocabulary(
        feature_set,
        split_manifest=(
            SplitManifestEntry(
                document_id="short-doc",
                relative_path="fixtures/short.json",
                split=DatasetSplit.VALIDATION,
                group_key="short-doc",
                split_version="split-v1",
            ),
        ),
        sequence_schema=SequenceSchemaContract(),
        vocabulary={
            "vocabulary_version": "vocab-v1",
            "feature_version": "feature-v1",
            "split_version": "split-v1",
            "token_count": 4,
            "vocabulary_size": 6,
            "token_to_id": {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "NOTE_DURATION:96": 4,
                "NOTE_PITCH:C4": 5,
            },
            "token_counts": [
                {"token": "<pad>", "count": 0},
                {"token": "<bos>", "count": 1},
                {"token": "<eos>", "count": 1},
                {"token": "<unk>", "count": 0},
                {"token": "NOTE_DURATION:96", "count": 1},
                {"token": "NOTE_PITCH:C4", "count": 1},
            ],
            "construction_parameters": {
                "time_resolution": 96,
                "minimum_frequency": 1,
                "maximum_size": 65536,
                "special_tokens": {
                    "pad": "<pad>",
                    "bos": "<bos>",
                    "eos": "<eos>",
                    "unk": "<unk>",
                },
            },
            "special_token_policy": {
                "policy_name": "baseline_special_tokens",
                "policy_mode": "baseline_v1",
                "bos": "document",
                "eos": "document",
                "padding_interaction": "outside_boundaries",
                "unknown_token_mapping": "map_to_unk",
            },
        },
        model_input_parameters={
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
            "context_length": 16,
            "stride": 8,
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
        },
    )

    record = model_input["records"][0]
    assert record.split is DatasetSplit.VALIDATION
    assert record.token_ids == (1, 5, 4, 2)
    assert record.window_start_offsets == (0,)
    assert record.token_ids[0] == 1
    assert record.token_ids[-1] == EXPECTED_EOS_TOKEN_ID


def test_tokenize_features_with_vocabulary_rejects_mismatched_sequence_mode() -> None:
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            sequence_mode="baseline_v1",
            feature_version="feature-v1",
            sequence_schema_version="sequence-schema-v1",
            normalized_ir_version="normalized-v1",
        ),
        records=(
            IrFeatureRecord(
                relative_path="fixtures/example.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_ONLY,
                    events=(),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="sequence_mode must match"):
        tokenize_features_with_vocabulary(
            feature_set,
            split_manifest=(
                SplitManifestEntry(
                    document_id="doc-1",
                    relative_path="fixtures/example.json",
                    split=DatasetSplit.TRAIN,
                    group_key="doc-1",
                    split_version="split-v1",
                ),
            ),
            sequence_schema=SequenceSchemaContract(),
            vocabulary={
                "vocabulary_version": "vocab-v1",
                "feature_version": "feature-v1",
                "split_version": "split-v1",
                "token_count": 2,
                "vocabulary_size": 4,
                "token_to_id": {
                    "<pad>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "<unk>": 3,
                },
                "token_counts": [
                    {"token": "<pad>", "count": 0},
                    {"token": "<bos>", "count": 1},
                    {"token": "<eos>", "count": 1},
                    {"token": "<unk>", "count": 0},
                ],
                "construction_parameters": {
                    "time_resolution": 96,
                    "minimum_frequency": 1,
                    "maximum_size": 65536,
                    "special_tokens": {
                        "pad": "<pad>",
                        "bos": "<bos>",
                        "eos": "<eos>",
                        "unk": "<unk>",
                    },
                },
                "special_token_policy": {
                    "policy_name": "baseline_special_tokens",
                    "policy_mode": "baseline_v1",
                    "bos": "document",
                    "eos": "document",
                    "padding_interaction": "outside_boundaries",
                    "unknown_token_mapping": "map_to_unk",
                },
            },
            model_input_parameters={
                "projection_type": "sequence",
                "sequence_mode": "notes_only",
                "context_length": 8,
                "stride": 4,
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
            },
        )


def test_tokenize_features_with_vocabulary_fails_fast_for_invalid_event_ordering() -> (
    None
):
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            sequence_mode="baseline_v1",
            feature_version="feature-v1",
            sequence_schema_version="sequence-schema-v1",
            normalized_ir_version="normalized-v1",
        ),
        records=(
            IrFeatureRecord(
                relative_path="fixtures/out_of_order.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS,
                    events=(
                        _build_note_event("C", ScoreTime(0, 1)),
                        StructureMarkerSequenceEvent(
                            time=ScoreTime(0, 1),
                            marker_kind=StructureMarkerKind.BAR,
                            entity_id="bar:1",
                            part_id="part:1",
                            staff_id="staff:part:1:1",
                            bar_id="bar:1",
                            voice_lane_id=None,
                        ),
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "document_id=doc-out-of-order.*"
            "relative_path=fixtures/out_of_order.json.*"
            "event_index=0"
        ),
    ):
        tokenize_features_with_vocabulary(
            feature_set,
            split_manifest=(
                SplitManifestEntry(
                    document_id="doc-out-of-order",
                    relative_path="fixtures/out_of_order.json",
                    split=DatasetSplit.TRAIN,
                    group_key="doc-out-of-order",
                    split_version="split-v1",
                ),
            ),
            sequence_schema=SequenceSchemaContract(),
            vocabulary=_build_vocabulary_artifact(
                ("STRUCTURE:BAR", "NOTE_PITCH:C4", "NOTE_DURATION:96"),
                feature_version="feature-v1",
            ),
            model_input_parameters={
                "projection_type": "sequence",
                "sequence_mode": "baseline_v1",
                "context_length": 4,
                "stride": 3,
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
            },
        )


def test_tokenize_features_with_vocabulary_rejects_non_sequence_projection_type() -> (
    None
):
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(
            projection_type="sequence",
            sequence_mode="baseline_v1",
            feature_version="feature-v1",
            sequence_schema_version="sequence-schema-v1",
            normalized_ir_version="normalized-v1",
        ),
        records=(
            IrFeatureRecord(
                relative_path="fixtures/example.json",
                projection_type="sequence",
                projection=SequenceProjection(
                    mode=SequenceProjectionMode.NOTES_ONLY,
                    events=(),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="projection_type must be 'sequence'"):
        tokenize_features_with_vocabulary(
            feature_set,
            split_manifest=(
                SplitManifestEntry(
                    document_id="doc-1",
                    relative_path="fixtures/example.json",
                    split=DatasetSplit.TRAIN,
                    group_key="doc-1",
                    split_version="split-v1",
                ),
            ),
            sequence_schema=SequenceSchemaContract(),
            vocabulary={
                "vocabulary_version": "vocab-v1",
                "feature_version": "feature-v1",
                "split_version": "split-v1",
                "token_count": 2,
                "vocabulary_size": 4,
                "token_to_id": {
                    "<pad>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "<unk>": 3,
                },
                "token_counts": [
                    {"token": "<pad>", "count": 0},
                    {"token": "<bos>", "count": 1},
                    {"token": "<eos>", "count": 1},
                    {"token": "<unk>", "count": 0},
                ],
                "construction_parameters": {
                    "time_resolution": 96,
                    "minimum_frequency": 1,
                    "maximum_size": 65536,
                    "special_tokens": {
                        "pad": "<pad>",
                        "bos": "<bos>",
                        "eos": "<eos>",
                        "unk": "<unk>",
                    },
                },
                "special_token_policy": {
                    "policy_name": "baseline_special_tokens",
                    "policy_mode": "baseline_v1",
                    "bos": "document",
                    "eos": "document",
                    "padding_interaction": "outside_boundaries",
                    "unknown_token_mapping": "map_to_unk",
                },
            },
            model_input_parameters={
                "projection_type": "graph",
                "sequence_mode": "baseline_v1",
                "context_length": 8,
                "stride": 4,
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
            },
        )


def _build_note_event(pitch_step: str, time: ScoreTime) -> NoteSequenceEvent:
    onset_id = f"onset:{pitch_step.lower()}:1"
    return NoteSequenceEvent(
        time=time,
        note=NoteEvent(
            note_id=f"note:{onset_id}:1",
            onset_id=onset_id,
            part_id="part:1",
            staff_id="staff:part:1:1",
            time=time,
            attack_duration=ScoreTime(1, 4),
            sounding_duration=ScoreTime(1, 4),
            pitch=Pitch(step=pitch_step, octave=4),
        ),
        part_id="part:1",
        staff_id="staff:part:1:1",
        bar_id="bar:1",
        voice_lane_id="voice:1",
        onset_id=onset_id,
    )


def _build_reduction_shard_counts() -> list[dict[str, object]]:
    return [
        {
            "feature_version": "feature-v1",
            "split_version": "split-v1",
            "time_resolution": 96,
            "special_token_policy": {
                "bos": "document",
                "eos": "document",
                "padding_interaction": "outside_boundaries",
                "unknown_token_mapping": "map_to_unk",
            },
            "counted_document_count": 1,
            "total_token_count": 10,
            "counted_relative_paths": ["fixtures/a.json"],
            "token_counts": [
                {"token": BOS_TOKEN, "count": 1},
                {"token": EOS_TOKEN, "count": 1},
                {"token": "NOTE_DURATION:96", "count": 3},
                {"token": "NOTE_PITCH:C4", "count": 3},
                {"token": "STRUCTURE:BAR", "count": 2},
            ],
        },
        {
            "feature_version": "feature-v1",
            "split_version": "split-v1",
            "time_resolution": 96,
            "special_token_policy": {
                "bos": "document",
                "eos": "document",
                "padding_interaction": "outside_boundaries",
                "unknown_token_mapping": "map_to_unk",
            },
            "counted_document_count": 1,
            "total_token_count": 7,
            "counted_relative_paths": ["fixtures/b.json"],
            "token_counts": [
                {"token": BOS_TOKEN, "count": 1},
                {"token": EOS_TOKEN, "count": 1},
                {"token": "NOTE_DURATION:96", "count": 1},
                {"token": "STRUCTURE:BAR", "count": 1},
                {"token": "TIME_SHIFT:96", "count": 2},
                {"token": "NOTE_PITCH:D4", "count": 1},
            ],
        },
    ]


def _build_vocabulary_artifact(
    token_strings: tuple[str, ...],
    *,
    feature_version: str,
) -> dict[str, object]:
    counts = Counter(token_strings)
    ordered_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        *sorted(
            token
            for token in counts
            if token not in {"<pad>", "<bos>", "<eos>", "<unk>"}
        ),
    ]
    return {
        "vocabulary_version": "fixture-vocab-v1",
        "feature_version": feature_version,
        "split_version": "split-v1",
        "token_count": sum(counts.values()),
        "vocabulary_size": len(ordered_tokens),
        "token_to_id": {token: index for index, token in enumerate(ordered_tokens)},
        "token_counts": [
            {"token": token, "count": counts.get(token, 0)} for token in ordered_tokens
        ],
        "construction_parameters": {
            "time_resolution": 96,
            "minimum_frequency": 1,
            "maximum_size": 65536,
            "special_tokens": {
                "pad": "<pad>",
                "bos": "<bos>",
                "eos": "<eos>",
                "unk": "<unk>",
            },
        },
        "special_token_policy": {
            "policy_name": "baseline_special_tokens",
            "policy_mode": "baseline_v1",
            "bos": "document",
            "eos": "document",
            "padding_interaction": "outside_boundaries",
            "unknown_token_mapping": "map_to_unk",
        },
    }
