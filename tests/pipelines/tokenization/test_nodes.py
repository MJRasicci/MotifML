"""Unit tests for the tokenization pipeline skeleton."""

from __future__ import annotations

from motifml.ir.projections.sequence import SequenceProjection, SequenceProjectionMode
from motifml.pipelines.feature_extraction.models import (
    FeatureExtractionParameters,
    IrFeatureRecord,
    IrFeatureSet,
)
from motifml.pipelines.tokenization.models import (
    PaddingStrategy,
    TokenizationParameters,
)
from motifml.pipelines.tokenization.nodes import (
    merge_model_input_shards,
    tokenize_features,
)
from motifml.training.token_families import PAD_TOKEN

EXPECTED_TIME_RESOLUTION = 48


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
