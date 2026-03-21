"""Tests for typed decoder-only Transformer configuration objects."""

from __future__ import annotations

import pytest

from motifml.model import (
    DecoderOnlyTransformerConfig,
    DecoderOnlyTransformerParameters,
    ModelArchitecture,
    PositionalEncodingType,
    build_decoder_only_transformer_config,
    coerce_decoder_only_transformer_parameters,
)


def test_coerce_decoder_only_transformer_parameters_normalizes_model_params() -> None:
    parameters = coerce_decoder_only_transformer_parameters(
        {
            "architecture": "decoder_only_transformer",
            "embedding_dim": 128,
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "dropout": 0.25,
            "positional_encoding": "sinusoidal",
        }
    )

    assert parameters == DecoderOnlyTransformerParameters(
        architecture=ModelArchitecture.DECODER_ONLY_TRANSFORMER,
        embedding_dim=128,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        dropout=0.25,
        positional_encoding=PositionalEncodingType.SINUSOIDAL,
    )
    assert parameters.to_json_dict() == {
        "architecture": "decoder_only_transformer",
        "embedding_dim": 128,
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.25,
        "positional_encoding": "sinusoidal",
    }


def test_build_decoder_only_transformer_config_joins_model_and_runtime_inputs() -> None:
    config = build_decoder_only_transformer_config(
        {
            "architecture": "decoder_only_transformer",
            "embedding_dim": 64,
            "hidden_size": 256,
            "num_layers": 3,
            "num_heads": 4,
            "dropout": 0.0,
            "positional_encoding": "learned",
        },
        vocabulary_size=1024,
        context_length=128,
        pad_token_id=0,
    )

    assert config == DecoderOnlyTransformerConfig(
        architecture=ModelArchitecture.DECODER_ONLY_TRANSFORMER,
        vocabulary_size=1024,
        context_length=128,
        embedding_dim=64,
        hidden_size=256,
        num_layers=3,
        num_heads=4,
        dropout=0.0,
        positional_encoding=PositionalEncodingType.LEARNED,
        pad_token_id=0,
    )
    assert config.to_json_dict() == {
        "architecture": "decoder_only_transformer",
        "vocabulary_size": 1024,
        "context_length": 128,
        "embedding_dim": 64,
        "hidden_size": 256,
        "num_layers": 3,
        "num_heads": 4,
        "dropout": 0.0,
        "positional_encoding": "learned",
        "pad_token_id": 0,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "architecture": "decoder_only_transformer",
                "embedding_dim": 63,
                "hidden_size": 256,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "positional_encoding": "learned",
            },
            "embedding_dim must be divisible by num_heads",
        ),
        (
            {
                "architecture": "decoder_only_transformer",
                "embedding_dim": 64,
                "hidden_size": 256,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 1.0,
                "positional_encoding": "learned",
            },
            "dropout must satisfy 0.0 <= dropout < 1.0",
        ),
        (
            {
                "architecture": "decoder_only_transformer",
                "embedding_dim": 64,
                "hidden_size": 256,
                "num_layers": 0,
                "num_heads": 4,
                "dropout": 0.1,
                "positional_encoding": "learned",
            },
            "num_layers must be a positive integer",
        ),
    ],
)
def test_decoder_only_transformer_parameters_reject_invalid_values(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        coerce_decoder_only_transformer_parameters(payload)


def test_decoder_only_transformer_config_rejects_invalid_runtime_values() -> None:
    with pytest.raises(ValueError, match="context_length must be a positive integer"):
        DecoderOnlyTransformerConfig(
            architecture=ModelArchitecture.DECODER_ONLY_TRANSFORMER,
            vocabulary_size=256,
            context_length=0,
            embedding_dim=64,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            positional_encoding=PositionalEncodingType.LEARNED,
            pad_token_id=0,
        )
