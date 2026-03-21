"""Model configuration and architecture definitions for MotifML."""

from motifml.model.config import (
    DecoderOnlyTransformerConfig,
    DecoderOnlyTransformerParameters,
    ModelArchitecture,
    PositionalEncodingType,
    build_decoder_only_transformer_config,
    coerce_decoder_only_transformer_parameters,
)

__all__ = [
    "DecoderOnlyTransformerConfig",
    "DecoderOnlyTransformerParameters",
    "ModelArchitecture",
    "PositionalEncodingType",
    "build_decoder_only_transformer_config",
    "coerce_decoder_only_transformer_parameters",
]
