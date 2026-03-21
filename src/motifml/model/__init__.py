"""Model configuration and architecture definitions for MotifML."""

from motifml.model.config import (
    DecoderOnlyTransformerConfig,
    DecoderOnlyTransformerParameters,
    ModelArchitecture,
    PositionalEncodingType,
    build_decoder_only_transformer_config,
    coerce_decoder_only_transformer_parameters,
)
from motifml.model.decoder_only_transformer import (
    DecoderOnlyTransformer,
    build_causal_attention_mask,
)

__all__ = [
    "DecoderOnlyTransformerConfig",
    "DecoderOnlyTransformerParameters",
    "DecoderOnlyTransformer",
    "ModelArchitecture",
    "PositionalEncodingType",
    "build_causal_attention_mask",
    "build_decoder_only_transformer_config",
    "coerce_decoder_only_transformer_parameters",
]
