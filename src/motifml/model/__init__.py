"""Model configuration and architecture definitions for MotifML."""

from motifml.model.baselines import (
    FrequencyNextTokenBaseline,
    FrequencyNextTokenBaselineMetrics,
    build_baseline_comparison_report,
)
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
    "FrequencyNextTokenBaseline",
    "FrequencyNextTokenBaselineMetrics",
    "ModelArchitecture",
    "PositionalEncodingType",
    "build_baseline_comparison_report",
    "build_causal_attention_mask",
    "build_decoder_only_transformer_config",
    "coerce_decoder_only_transformer_parameters",
]
