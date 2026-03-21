"""Evaluation utilities for MotifML baseline model runs."""

from motifml.evaluation.metrics import (
    QuantitativeMetrics,
    QuantitativeMetricTotals,
    accumulate_quantitative_metric_totals,
    build_frequency_baseline_comparison,
    evaluate_causal_language_model,
    finalize_quantitative_metrics,
)
from motifml.evaluation.sampling import (
    QualitativeSample,
    build_prompt_continuation_samples,
    coerce_loaded_tokenized_documents,
    generate_greedy_continuation,
    summarize_decoded_tokens,
)
from motifml.evaluation.structural_checks import (
    DecodedTokenSequence,
    StructuralCheckReport,
    StructuralSequenceFailure,
    coerce_decoded_token_sequences,
    evaluate_structural_quality,
)

__all__ = [
    "DecodedTokenSequence",
    "QuantitativeMetrics",
    "QuantitativeMetricTotals",
    "QualitativeSample",
    "StructuralCheckReport",
    "StructuralSequenceFailure",
    "accumulate_quantitative_metric_totals",
    "build_frequency_baseline_comparison",
    "build_prompt_continuation_samples",
    "coerce_decoded_token_sequences",
    "coerce_loaded_tokenized_documents",
    "evaluate_causal_language_model",
    "evaluate_structural_quality",
    "finalize_quantitative_metrics",
    "generate_greedy_continuation",
    "summarize_decoded_tokens",
]
