"""Evaluation utilities for MotifML baseline model runs."""

from motifml.evaluation.config import (
    EvaluationParameters,
    QualitativeSamplingParameters,
    coerce_evaluation_parameters,
    coerce_qualitative_sampling_parameters,
)
from motifml.evaluation.metrics import (
    QuantitativeMetrics,
    QuantitativeMetricTotals,
    accumulate_quantitative_metric_totals,
    build_frequency_baseline_comparison,
    evaluate_causal_language_model,
    finalize_quantitative_metrics,
)
from motifml.evaluation.reporting import render_qualitative_report_markdown
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
    "EvaluationParameters",
    "QuantitativeMetrics",
    "QuantitativeMetricTotals",
    "QualitativeSample",
    "QualitativeSamplingParameters",
    "StructuralCheckReport",
    "StructuralSequenceFailure",
    "accumulate_quantitative_metric_totals",
    "build_frequency_baseline_comparison",
    "build_prompt_continuation_samples",
    "coerce_evaluation_parameters",
    "coerce_decoded_token_sequences",
    "coerce_loaded_tokenized_documents",
    "coerce_qualitative_sampling_parameters",
    "evaluate_causal_language_model",
    "evaluate_structural_quality",
    "finalize_quantitative_metrics",
    "generate_greedy_continuation",
    "render_qualitative_report_markdown",
    "summarize_decoded_tokens",
]
