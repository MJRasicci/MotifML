"""Evaluation utilities for MotifML baseline model runs."""

from motifml.evaluation.config import (
    EvaluationGuardrailParameters,
    EvaluationParameters,
    QualitativeSamplingParameters,
    coerce_evaluation_guardrail_parameters,
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
from motifml.evaluation.unknown_tokens import (
    UnknownTokenUsageReport,
    build_unknown_token_usage_report,
    raise_if_unknown_token_rate_exceeds,
)

__all__ = [
    "DecodedTokenSequence",
    "EvaluationGuardrailParameters",
    "EvaluationParameters",
    "UnknownTokenUsageReport",
    "QuantitativeMetrics",
    "QuantitativeMetricTotals",
    "QualitativeSample",
    "QualitativeSamplingParameters",
    "StructuralCheckReport",
    "StructuralSequenceFailure",
    "accumulate_quantitative_metric_totals",
    "build_frequency_baseline_comparison",
    "build_prompt_continuation_samples",
    "build_unknown_token_usage_report",
    "coerce_evaluation_guardrail_parameters",
    "coerce_evaluation_parameters",
    "coerce_decoded_token_sequences",
    "coerce_loaded_tokenized_documents",
    "coerce_qualitative_sampling_parameters",
    "evaluate_causal_language_model",
    "evaluate_structural_quality",
    "finalize_quantitative_metrics",
    "generate_greedy_continuation",
    "raise_if_unknown_token_rate_exceeds",
    "render_qualitative_report_markdown",
    "summarize_decoded_tokens",
]
