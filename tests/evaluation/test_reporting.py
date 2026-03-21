"""Tests for baseline evaluation reporting helpers."""

from __future__ import annotations

from motifml.evaluation.config import coerce_evaluation_parameters
from motifml.evaluation.reporting import render_qualitative_report_markdown

EXPECTED_BATCH_SIZE = 4
EXPECTED_PROMPT_TOKEN_COUNT = 8
EXPECTED_MAXIMUM_SPLIT_UNK_RATE = 0.25


def test_coerce_evaluation_parameters_builds_typed_nested_config() -> None:
    parameters = coerce_evaluation_parameters(
        {
            "device": "cpu",
            "batch_size": EXPECTED_BATCH_SIZE,
            "top_k": 3,
            "decode_max_tokens": 12,
            "splits": ["validation"],
            "qualitative": {
                "samples_per_split": 1,
                "prompt_token_count": EXPECTED_PROMPT_TOKEN_COUNT,
                "summary_token_limit": 4,
            },
        }
    )

    assert parameters.device == "cpu"
    assert parameters.batch_size == EXPECTED_BATCH_SIZE
    assert tuple(split.value for split in parameters.splits) == ("validation",)
    assert parameters.qualitative.prompt_token_count == EXPECTED_PROMPT_TOKEN_COUNT
    assert (
        parameters.guardrails.maximum_split_unk_rate == EXPECTED_MAXIMUM_SPLIT_UNK_RATE
    )


def test_render_qualitative_report_markdown_includes_metrics_and_samples() -> None:
    report = render_qualitative_report_markdown(
        evaluation_run_id="evaluation-001",
        training_run_id="training-001",
        split_metrics={
            "validation": {
                "quantitative": {
                    "cross_entropy_loss": 1.25,
                    "perplexity": 3.49,
                    "accuracy": 0.5,
                    "top_k": 3,
                    "top_k_accuracy": 0.75,
                },
                "unknown_token_usage": {
                    "unk_token_count": 1,
                    "token_count": 8,
                    "unk_rate": 0.125,
                    "maximum_unk_rate": 0.25,
                },
                "generated_unknown_token_usage": {
                    "unk_token_count": 0,
                    "token_count": 4,
                    "unk_rate": 0.0,
                    "maximum_unk_rate": 0.25,
                },
                "structural": {
                    "valid_transition_rate": 0.9,
                    "boundary_order_pass_rate": 1.0,
                    "out_of_range_pitch_fraction": 0.1,
                    "duration_distribution_total_variation": 0.2,
                },
            }
        },
        samples_by_split={
            "validation": [
                {
                    "document_id": "doc-a",
                    "relative_path": "fixtures/a.json",
                    "prompt_summary": "<bos> A",
                    "reference_summary": "B <eos>",
                    "generated_summary": "C <eos>",
                }
            ]
        },
    )

    assert "# Baseline Evaluation Report" in report
    assert "Cross-Entropy Loss: 1.250000" in report
    assert "Evaluation `<unk>` Rate: 0.125000 (1/8, max 0.250000)" in report
    assert "fixtures/a.json" in report
    assert "Generated Continuation: `C <eos>`" in report
