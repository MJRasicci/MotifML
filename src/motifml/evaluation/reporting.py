"""Human-reviewable reporting helpers for baseline evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_qualitative_report_markdown(
    *,
    evaluation_run_id: str,
    training_run_id: str,
    split_metrics: Mapping[str, Mapping[str, Any]],
    samples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Render one stable Markdown summary for baseline evaluation artifacts."""
    lines = [
        "# Baseline Evaluation Report",
        "",
        f"- Evaluation Run: `{evaluation_run_id}`",
        f"- Training Run: `{training_run_id}`",
        "",
    ]

    for split_name in sorted(split_metrics):
        split_payload = split_metrics[split_name]
        quantitative = split_payload["quantitative"]
        structural = split_payload["structural"]
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "### Quantitative Metrics",
                "",
                f"- Cross-Entropy Loss: {quantitative['cross_entropy_loss']:.6f}",
                f"- Perplexity: {quantitative['perplexity']:.6f}",
                f"- Accuracy: {quantitative['accuracy']:.6f}",
                f"- Top-{quantitative['top_k']} Accuracy: {quantitative['top_k_accuracy']:.6f}",
                "",
                "### Structural Checks",
                "",
                f"- Valid Transition Rate: {structural['valid_transition_rate']:.6f}",
                f"- Boundary-Order Pass Rate: {structural['boundary_order_pass_rate']:.6f}",
                f"- Generated `<unk>` Rate: {structural['generated_unk_rate']:.6f}",
                f"- Pitch Out-of-Range Fraction: {structural['out_of_range_pitch_fraction']:.6f}",
                "- Duration Distribution TV Distance: "
                f"{structural['duration_distribution_total_variation']:.6f}",
                "",
                "### Samples",
                "",
            ]
        )

        split_samples = samples_by_split.get(split_name, ())
        if not split_samples:
            lines.extend(["No qualitative samples were produced for this split.", ""])
            continue

        for sample_index, sample in enumerate(split_samples, start=1):
            lines.extend(
                [
                    f"#### Sample {sample_index}: `{sample['relative_path']}`",
                    "",
                    f"- Document ID: `{sample['document_id']}`",
                    f"- Prompt: `{sample['prompt_summary']}`",
                    f"- Reference Continuation: `{sample['reference_summary']}`",
                    f"- Generated Continuation: `{sample['generated_summary']}`",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_qualitative_report_markdown"]
