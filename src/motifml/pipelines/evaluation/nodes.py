"""Nodes for baseline decoder-only Transformer evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from motifml.evaluation.config import coerce_evaluation_parameters
from motifml.evaluation.metrics import (
    build_frequency_baseline_comparison,
    evaluate_causal_language_model,
)
from motifml.evaluation.reporting import render_qualitative_report_markdown
from motifml.evaluation.sampling import (
    QualitativeSample,
    build_prompt_continuation_samples,
)
from motifml.evaluation.structural_checks import (
    DecodedTokenSequence,
    evaluate_structural_quality,
)
from motifml.evaluation.unknown_tokens import (
    build_unknown_token_usage_report,
    raise_if_unknown_token_rate_exceeds,
)
from motifml.model import DecoderOnlyTransformer
from motifml.model.config import DecoderOnlyTransformerConfig
from motifml.training.config import freeze_parameter_snapshot
from motifml.training.contracts import EvaluationRunMetadata, TrainingRunMetadata
from motifml.training.model_input_runtime import TokenizedModelInputRuntimeHandle
from motifml.training.token_codec import coerce_frozen_vocabulary
from motifml.training.training_loop import resolve_torch_device, seed_training_libraries
from motifml.training.versioning import build_contract_version


def evaluate_decoder_only_transformer(
    training_artifacts: Mapping[str, Any],
    model_input_runtime: TokenizedModelInputRuntimeHandle,
    vocabulary: Mapping[str, Any],
    evaluation_parameters: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """Evaluate the best persisted baseline checkpoint over configured splits."""
    typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
    typed_evaluation_parameters = coerce_evaluation_parameters(evaluation_parameters)
    training_run_metadata = _coerce_training_run_metadata(
        training_artifacts.get("run_metadata")
    )
    model_config = DecoderOnlyTransformerConfig(**training_artifacts["model_config"])
    evaluation_parameter_snapshot = freeze_parameter_snapshot(
        typed_evaluation_parameters.to_json_dict()
    )
    evaluation_run_id = build_contract_version(
        namespace="evaluation_run",
        payload={
            "training_run_id": training_run_metadata.training_run_id,
            "feature_version": training_run_metadata.feature_version,
            "vocabulary_version": training_run_metadata.vocabulary_version,
            "model_input_version": training_run_metadata.model_input_version,
            "evaluation_parameters": evaluation_parameter_snapshot,
            "seed": seed,
        },
    )

    seed_training_libraries(seed)
    device = resolve_torch_device(typed_evaluation_parameters.device)
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(
        _best_checkpoint_state(training_artifacts)["model_state_dict"]
    )
    model.to(device)

    split_metrics: dict[str, Any] = {}
    samples_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in typed_evaluation_parameters.splits:
        quantitative_metrics = evaluate_causal_language_model(
            model,
            evaluation_batches=model_input_runtime.build_window_data_loader(
                split=split,
                vocabulary=typed_vocabulary,
                batch_size=typed_evaluation_parameters.batch_size,
                iteration_options=_ordered_iteration_options(),
            ),
            device=device,
            top_k=typed_evaluation_parameters.top_k,
        )
        baseline_comparison = build_frequency_baseline_comparison(
            training_token_sequences=_iter_token_sequences(
                model_input_runtime.build_document_dataset(
                    split="train",
                    iteration_options=_ordered_iteration_options(),
                )
            ),
            evaluation_token_sequences=_iter_token_sequences(
                model_input_runtime.build_document_dataset(
                    split=split,
                    iteration_options=_ordered_iteration_options(),
                )
            ),
            model_metrics=quantitative_metrics,
            top_k=typed_evaluation_parameters.top_k,
        )
        qualitative_samples = build_prompt_continuation_samples(
            model,
            documents=model_input_runtime.build_document_dataset(
                split=split,
                iteration_options=_ordered_iteration_options(),
            ),
            vocabulary=typed_vocabulary,
            samples_per_split=typed_evaluation_parameters.qualitative.samples_per_split,
            prompt_token_count=typed_evaluation_parameters.qualitative.prompt_token_count,
            continuation_token_count=typed_evaluation_parameters.decode_max_tokens,
            summary_token_limit=typed_evaluation_parameters.qualitative.summary_token_limit,
            device=device,
            context_length=model_config.context_length,
            eos_token_id=typed_vocabulary.token_to_id["<eos>"],
        )
        structural_report = evaluate_structural_quality(
            _generated_sequences(split.value, qualitative_samples),
            reference_sequences=_reference_sequences(split.value, qualitative_samples),
        )
        split_unknown_token_usage = build_unknown_token_usage_report(
            _iter_token_sequences(
                model_input_runtime.build_document_dataset(
                    split=split,
                    iteration_options=_ordered_iteration_options(),
                )
            ),
            unk_token=typed_vocabulary.token_to_id["<unk>"],
            maximum_unk_rate=(
                typed_evaluation_parameters.guardrails.maximum_split_unk_rate
            ),
        )
        generated_unknown_token_usage = build_unknown_token_usage_report(
            (sample.generated_continuation_tokens for sample in qualitative_samples),
            unk_token="<unk>",
            maximum_unk_rate=(
                typed_evaluation_parameters.guardrails.maximum_generated_unk_rate
            ),
        )
        raise_if_unknown_token_rate_exceeds(
            split_unknown_token_usage,
            context=f"{split.value} evaluation split",
        )
        raise_if_unknown_token_rate_exceeds(
            generated_unknown_token_usage,
            context=f"{split.value} generated samples",
        )
        split_metrics[split.value] = {
            "quantitative": quantitative_metrics.to_json_dict(),
            "baseline_comparison": baseline_comparison,
            "unknown_token_usage": split_unknown_token_usage.to_json_dict(),
            "generated_unknown_token_usage": (
                generated_unknown_token_usage.to_json_dict()
            ),
            "structural": structural_report.to_json_dict(),
        }
        samples_by_split[split.value] = [
            sample.to_json_dict() for sample in qualitative_samples
        ]

    evaluation_run_metadata = EvaluationRunMetadata(
        evaluation_run_id=evaluation_run_id,
        training_run_id=training_run_metadata.training_run_id,
        feature_version=training_run_metadata.feature_version,
        vocabulary_version=training_run_metadata.vocabulary_version,
        model_input_version=training_run_metadata.model_input_version,
        evaluation_parameters=evaluation_parameter_snapshot,
        evaluated_splits=typed_evaluation_parameters.splits,
        started_at=_timestamp_now(),
    )
    metrics_payload = {
        "evaluation_run_id": evaluation_run_id,
        "training_run_id": training_run_metadata.training_run_id,
        "best_checkpoint": training_artifacts["best_checkpoint"],
        "splits": split_metrics,
    }
    samples_payload = {
        "evaluation_run_id": evaluation_run_id,
        "training_run_id": training_run_metadata.training_run_id,
        "samples": samples_by_split,
    }
    qualitative_report = render_qualitative_report_markdown(
        evaluation_run_id=evaluation_run_id,
        training_run_id=training_run_metadata.training_run_id,
        split_metrics=split_metrics,
        samples_by_split=samples_by_split,
    )
    return (
        samples_payload,
        metrics_payload,
        qualitative_report,
        evaluation_run_metadata.to_json_dict(),
    )


def _best_checkpoint_state(training_artifacts: Mapping[str, Any]) -> Mapping[str, Any]:
    best_checkpoint_name = str(training_artifacts["best_checkpoint"]["checkpoint_name"])
    for checkpoint in training_artifacts["checkpoints"]:
        if str(checkpoint["checkpoint_name"]) == best_checkpoint_name:
            return checkpoint["state"]
    raise ValueError(
        "training_artifacts is missing the best checkpoint payload: "
        f"{best_checkpoint_name}."
    )


def _coerce_training_run_metadata(value: Any) -> TrainingRunMetadata:
    if isinstance(value, TrainingRunMetadata):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("training_artifacts.run_metadata must be a mapping payload.")
    return TrainingRunMetadata.from_json_dict(value)


def _iter_token_sequences(documents: Iterable[Any]) -> Iterable[tuple[int, ...]]:
    for document in documents:
        yield tuple(document.row.token_ids)


def _generated_sequences(
    split_name: str,
    samples: Sequence[QualitativeSample],
) -> tuple[DecodedTokenSequence, ...]:
    return tuple(
        DecodedTokenSequence(
            sequence_id=(
                f"{split_name}:{sample.document_id}:"
                f"{sample.relative_path}:generated"
            ),
            tokens=sample.generated_continuation_tokens,
        )
        for sample in samples
    )


def _reference_sequences(
    split_name: str,
    samples: Sequence[QualitativeSample],
) -> tuple[DecodedTokenSequence, ...]:
    return tuple(
        DecodedTokenSequence(
            sequence_id=(
                f"{split_name}:{sample.document_id}:"
                f"{sample.relative_path}:reference"
            ),
            tokens=sample.reference_continuation_tokens,
        )
        for sample in samples
    )


def _ordered_iteration_options() -> dict[str, bool]:
    return {
        "shuffle_shards": False,
        "shuffle_documents": False,
        "shuffle_windows": False,
    }


def _timestamp_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = ["evaluate_decoder_only_transformer"]
