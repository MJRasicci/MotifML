"""Nodes for baseline decoder-only Transformer training."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import torch

from motifml.model import DecoderOnlyTransformer, build_decoder_only_transformer_config
from motifml.training.checkpoints import build_checkpoint_name
from motifml.training.config import freeze_parameter_snapshot
from motifml.training.contracts import TrainingRunMetadata
from motifml.training.model_input_runtime import TokenizedModelInputRuntimeHandle
from motifml.training.token_codec import coerce_frozen_vocabulary
from motifml.training.training_loop import (
    TrainingEpochMetrics,
    TrainingHistory,
    build_lr_scheduler,
    build_optimizer,
    coerce_training_loop_parameters,
    resolve_torch_device,
    run_training_epoch,
    run_validation_pass,
    seed_training_libraries,
)
from motifml.training.versioning import build_contract_version


def train_decoder_only_transformer(
    model_input_runtime: TokenizedModelInputRuntimeHandle,
    vocabulary: Mapping[str, Any],
    model_parameters: Mapping[str, Any],
    training_parameters: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Train the baseline decoder-only Transformer over lazy `05_model_input` rows."""
    typed_training_parameters = coerce_training_loop_parameters(training_parameters)
    typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
    started_at = _timestamp_now()

    model_config = build_decoder_only_transformer_config(
        model_parameters,
        vocabulary_size=len(typed_vocabulary.token_to_id),
        context_length=model_input_runtime.metadata.context_length,
        pad_token_id=typed_vocabulary.token_to_id["<pad>"],
    )
    model_config_snapshot = model_config.to_json_dict()
    training_config_snapshot = freeze_parameter_snapshot(
        {
            "training": typed_training_parameters.to_json_dict(),
            "model_input": model_input_runtime.metadata.to_json_dict(),
            "seed": seed,
        }
    )
    training_run_id = build_contract_version(
        namespace="training_run",
        payload={
            "normalized_ir_version": model_input_runtime.metadata.normalized_ir_version,
            "feature_version": model_input_runtime.metadata.feature_version,
            "vocabulary_version": model_input_runtime.metadata.vocabulary_version,
            "model_input_version": model_input_runtime.metadata.model_input_version,
            "model_config": model_config_snapshot,
            "training_config": training_config_snapshot,
            "seed": seed,
        },
    )

    seed_training_libraries(seed)
    device = resolve_torch_device(typed_training_parameters.device)
    model = DecoderOnlyTransformer(model_config)
    model.to(device)
    optimizer = build_optimizer(model, typed_training_parameters)
    train_batch_count = model_input_runtime.count_batches(
        split="train",
        vocabulary=typed_vocabulary,
        batch_size=typed_training_parameters.batch_size,
        iteration_options={"seed": seed},
    )
    if train_batch_count <= 0:
        raise ValueError("Training split produced no batches for baseline training.")
    scheduler = build_lr_scheduler(
        optimizer,
        typed_training_parameters,
        total_training_steps=train_batch_count * typed_training_parameters.num_epochs,
    )

    history_entries: list[TrainingEpochMetrics] = []
    checkpoint_payloads: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch_index = 0
    for epoch_index in range(typed_training_parameters.num_epochs):
        train_loader = model_input_runtime.build_window_data_loader(
            split="train",
            vocabulary=typed_vocabulary,
            batch_size=typed_training_parameters.batch_size,
            iteration_options={"seed": seed, "epoch": epoch_index},
        )
        train_loss, train_token_count = run_training_epoch(
            model,
            train_batches=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=typed_training_parameters.gradient_clip_norm,
        )
        validation_loader = model_input_runtime.build_window_data_loader(
            split="validation",
            vocabulary=typed_vocabulary,
            batch_size=typed_training_parameters.batch_size,
            iteration_options={"seed": seed},
        )
        validation_result = run_validation_pass(
            model,
            validation_batches=validation_loader,
            device=device,
        )
        history_entry = TrainingEpochMetrics(
            epoch_index=epoch_index,
            train_loss=train_loss,
            validation_loss=validation_result.average_loss,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            train_token_count=train_token_count,
            validation_token_count=validation_result.token_count,
        )
        history_entries.append(history_entry)
        if validation_result.average_loss < best_validation_loss:
            best_validation_loss = validation_result.average_loss
            best_epoch_index = epoch_index
        checkpoint_payloads.append(
            {
                "epoch_index": epoch_index,
                "validation_loss": validation_result.average_loss,
                "checkpoint_name": build_checkpoint_name(epoch_index),
                "saved_at": _timestamp_now(),
                "state": {
                    "training_run_id": training_run_id,
                    "epoch_index": epoch_index,
                    "validation_loss": validation_result.average_loss,
                    "model_state_dict": _move_state_to_cpu(model.state_dict()),
                    "optimizer_state_dict": _move_state_to_cpu(optimizer.state_dict()),
                },
            }
        )

    history = TrainingHistory(
        epochs=tuple(history_entries),
        best_epoch_index=best_epoch_index,
        best_validation_loss=best_validation_loss,
    )
    run_metadata = TrainingRunMetadata(
        training_run_id=training_run_id,
        normalized_ir_version=model_input_runtime.metadata.normalized_ir_version,
        feature_version=model_input_runtime.metadata.feature_version,
        vocabulary_version=model_input_runtime.metadata.vocabulary_version,
        model_input_version=model_input_runtime.metadata.model_input_version,
        seed=seed,
        model_parameters=model_config_snapshot,
        training_parameters=training_config_snapshot,
        started_at=started_at,
        device=str(device),
    )
    checkpoint_bundle = {
        "model_config": model_config_snapshot,
        "training_config": training_config_snapshot,
        "run_metadata": run_metadata,
        "checkpoints": checkpoint_payloads,
    }
    return (
        checkpoint_bundle,
        history.to_json_dict(),
        run_metadata.to_json_dict(),
    )


def _move_state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _move_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _timestamp_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = ["train_decoder_only_transformer"]
