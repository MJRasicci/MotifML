"""Tests for baseline training-loop helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from motifml.training.data_loading import TokenWindowBatch
from motifml.training.training_loop import (
    LearningRateSchedulerName,
    LearningRateSchedulerParameters,
    TrainingLoopParameters,
    build_lr_scheduler,
    build_optimizer,
    coerce_training_loop_parameters,
    compute_causal_language_model_loss,
    resolve_torch_device,
    run_validation_pass,
    seed_training_libraries,
)

EXPECTED_BATCH_SIZE = 4
TOTAL_TRAINING_STEPS = 4


def test_coerce_training_loop_parameters_normalizes_scheduler_config() -> None:
    parameters = coerce_training_loop_parameters(
        {
            "device": "cpu",
            "batch_size": EXPECTED_BATCH_SIZE,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "gradient_clip_norm": 0.5,
            "optimizer": "adamw",
            "lr_scheduler": {
                "name": "cosine_decay",
                "warmup_steps": 1,
            },
        }
    )

    assert parameters == TrainingLoopParameters(
        device="cpu",
        batch_size=EXPECTED_BATCH_SIZE,
        num_epochs=2,
        learning_rate=0.001,
        weight_decay=0.1,
        gradient_clip_norm=0.5,
        optimizer="adamw",
        lr_scheduler=LearningRateSchedulerParameters(
            name=LearningRateSchedulerName.COSINE_DECAY,
            warmup_steps=1,
        ),
    )


def test_build_optimizer_and_scheduler_wire_adamw_with_cosine_decay() -> None:
    model = torch.nn.Linear(3, 2)
    parameters = TrainingLoopParameters(
        device="cpu",
        batch_size=2,
        num_epochs=2,
        learning_rate=0.01,
        weight_decay=0.1,
        gradient_clip_norm=1.0,
        optimizer="adamw",
        lr_scheduler=LearningRateSchedulerParameters(
            name=LearningRateSchedulerName.COSINE_DECAY,
            warmup_steps=1,
        ),
    )

    optimizer = build_optimizer(model, parameters)
    scheduler = build_lr_scheduler(
        optimizer,
        parameters,
        total_training_steps=TOTAL_TRAINING_STEPS,
    )

    learning_rates: list[float] = []
    for _ in range(TOTAL_TRAINING_STEPS):
        optimizer.step()
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

    assert isinstance(optimizer, torch.optim.AdamW)
    assert learning_rates[0] > learning_rates[-1]


def test_compute_causal_language_model_loss_ignores_masked_positions() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    target_ids = torch.tensor([[0, 2]], dtype=torch.long)
    attention_mask = torch.tensor([[True, False]], dtype=torch.bool)

    loss = compute_causal_language_model_loss(logits, target_ids, attention_mask)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :1, :].reshape(-1, 3), torch.tensor([0])
    )

    torch.testing.assert_close(loss, expected)


def test_seed_training_libraries_reseeds_python_numpy_and_torch() -> None:
    seed_training_libraries(17)
    first_random = random.random()
    first_numpy = float(np.random.rand())
    first_torch = float(torch.rand(1).item())

    seed_training_libraries(17)
    second_random = random.random()
    second_numpy = float(np.random.rand())
    second_torch = float(torch.rand(1).item())

    assert first_random == second_random
    assert first_numpy == second_numpy
    assert first_torch == second_torch


def test_resolve_torch_device_supports_cpu_and_auto() -> None:
    assert resolve_torch_device("cpu").type == "cpu"
    assert resolve_torch_device("auto").type in {"cpu", "cuda"}


def test_run_validation_pass_aggregates_masked_loss() -> None:
    model = _FixedLogitModel(
        logits=torch.tensor(
            [
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
    )
    batch = TokenWindowBatch(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        target_ids=torch.tensor([[0, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[True, False]], dtype=torch.bool),
        splits=("validation",),
        shard_ids=("global",),
        relative_paths=("fixtures/example.json",),
        document_ids=("doc-1",),
        window_indices=(0,),
        window_start_offsets=(0,),
    )

    result = run_validation_pass(
        model,
        validation_batches=(batch,),
        device=torch.device("cpu"),
    )

    assert result.token_count == 1
    assert result.average_loss >= 0.0


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, *, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids, attention_mask
        return self._logits


def test_resolve_torch_device_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_torch_device("tpu")
