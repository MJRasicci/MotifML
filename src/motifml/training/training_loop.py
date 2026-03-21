"""Core training-loop helpers for MotifML decoder-only baseline runs."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from motifml.datasets.json_dataset import to_json_compatible
from motifml.training.data_loading import TokenWindowBatch

_LOGITS_RANK = 3


class LearningRateSchedulerName(StrEnum):
    """Supported learning-rate scheduler names for baseline training."""

    CONSTANT = "constant"
    COSINE_DECAY = "cosine_decay"
    LINEAR_DECAY = "linear_decay"


@dataclass(frozen=True, slots=True)
class LearningRateSchedulerParameters:
    """Typed configuration for the baseline learning-rate scheduler."""

    name: LearningRateSchedulerName = LearningRateSchedulerName.CONSTANT
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", LearningRateSchedulerName(self.name))
        _require_non_negative_int(self.warmup_steps, "warmup_steps")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the scheduler parameters into a JSON-compatible mapping."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class TrainingLoopParameters:
    """Typed configuration for baseline model training."""

    device: str = "cpu"
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    optimizer: str = "adamw"
    lr_scheduler: LearningRateSchedulerParameters = field(
        default_factory=LearningRateSchedulerParameters
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device", _normalize_non_empty_text(self.device, "device")
        )
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.num_epochs, "num_epochs")
        object.__setattr__(
            self,
            "learning_rate",
            _normalize_positive_float(self.learning_rate, "learning_rate"),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _normalize_non_negative_float(self.weight_decay, "weight_decay"),
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _normalize_non_negative_float(
                self.gradient_clip_norm,
                "gradient_clip_norm",
            ),
        )
        normalized_optimizer = _normalize_non_empty_text(
            self.optimizer, "optimizer"
        ).lower()
        if normalized_optimizer != "adamw":
            raise ValueError(
                "Only the AdamW optimizer is supported for baseline training."
            )
        object.__setattr__(self, "optimizer", normalized_optimizer)
        object.__setattr__(
            self,
            "lr_scheduler",
            coerce_learning_rate_scheduler_parameters(self.lr_scheduler),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the training parameters into a JSON-compatible mapping."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ValidationPassResult:
    """Aggregated masked validation loss for one evaluation pass."""

    average_loss: float
    token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "average_loss",
            _normalize_non_negative_float(self.average_loss, "average_loss"),
        )
        _require_non_negative_int(self.token_count, "token_count")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the validation summary into a JSON-compatible mapping."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class TrainingEpochMetrics:
    """One epoch of aggregated training and validation metrics."""

    epoch_index: int
    train_loss: float
    validation_loss: float
    learning_rate: float
    train_token_count: int
    validation_token_count: int

    def __post_init__(self) -> None:
        _require_non_negative_int(self.epoch_index, "epoch_index")
        object.__setattr__(
            self,
            "train_loss",
            _normalize_non_negative_float(self.train_loss, "train_loss"),
        )
        object.__setattr__(
            self,
            "validation_loss",
            _normalize_non_negative_float(self.validation_loss, "validation_loss"),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _normalize_non_negative_float(self.learning_rate, "learning_rate"),
        )
        _require_non_negative_int(self.train_token_count, "train_token_count")
        _require_non_negative_int(self.validation_token_count, "validation_token_count")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the epoch metrics into a JSON-compatible mapping."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Stable JSON-backed history for one complete training run."""

    epochs: tuple[TrainingEpochMetrics, ...]
    best_epoch_index: int
    best_validation_loss: float

    def __post_init__(self) -> None:
        normalized_epochs = tuple(self.epochs)
        if not normalized_epochs:
            raise ValueError("epochs must contain at least one completed epoch.")
        object.__setattr__(self, "epochs", normalized_epochs)
        _require_non_negative_int(self.best_epoch_index, "best_epoch_index")
        if self.best_epoch_index >= len(normalized_epochs):
            raise ValueError("best_epoch_index must reference one completed epoch.")
        object.__setattr__(
            self,
            "best_validation_loss",
            _normalize_non_negative_float(
                self.best_validation_loss,
                "best_validation_loss",
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the training history into a JSON-compatible mapping."""
        return to_json_compatible(self)


def coerce_learning_rate_scheduler_parameters(
    value: LearningRateSchedulerParameters | Mapping[str, Any],
) -> LearningRateSchedulerParameters:
    """Coerce JSON-loaded scheduler parameters into the typed training config."""
    if isinstance(value, LearningRateSchedulerParameters):
        return value
    return LearningRateSchedulerParameters(
        name=LearningRateSchedulerName(
            str(value.get("name", LearningRateSchedulerName.CONSTANT.value))
        ),
        warmup_steps=int(value.get("warmup_steps", 0)),
    )


def coerce_training_loop_parameters(
    value: TrainingLoopParameters | Mapping[str, Any],
) -> TrainingLoopParameters:
    """Coerce ``params:training`` into the typed baseline training config."""
    if isinstance(value, TrainingLoopParameters):
        return value
    return TrainingLoopParameters(
        device=str(value.get("device", "cpu")),
        batch_size=int(value.get("batch_size", 8)),
        num_epochs=int(value.get("num_epochs", 5)),
        learning_rate=float(value.get("learning_rate", 0.0005)),
        weight_decay=float(value.get("weight_decay", 0.01)),
        gradient_clip_norm=float(value.get("gradient_clip_norm", 1.0)),
        optimizer=str(value.get("optimizer", "adamw")),
        lr_scheduler=coerce_learning_rate_scheduler_parameters(
            value.get("lr_scheduler", {})
        ),
    )


def seed_training_libraries(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch from one explicit config value."""
    normalized_seed = _require_int(seed, "seed")
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_torch_device(device_preference: str) -> torch.device:
    """Resolve a config-driven CPU/CUDA device selection into ``torch.device``."""
    normalized_device = _normalize_non_empty_text(
        device_preference,
        "device_preference",
    ).lower()
    if normalized_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_device == "cpu":
        return torch.device("cpu")
    if normalized_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for training but no CUDA device is available."
            )
        return torch.device(normalized_device)
    raise ValueError(
        "device must be one of: auto, cpu, cuda, or a concrete cuda device."
    )


def build_optimizer(
    model: nn.Module,
    parameters: TrainingLoopParameters | Mapping[str, Any],
) -> Optimizer:
    """Build the baseline AdamW optimizer from the typed training config."""
    typed_parameters = coerce_training_loop_parameters(parameters)
    return AdamW(
        model.parameters(),
        lr=typed_parameters.learning_rate,
        weight_decay=typed_parameters.weight_decay,
    )


def build_lr_scheduler(
    optimizer: Optimizer,
    parameters: TrainingLoopParameters | Mapping[str, Any],
    *,
    total_training_steps: int,
) -> LRScheduler:
    """Build the configured warmup-aware learning-rate scheduler."""
    typed_parameters = coerce_training_loop_parameters(parameters)
    normalized_total_training_steps = _require_positive_int(
        total_training_steps,
        "total_training_steps",
    )

    def lr_lambda(step_index: int) -> float:
        effective_step = max(step_index, 0)
        if (
            typed_parameters.lr_scheduler.warmup_steps > 0
            and effective_step < typed_parameters.lr_scheduler.warmup_steps
        ):
            return (effective_step + 1) / typed_parameters.lr_scheduler.warmup_steps

        post_warmup_steps = max(
            1,
            normalized_total_training_steps
            - typed_parameters.lr_scheduler.warmup_steps,
        )
        progress = min(
            1.0,
            max(
                0.0,
                (effective_step - typed_parameters.lr_scheduler.warmup_steps + 1)
                / post_warmup_steps,
            ),
        )
        if typed_parameters.lr_scheduler.name is LearningRateSchedulerName.CONSTANT:
            return 1.0
        if typed_parameters.lr_scheduler.name is LearningRateSchedulerName.LINEAR_DECAY:
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_causal_language_model_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked next-token cross-entropy for one batch of logits."""
    _validate_loss_inputs(logits, target_ids, attention_mask)
    active_positions = attention_mask.bool()
    if not bool(active_positions.any()):
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[active_positions], target_ids[active_positions])


def run_validation_pass(
    model: nn.Module,
    *,
    validation_batches: Iterable[TokenWindowBatch],
    device: torch.device,
) -> ValidationPassResult:
    """Run one masked validation pass without updating model weights."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in validation_batches:
            logits = model(
                batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            loss = compute_causal_language_model_loss(
                logits,
                batch.target_ids.to(device),
                batch.attention_mask.to(device),
            )
            active_tokens = int(batch.attention_mask.sum().item())
            total_loss += float(loss.item()) * active_tokens
            total_tokens += active_tokens
    if total_tokens <= 0:
        return ValidationPassResult(average_loss=0.0, token_count=0)
    return ValidationPassResult(
        average_loss=total_loss / total_tokens,
        token_count=total_tokens,
    )


def train_decoder_only_model(  # noqa: PLR0913
    model: nn.Module,
    *,
    training_parameters: TrainingLoopParameters | Mapping[str, Any],
    seed: int,
    total_training_steps: int,
    train_batches_for_epoch: Callable[[int], Iterable[TokenWindowBatch]],
    validation_batches_factory: Callable[[], Iterable[TokenWindowBatch]],
) -> TrainingHistory:
    """Run the baseline training loop over lazy token-window batches."""
    typed_parameters = coerce_training_loop_parameters(training_parameters)
    seed_training_libraries(seed)
    device = resolve_torch_device(typed_parameters.device)
    model.to(device)

    optimizer = build_optimizer(model, typed_parameters)
    scheduler = build_lr_scheduler(
        optimizer,
        typed_parameters,
        total_training_steps=total_training_steps,
    )

    history_entries: list[TrainingEpochMetrics] = []
    best_epoch_index = 0
    best_validation_loss = float("inf")
    for epoch_index in range(typed_parameters.num_epochs):
        train_loss, train_token_count = run_training_epoch(
            model,
            train_batches=train_batches_for_epoch(epoch_index),
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=typed_parameters.gradient_clip_norm,
        )
        validation_result = run_validation_pass(
            model,
            validation_batches=validation_batches_factory(),
            device=device,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_entry = TrainingEpochMetrics(
            epoch_index=epoch_index,
            train_loss=train_loss,
            validation_loss=validation_result.average_loss,
            learning_rate=learning_rate,
            train_token_count=train_token_count,
            validation_token_count=validation_result.token_count,
        )
        history_entries.append(history_entry)
        if validation_result.average_loss < best_validation_loss:
            best_validation_loss = validation_result.average_loss
            best_epoch_index = epoch_index

    return TrainingHistory(
        epochs=tuple(history_entries),
        best_epoch_index=best_epoch_index,
        best_validation_loss=best_validation_loss,
    )


def run_training_epoch(  # noqa: PLR0913
    model: nn.Module,
    *,
    train_batches: Iterable[TokenWindowBatch],
    device: torch.device,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    gradient_clip_norm: float,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for batch in train_batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
        )
        loss = compute_causal_language_model_loss(
            logits,
            batch.target_ids.to(device),
            batch.attention_mask.to(device),
        )
        active_tokens = int(batch.attention_mask.sum().item())
        if active_tokens <= 0:
            continue
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item()) * active_tokens
        total_tokens += active_tokens

    if total_tokens <= 0:
        return 0.0, 0
    return total_loss / total_tokens, total_tokens


def _validate_loss_inputs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if logits.ndim != _LOGITS_RANK:
        raise ValueError("logits must be a rank-3 tensor.")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must match the batch and time dimensions.")
    if attention_mask.shape != logits.shape[:2]:
        raise ValueError("attention_mask must match the batch and time dimensions.")


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric.")
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return normalized


def _normalize_non_negative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric.")
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return normalized


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


__all__ = [
    "LearningRateSchedulerName",
    "LearningRateSchedulerParameters",
    "TrainingEpochMetrics",
    "TrainingHistory",
    "TrainingLoopParameters",
    "ValidationPassResult",
    "build_lr_scheduler",
    "build_optimizer",
    "compute_causal_language_model_loss",
    "coerce_learning_rate_scheduler_parameters",
    "coerce_training_loop_parameters",
    "resolve_torch_device",
    "run_training_epoch",
    "run_validation_pass",
    "seed_training_libraries",
    "train_decoder_only_model",
]
