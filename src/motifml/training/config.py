"""Shared configuration snapshot helpers for MotifML training work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible

_REQUIRED_PARAMETER_FAMILIES = (
    "data_split",
    "sequence_schema",
    "vocabulary",
    "model_input",
    "model",
    "training",
    "evaluation",
    "seed",
)


@dataclass(frozen=True, slots=True)
class TrainingParameterSnapshots:
    """Frozen parameter snapshots for the training pipeline lifecycle."""

    split_generation: dict[str, Any]
    vocabulary_construction: dict[str, Any]
    model_input_persistence: dict[str, Any]
    training_run: dict[str, Any]
    evaluation_run: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the parameter snapshots for JSON persistence."""
        return to_json_compatible(self)


def freeze_parameter_snapshot(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize one parameter surface into a stable JSON-ready snapshot."""
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be provided as a mapping.")
    return _canonicalize_json_value(dict(parameters))


def build_parameter_snapshots(
    parameters: Mapping[str, Any],
) -> TrainingParameterSnapshots:
    """Build the frozen parameter snapshots required by the training design."""
    missing = tuple(
        family_name
        for family_name in _REQUIRED_PARAMETER_FAMILIES
        if family_name not in parameters
    )
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Training parameter families are incomplete. Missing: " f"{missing_text}."
        )

    seed = parameters["seed"]
    return TrainingParameterSnapshots(
        split_generation=freeze_parameter_snapshot(
            {
                "data_split": parameters["data_split"],
                "seed": seed,
            }
        ),
        vocabulary_construction=freeze_parameter_snapshot(
            {
                "data_split": parameters["data_split"],
                "sequence_schema": parameters["sequence_schema"],
                "vocabulary": parameters["vocabulary"],
                "seed": seed,
            }
        ),
        model_input_persistence=freeze_parameter_snapshot(
            {
                "sequence_schema": parameters["sequence_schema"],
                "vocabulary": parameters["vocabulary"],
                "model_input": parameters["model_input"],
                "seed": seed,
            }
        ),
        training_run=freeze_parameter_snapshot(
            {
                "model": parameters["model"],
                "model_input": parameters["model_input"],
                "training": parameters["training"],
                "seed": seed,
            }
        ),
        evaluation_run=freeze_parameter_snapshot(
            {
                "evaluation": parameters["evaluation"],
                "model_input": parameters["model_input"],
                "seed": seed,
            }
        ),
    )


def _canonicalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, list | tuple):
        return [_canonicalize_json_value(item) for item in value]

    if isinstance(value, set | frozenset):
        return sorted(
            (_canonicalize_json_value(item) for item in value),
            key=lambda item: repr(item),
        )

    return value


__all__ = [
    "TrainingParameterSnapshots",
    "build_parameter_snapshots",
    "freeze_parameter_snapshot",
]
