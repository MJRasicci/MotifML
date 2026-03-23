"""Typed contracts for the V1 continuation-example extraction pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible
from motifml.ir.models import TimeSignature
from motifml.ir.time import ScoreTime
from motifml.training.contracts import DatasetSplit


@dataclass(frozen=True, slots=True)
class ContinuationDatasetParameters:
    """Frozen configuration for V1 continuation-example extraction."""

    contract_name: str = "motifml.v1_continuation_task"
    contract_version: str = "1.0.0"
    prompt_bar_count: int = 4
    require_single_part: bool = True
    require_single_staff: bool = True
    require_single_voice_lane_chain: bool = True
    reject_control_events: bool = True
    reject_grace_notes: bool = True
    reject_techniques: bool = True
    require_attack_equals_sounding: bool = True
    require_pitched_notes: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_name", _normalize_text(self.contract_name, "contract_name")
        )
        object.__setattr__(
            self,
            "contract_version",
            _normalize_text(self.contract_version, "contract_version"),
        )
        _require_positive_int(self.prompt_bar_count, "prompt_bar_count")
        for field_name in (
            "require_single_part",
            "require_single_staff",
            "require_single_voice_lane_chain",
            "reject_control_events",
            "reject_grace_notes",
            "reject_techniques",
            "require_attack_equals_sounding",
            "require_pitched_notes",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    def to_version_payload(self) -> dict[str, Any]:
        """Serialize the configuration payload used for versioning."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class PromptNote:
    """One prompt-bar note payload."""

    note_id: str
    note_slot_index: int
    pitch_text: str
    attack_duration: ScoreTime

    def __post_init__(self) -> None:
        object.__setattr__(self, "note_id", _normalize_text(self.note_id, "note_id"))
        _require_non_negative_int(self.note_slot_index, "note_slot_index")
        object.__setattr__(
            self, "pitch_text", _normalize_text(self.pitch_text, "pitch_text")
        )
        self.attack_duration.require_non_negative("attack_duration")
        if self.attack_duration.numerator <= 0:
            raise ValueError("attack_duration must be positive.")


@dataclass(frozen=True, slots=True)
class PromptOnset:
    """One prompt-bar onset slot with concrete note content."""

    onset_id: str
    onset_slot_index: int
    attack_order_in_voice: int
    bar_offset: ScoreTime
    duration_notated: ScoreTime
    is_rest: bool
    notes: tuple[PromptNote, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset_id", _normalize_text(self.onset_id, "onset_id"))
        _require_non_negative_int(self.onset_slot_index, "onset_slot_index")
        _require_non_negative_int(
            self.attack_order_in_voice,
            "attack_order_in_voice",
        )
        self.bar_offset.require_non_negative("bar_offset")
        self.duration_notated.require_non_negative("duration_notated")
        if self.duration_notated.numerator <= 0:
            raise ValueError("duration_notated must be positive.")
        object.__setattr__(self, "is_rest", bool(self.is_rest))
        normalized_notes = tuple(self.notes)
        if self.is_rest and normalized_notes:
            raise ValueError("Rest onsets must not contain prompt notes.")
        object.__setattr__(self, "notes", normalized_notes)


@dataclass(frozen=True, slots=True)
class PromptBar:
    """One fully materialized prompt bar."""

    bar_id: str
    bar_index: int
    duration: ScoreTime
    time_signature: TimeSignature
    onsets: tuple[PromptOnset, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar_id", _normalize_text(self.bar_id, "bar_id"))
        _require_non_negative_int(self.bar_index, "bar_index")
        self.duration.require_non_negative("duration")
        if self.duration.numerator <= 0:
            raise ValueError("duration must be positive.")
        normalized_onsets = tuple(self.onsets)
        object.__setattr__(self, "onsets", normalized_onsets)
        _validate_sorted_unique_offsets(
            tuple(onset.bar_offset for onset in normalized_onsets),
            field_name="prompt onsets",
        )


@dataclass(frozen=True, slots=True)
class TargetNoteSlot:
    """One target-fill note payload aligned to one scaffold slot."""

    note_id: str
    note_slot_index: int
    pitch_text: str
    attack_duration: ScoreTime

    def __post_init__(self) -> None:
        object.__setattr__(self, "note_id", _normalize_text(self.note_id, "note_id"))
        _require_non_negative_int(self.note_slot_index, "note_slot_index")
        object.__setattr__(
            self, "pitch_text", _normalize_text(self.pitch_text, "pitch_text")
        )
        self.attack_duration.require_non_negative("attack_duration")
        if self.attack_duration.numerator <= 0:
            raise ValueError("attack_duration must be positive.")


@dataclass(frozen=True, slots=True)
class TargetOnsetScaffold:
    """One target onset slot with copied structural constraints."""

    onset_id: str
    onset_slot_index: int
    attack_order_in_voice: int
    bar_offset: ScoreTime
    duration_notated: ScoreTime
    is_rest: bool
    note_slot_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset_id", _normalize_text(self.onset_id, "onset_id"))
        _require_non_negative_int(self.onset_slot_index, "onset_slot_index")
        _require_non_negative_int(
            self.attack_order_in_voice,
            "attack_order_in_voice",
        )
        self.bar_offset.require_non_negative("bar_offset")
        self.duration_notated.require_non_negative("duration_notated")
        if self.duration_notated.numerator <= 0:
            raise ValueError("duration_notated must be positive.")
        object.__setattr__(self, "is_rest", bool(self.is_rest))
        _require_non_negative_int(self.note_slot_count, "note_slot_count")
        if self.is_rest and self.note_slot_count != 0:
            raise ValueError("Rest onset scaffold rows must not require note slots.")


@dataclass(frozen=True, slots=True)
class TargetOnsetFill:
    """One target onset's concrete note payloads."""

    onset_id: str
    onset_slot_index: int
    note_slots: tuple[TargetNoteSlot, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset_id", _normalize_text(self.onset_id, "onset_id"))
        _require_non_negative_int(self.onset_slot_index, "onset_slot_index")
        normalized_note_slots = tuple(self.note_slots)
        note_slot_indexes = tuple(
            note.note_slot_index for note in normalized_note_slots
        )
        if note_slot_indexes != tuple(sorted(note_slot_indexes)):
            raise ValueError("Target note slots must be sorted by note_slot_index.")
        if len(note_slot_indexes) != len(set(note_slot_indexes)):
            raise ValueError("Target note slots must be unique within one onset.")
        object.__setattr__(self, "note_slots", normalized_note_slots)


@dataclass(frozen=True, slots=True)
class TargetBarScaffold:
    """Copied structural template for the target bar."""

    bar_id: str
    bar_index: int
    duration: ScoreTime
    time_signature: TimeSignature
    onsets: tuple[TargetOnsetScaffold, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar_id", _normalize_text(self.bar_id, "bar_id"))
        _require_non_negative_int(self.bar_index, "bar_index")
        self.duration.require_non_negative("duration")
        if self.duration.numerator <= 0:
            raise ValueError("duration must be positive.")
        normalized_onsets = tuple(self.onsets)
        object.__setattr__(self, "onsets", normalized_onsets)
        _validate_sorted_unique_offsets(
            tuple(onset.bar_offset for onset in normalized_onsets),
            field_name="target scaffold onsets",
        )


@dataclass(frozen=True, slots=True)
class TargetBarFill:
    """Ground-truth content that fills the target scaffold."""

    bar_id: str
    bar_index: int
    onsets: tuple[TargetOnsetFill, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bar_id", _normalize_text(self.bar_id, "bar_id"))
        _require_non_negative_int(self.bar_index, "bar_index")
        normalized_onsets = tuple(self.onsets)
        onset_indexes = tuple(onset.onset_slot_index for onset in normalized_onsets)
        if onset_indexes != tuple(sorted(onset_indexes)):
            raise ValueError("Target fill onsets must be sorted by onset_slot_index.")
        if len(onset_indexes) != len(set(onset_indexes)):
            raise ValueError("Target fill onsets must be unique.")
        object.__setattr__(self, "onsets", normalized_onsets)


@dataclass(frozen=True, slots=True)
class ContinuationExample:
    """One persisted V1 prompt/scaffold/target continuation example."""

    relative_path: str
    trace_id: str
    example_id: str
    source_relative_path: str
    document_id: str
    split: DatasetSplit
    split_version: str
    normalized_ir_version: str
    continuation_dataset_version: str
    task_contract_name: str
    task_contract_version: str
    part_id: str
    staff_id: str
    voice_lane_chain_id: str
    target_bar_id: str
    target_bar_index: int
    prompt_bars: tuple[PromptBar, ...]
    target_scaffold: TargetBarScaffold
    target_fill: TargetBarFill

    def __post_init__(self) -> None:
        for field_name in (
            "relative_path",
            "trace_id",
            "example_id",
            "source_relative_path",
            "document_id",
            "split_version",
            "normalized_ir_version",
            "continuation_dataset_version",
            "task_contract_name",
            "task_contract_version",
            "part_id",
            "staff_id",
            "voice_lane_chain_id",
            "target_bar_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "split", DatasetSplit(self.split))
        _require_non_negative_int(self.target_bar_index, "target_bar_index")
        normalized_prompt_bars = tuple(self.prompt_bars)
        if not normalized_prompt_bars:
            raise ValueError("prompt_bars must contain at least one prompt bar.")
        prompt_indexes = tuple(bar.bar_index for bar in normalized_prompt_bars)
        if prompt_indexes != tuple(sorted(prompt_indexes)):
            raise ValueError("prompt_bars must be sorted by bar_index.")
        if len(prompt_indexes) != len(set(prompt_indexes)):
            raise ValueError("prompt_bars must not repeat bar_index values.")
        object.__setattr__(self, "prompt_bars", normalized_prompt_bars)
        if self.target_scaffold.bar_id != self.target_bar_id:
            raise ValueError("target_scaffold.bar_id must match target_bar_id.")
        if self.target_scaffold.bar_index != self.target_bar_index:
            raise ValueError("target_scaffold.bar_index must match target_bar_index.")
        if self.target_fill.bar_id != self.target_bar_id:
            raise ValueError("target_fill.bar_id must match target_bar_id.")
        if self.target_fill.bar_index != self.target_bar_index:
            raise ValueError("target_fill.bar_index must match target_bar_index.")
        if normalized_prompt_bars[-1].bar_index >= self.target_bar_index:
            raise ValueError("prompt_bars must precede the target bar.")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the example for JSON persistence."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ContinuationDocumentRejection:
    """One attributable source-document rejection emitted during extraction."""

    relative_path: str
    document_id: str
    split: DatasetSplit
    split_version: str
    rejection_reason: str
    detail: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "relative_path",
            "document_id",
            "split_version",
            "rejection_reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "split", DatasetSplit(self.split))
        if self.detail is not None:
            object.__setattr__(self, "detail", _normalize_text(self.detail, "detail"))

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the rejection for JSON persistence."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ContinuationDatasetMetadata:
    """Shared metadata for the persisted V1 continuation dataset."""

    continuation_dataset_version: str
    normalized_ir_version: str
    split_version: str
    task_contract_name: str
    task_contract_version: str
    prompt_bar_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "continuation_dataset_version",
            "normalized_ir_version",
            "split_version",
            "task_contract_name",
            "task_contract_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )
        _require_positive_int(self.prompt_bar_count, "prompt_bar_count")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the metadata for JSON persistence."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ContinuationDatasetArtifacts:
    """Persisted record-set payload for continuation examples."""

    parameters: ContinuationDatasetMetadata
    records: tuple[ContinuationExample, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", sort_continuation_examples(self.records))

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the artifact payload for record-set persistence."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ContinuationDatasetReport:
    """Aggregate build report for V1 continuation-example extraction."""

    continuation_dataset_version: str
    normalized_ir_version: str
    split_version: str
    source_document_count: int
    source_document_used_count: int
    emitted_example_count: int
    split_example_counts: dict[str, int]
    rejection_counts_by_reason: dict[str, int]
    rejections: tuple[ContinuationDocumentRejection, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "continuation_dataset_version",
            "normalized_ir_version",
            "split_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_text(getattr(self, field_name), field_name),
            )
        for field_name in (
            "source_document_count",
            "source_document_used_count",
            "emitted_example_count",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        if self.source_document_used_count > self.source_document_count:
            raise ValueError(
                "source_document_used_count must not exceed source_document_count."
            )
        normalized_split_counts = _normalize_count_mapping(
            self.split_example_counts,
            "split_example_counts",
        )
        if sum(normalized_split_counts.values()) != self.emitted_example_count:
            raise ValueError("split_example_counts must sum to emitted_example_count.")
        object.__setattr__(self, "split_example_counts", normalized_split_counts)
        normalized_rejections = sort_continuation_rejections(self.rejections)
        object.__setattr__(self, "rejections", normalized_rejections)
        normalized_rejection_counts = _normalize_count_mapping(
            self.rejection_counts_by_reason,
            "rejection_counts_by_reason",
        )
        actual_rejection_counts: dict[str, int] = {}
        for rejection in normalized_rejections:
            actual_rejection_counts[rejection.rejection_reason] = (
                actual_rejection_counts.get(rejection.rejection_reason, 0) + 1
            )
        if normalized_rejection_counts != actual_rejection_counts:
            raise ValueError(
                "rejection_counts_by_reason must match the emitted rejection records."
            )
        object.__setattr__(
            self,
            "rejection_counts_by_reason",
            normalized_rejection_counts,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the report for JSON persistence."""
        return to_json_compatible(self)


def coerce_continuation_dataset_parameters(
    value: ContinuationDatasetParameters | Mapping[str, Any],
) -> ContinuationDatasetParameters:
    """Coerce Kedro-loaded configuration mappings into typed parameters."""
    if isinstance(value, ContinuationDatasetParameters):
        return value
    return ContinuationDatasetParameters(
        contract_name=str(value.get("contract_name", "motifml.v1_continuation_task")),
        contract_version=str(value.get("contract_version", "1.0.0")),
        prompt_bar_count=int(value.get("prompt_bar_count", 4)),
        require_single_part=bool(value.get("require_single_part", True)),
        require_single_staff=bool(value.get("require_single_staff", True)),
        require_single_voice_lane_chain=bool(
            value.get("require_single_voice_lane_chain", True)
        ),
        reject_control_events=bool(value.get("reject_control_events", True)),
        reject_grace_notes=bool(value.get("reject_grace_notes", True)),
        reject_techniques=bool(value.get("reject_techniques", True)),
        require_attack_equals_sounding=bool(
            value.get("require_attack_equals_sounding", True)
        ),
        require_pitched_notes=bool(value.get("require_pitched_notes", True)),
    )


def sort_continuation_examples(
    examples: Sequence[ContinuationExample],
) -> tuple[ContinuationExample, ...]:
    """Sort continuation examples into deterministic persisted order."""
    return tuple(
        sorted(
            tuple(examples),
            key=lambda example: (
                example.split.value,
                example.source_relative_path.casefold(),
                example.target_bar_index,
                example.voice_lane_chain_id.casefold(),
                example.relative_path.casefold(),
            ),
        )
    )


def sort_continuation_rejections(
    rejections: Sequence[ContinuationDocumentRejection],
) -> tuple[ContinuationDocumentRejection, ...]:
    """Sort document rejections into deterministic persisted order."""
    return tuple(
        sorted(
            tuple(rejections),
            key=lambda rejection: (
                rejection.split.value,
                rejection.relative_path.casefold(),
                rejection.rejection_reason.casefold(),
            ),
        )
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _normalize_count_mapping(
    value: Mapping[str, int],
    field_name: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    normalized: dict[str, int] = {}
    for key, count in sorted(value.items(), key=lambda item: str(item[0])):
        normalized_key = _normalize_text(str(key), field_name)
        normalized_count = int(count)
        if normalized_count < 0:
            raise ValueError(f"{field_name} counts must be non-negative.")
        normalized[normalized_key] = normalized_count
    return normalized


def _validate_sorted_unique_offsets(
    offsets: Sequence[ScoreTime],
    *,
    field_name: str,
) -> None:
    offset_tuple = tuple(offsets)
    if offset_tuple != tuple(sorted(offset_tuple)):
        raise ValueError(f"{field_name} must be sorted by bar_offset.")
    if len(offset_tuple) != len(set(offset_tuple)):
        raise ValueError(f"{field_name} must not repeat bar_offset values.")


__all__ = [
    "ContinuationDatasetArtifacts",
    "ContinuationDatasetMetadata",
    "ContinuationDatasetParameters",
    "ContinuationDatasetReport",
    "ContinuationDocumentRejection",
    "ContinuationExample",
    "PromptBar",
    "PromptNote",
    "PromptOnset",
    "TargetBarFill",
    "TargetBarScaffold",
    "TargetNoteSlot",
    "TargetOnsetFill",
    "TargetOnsetScaffold",
    "coerce_continuation_dataset_parameters",
    "sort_continuation_examples",
    "sort_continuation_rejections",
]
