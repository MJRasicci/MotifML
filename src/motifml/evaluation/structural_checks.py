"""Decoded-sequence structural checks for baseline qualitative evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible
from motifml.ir.projections.sequence import StructureMarkerKind
from motifml.training.token_families import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SEGMENT_SEPARATOR,
    UNK_TOKEN,
    TokenFamily,
)

_PITCH_CLASS_BY_STEP = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_ACCIDENTAL_OFFSETS = {
    "": 0,
    "#": 1,
    "##": 2,
    "B": -1,
    "BB": -2,
    "N": 0,
}
_STRUCTURE_ORDER = {
    StructureMarkerKind.PART.value.upper(): 0,
    StructureMarkerKind.STAFF.value.upper(): 1,
    StructureMarkerKind.BAR.value.upper(): 2,
    StructureMarkerKind.VOICE_LANE.value.upper(): 3,
    StructureMarkerKind.ONSET_GROUP.value.upper(): 4,
}
_SPECIAL_START_TOKENS = frozenset({BOS_TOKEN, UNK_TOKEN})
_SPECIAL_END_TOKENS = frozenset({EOS_TOKEN, UNK_TOKEN})
_TOKEN_PART_COUNT_WITH_PAYLOAD = 2


@dataclass(frozen=True, slots=True)
class DecodedTokenSequence:
    """One decoded token sequence with a stable report identifier."""

    sequence_id: str
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_id",
            _normalize_non_empty_text(self.sequence_id, "sequence_id"),
        )
        object.__setattr__(
            self,
            "tokens",
            tuple(_normalize_non_empty_text(token, "token") for token in self.tokens),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the decoded token sequence for report artifacts."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class StructuralSequenceFailure:
    """One explicit structural failure surfaced for human review."""

    sequence_id: str
    check_name: str
    token_index: int
    token: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_id",
            _normalize_non_empty_text(self.sequence_id, "sequence_id"),
        )
        object.__setattr__(
            self,
            "check_name",
            _normalize_non_empty_text(self.check_name, "check_name"),
        )
        object.__setattr__(
            self,
            "token_index",
            _require_non_negative_int(self.token_index, "token_index"),
        )
        object.__setattr__(
            self,
            "token",
            _normalize_non_empty_text(self.token, "token"),
        )
        object.__setattr__(
            self,
            "message",
            _normalize_non_empty_text(self.message, "message"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the failure for report artifacts."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class StructuralCheckReport:
    """Aggregate decoded-sequence structural quality report."""

    sequence_count: int
    transition_count: int
    valid_transition_count: int
    valid_transition_rate: float
    boundary_order_pass_count: int
    boundary_order_pass_rate: float
    reference_pitch_min: int | None
    reference_pitch_max: int | None
    generated_pitch_min: int | None
    generated_pitch_max: int | None
    out_of_range_pitch_fraction: float
    duration_distribution_total_variation: float
    generated_unk_token_count: int
    generated_unk_rate: float
    failures: tuple[StructuralSequenceFailure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence_count",
            _require_non_negative_int(self.sequence_count, "sequence_count"),
        )
        object.__setattr__(
            self,
            "transition_count",
            _require_non_negative_int(self.transition_count, "transition_count"),
        )
        object.__setattr__(
            self,
            "valid_transition_count",
            _require_non_negative_int(
                self.valid_transition_count,
                "valid_transition_count",
            ),
        )
        if self.valid_transition_count > self.transition_count:
            raise ValueError("valid_transition_count must not exceed transition_count.")
        object.__setattr__(
            self,
            "valid_transition_rate",
            _normalize_fraction(self.valid_transition_rate, "valid_transition_rate"),
        )
        object.__setattr__(
            self,
            "boundary_order_pass_count",
            _require_non_negative_int(
                self.boundary_order_pass_count,
                "boundary_order_pass_count",
            ),
        )
        if self.boundary_order_pass_count > self.sequence_count:
            raise ValueError(
                "boundary_order_pass_count must not exceed sequence_count."
            )
        object.__setattr__(
            self,
            "boundary_order_pass_rate",
            _normalize_fraction(
                self.boundary_order_pass_rate,
                "boundary_order_pass_rate",
            ),
        )
        if self.reference_pitch_min is not None and self.reference_pitch_max is None:
            raise ValueError(
                "reference_pitch_max is required when reference_pitch_min is set."
            )
        if self.generated_pitch_min is not None and self.generated_pitch_max is None:
            raise ValueError(
                "generated_pitch_max is required when generated_pitch_min is set."
            )
        object.__setattr__(
            self,
            "out_of_range_pitch_fraction",
            _normalize_fraction(
                self.out_of_range_pitch_fraction,
                "out_of_range_pitch_fraction",
            ),
        )
        object.__setattr__(
            self,
            "duration_distribution_total_variation",
            _normalize_fraction(
                self.duration_distribution_total_variation,
                "duration_distribution_total_variation",
            ),
        )
        object.__setattr__(
            self,
            "generated_unk_token_count",
            _require_non_negative_int(
                self.generated_unk_token_count,
                "generated_unk_token_count",
            ),
        )
        object.__setattr__(
            self,
            "generated_unk_rate",
            _normalize_fraction(self.generated_unk_rate, "generated_unk_rate"),
        )
        object.__setattr__(
            self,
            "failures",
            tuple(
                failure
                if isinstance(failure, StructuralSequenceFailure)
                else StructuralSequenceFailure(
                    sequence_id=str(failure["sequence_id"]),
                    check_name=str(failure["check_name"]),
                    token_index=int(failure["token_index"]),
                    token=str(failure["token"]),
                    message=str(failure["message"]),
                )
                for failure in self.failures
            ),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the aggregate structural report for JSON artifacts."""
        return to_json_compatible(self)


def evaluate_structural_quality(
    generated_sequences: Sequence[DecodedTokenSequence | Mapping[str, Any]],
    *,
    reference_sequences: Sequence[DecodedTokenSequence | Mapping[str, Any]],
) -> StructuralCheckReport:
    """Evaluate generated token sequences against the baseline structural checks."""
    typed_generated = coerce_decoded_token_sequences(generated_sequences)
    typed_reference = coerce_decoded_token_sequences(reference_sequences)

    transition_count = 0
    valid_transition_count = 0
    boundary_order_pass_count = 0
    generated_unk_token_count = 0
    failures: list[StructuralSequenceFailure] = []
    generated_pitch_values: list[int] = []
    generated_duration_counts: Counter[int] = Counter()
    out_of_range_pitch_count = 0

    reference_pitch_values: list[int] = []
    reference_duration_counts: Counter[int] = Counter()
    for sequence in typed_reference:
        reference_pitch_values.extend(_extract_pitch_values(sequence.tokens))
        reference_duration_counts.update(_extract_duration_ticks(sequence.tokens))

    reference_pitch_min = (
        min(reference_pitch_values) if reference_pitch_values else None
    )
    reference_pitch_max = (
        max(reference_pitch_values) if reference_pitch_values else None
    )

    for sequence in typed_generated:
        sequence_failures, valid_boundaries = _validate_sequence_structure(sequence)
        failures.extend(sequence_failures)
        if valid_boundaries:
            boundary_order_pass_count += 1
        sequence_transition_count = max(len(sequence.tokens) - 1, 0)
        transition_count += sequence_transition_count
        valid_transition_count += sequence_transition_count - sum(
            1 for failure in sequence_failures if failure.check_name == "transition"
        )
        generated_unk_token_count += sum(
            1 for token in sequence.tokens if token == UNK_TOKEN
        )

        sequence_pitch_values = _extract_pitch_values(sequence.tokens)
        generated_pitch_values.extend(sequence_pitch_values)
        if reference_pitch_min is not None and reference_pitch_max is not None:
            out_of_range_pitch_count += sum(
                1
                for pitch_value in sequence_pitch_values
                if pitch_value < reference_pitch_min
                or pitch_value > reference_pitch_max
            )
        generated_duration_counts.update(_extract_duration_ticks(sequence.tokens))

    generated_pitch_min = (
        min(generated_pitch_values) if generated_pitch_values else None
    )
    generated_pitch_max = (
        max(generated_pitch_values) if generated_pitch_values else None
    )
    generated_pitch_token_count = len(generated_pitch_values)
    generated_token_count = sum(len(sequence.tokens) for sequence in typed_generated)

    return StructuralCheckReport(
        sequence_count=len(typed_generated),
        transition_count=transition_count,
        valid_transition_count=valid_transition_count,
        valid_transition_rate=(
            valid_transition_count / transition_count if transition_count > 0 else 1.0
        ),
        boundary_order_pass_count=boundary_order_pass_count,
        boundary_order_pass_rate=(
            boundary_order_pass_count / len(typed_generated) if typed_generated else 1.0
        ),
        reference_pitch_min=reference_pitch_min,
        reference_pitch_max=reference_pitch_max,
        generated_pitch_min=generated_pitch_min,
        generated_pitch_max=generated_pitch_max,
        out_of_range_pitch_fraction=(
            out_of_range_pitch_count / generated_pitch_token_count
            if generated_pitch_token_count > 0
            else 0.0
        ),
        duration_distribution_total_variation=_total_variation_distance(
            generated_duration_counts,
            reference_duration_counts,
        ),
        generated_unk_token_count=generated_unk_token_count,
        generated_unk_rate=(
            generated_unk_token_count / generated_token_count
            if generated_token_count > 0
            else 0.0
        ),
        failures=tuple(failures),
    )


def coerce_decoded_token_sequences(
    sequences: Sequence[DecodedTokenSequence | Mapping[str, Any]],
) -> tuple[DecodedTokenSequence, ...]:
    """Coerce loaded decoded-sequence payloads into the typed report contract."""
    return tuple(
        sequence
        if isinstance(sequence, DecodedTokenSequence)
        else DecodedTokenSequence(
            sequence_id=str(sequence["sequence_id"]),
            tokens=tuple(sequence["tokens"]),
        )
        for sequence in sequences
    )


def _validate_sequence_structure(
    sequence: DecodedTokenSequence,
) -> tuple[tuple[StructuralSequenceFailure, ...], bool]:
    failures: list[StructuralSequenceFailure] = []
    for index, (previous_token, current_token) in enumerate(
        zip(sequence.tokens, sequence.tokens[1:], strict=False),
        start=1,
    ):
        if _is_valid_token_transition(previous_token, current_token):
            continue
        failures.append(
            StructuralSequenceFailure(
                sequence_id=sequence.sequence_id,
                check_name="transition",
                token_index=index,
                token=current_token,
                message=(
                    "Token transition violates the baseline sequence grammar: "
                    f"{previous_token!r} -> {current_token!r}."
                ),
            )
        )

    boundary_failure = _structure_boundary_failure(sequence)
    if boundary_failure is not None:
        failures.append(boundary_failure)
        return tuple(failures), False
    return tuple(failures), True


def _structure_boundary_failure(
    sequence: DecodedTokenSequence,
) -> StructuralSequenceFailure | None:
    previous_marker_order: int | None = None
    for index, token in enumerate(sequence.tokens):
        if _token_family(token) == TokenFamily.TIME_SHIFT.value:
            previous_marker_order = None
            continue
        if _token_family(token) != TokenFamily.STRUCTURE.value:
            if token not in {BOS_TOKEN, UNK_TOKEN}:
                previous_marker_order = None
            continue

        marker_payload = _token_payload(token)
        marker_order = _STRUCTURE_ORDER.get(marker_payload)
        if marker_order is None:
            return StructuralSequenceFailure(
                sequence_id=sequence.sequence_id,
                check_name="boundary_order",
                token_index=index,
                token=token,
                message=(
                    "Structure token does not map to one supported boundary kind: "
                    f"{token!r}."
                ),
            )
        if previous_marker_order is not None and marker_order < previous_marker_order:
            return StructuralSequenceFailure(
                sequence_id=sequence.sequence_id,
                check_name="boundary_order",
                token_index=index,
                token=token,
                message=(
                    "Structure boundary ordering moved backward inside one structural "
                    f"run: {token!r}."
                ),
            )
        previous_marker_order = marker_order
    return None


def _is_valid_token_transition(previous_token: str, current_token: str) -> bool:
    if UNK_TOKEN in {previous_token, current_token}:
        return True
    previous_family = _token_family(previous_token)
    current_family = _token_family(current_token)
    allowed_next = {
        "START": {
            BOS_TOKEN,
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
        BOS_TOKEN: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
        EOS_TOKEN: set(),
        PAD_TOKEN: {PAD_TOKEN},
        TokenFamily.TIME_SHIFT.value: {
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
        TokenFamily.STRUCTURE.value: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
        TokenFamily.CONTROL_POINT.value: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
        TokenFamily.CONTROL_SPAN.value: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
        TokenFamily.NOTE_PITCH.value: {TokenFamily.NOTE_DURATION.value, UNK_TOKEN},
        TokenFamily.NOTE_DURATION.value: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            TokenFamily.NOTE_STRING.value,
            TokenFamily.NOTE_VELOCITY.value,
            UNK_TOKEN,
        },
        TokenFamily.NOTE_STRING.value: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            TokenFamily.NOTE_VELOCITY.value,
            UNK_TOKEN,
        },
        TokenFamily.NOTE_VELOCITY.value: {
            EOS_TOKEN,
            TokenFamily.TIME_SHIFT.value,
            TokenFamily.STRUCTURE.value,
            TokenFamily.CONTROL_POINT.value,
            TokenFamily.CONTROL_SPAN.value,
            TokenFamily.NOTE_PITCH.value,
            UNK_TOKEN,
        },
    }

    if previous_token in _SPECIAL_START_TOKENS:
        return (
            current_token in allowed_next[previous_token]
            or current_family in allowed_next[previous_token]
        )
    if previous_token == PAD_TOKEN:
        return current_token == PAD_TOKEN
    if previous_token == EOS_TOKEN:
        return False
    return current_token in allowed_next.get(
        previous_family, set()
    ) or current_family in allowed_next.get(previous_family, set())


def _extract_pitch_values(tokens: Sequence[str]) -> list[int]:
    values: list[int] = []
    for token in tokens:
        if _token_family(token) != TokenFamily.NOTE_PITCH.value:
            continue
        pitch_value = _parse_pitch_token(token)
        if pitch_value is not None:
            values.append(pitch_value)
    return values


def _extract_duration_ticks(tokens: Sequence[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for token in tokens:
        if _token_family(token) != TokenFamily.NOTE_DURATION.value:
            continue
        payload = _token_payload(token)
        if payload.isdigit():
            counts[int(payload)] += 1
    return counts


def _parse_pitch_token(token: str) -> int | None:
    payload = _token_payload(token)
    if payload == "UNPITCHED":
        return None
    step = payload[0]
    if step not in _PITCH_CLASS_BY_STEP:
        return None

    octave_index = len(payload)
    while octave_index > 0 and payload[octave_index - 1].isdigit():
        octave_index -= 1
    accidental = payload[1:octave_index]
    octave_text = payload[octave_index:]
    if not octave_text:
        return None
    accidental_offset = _ACCIDENTAL_OFFSETS.get(accidental.upper())
    if accidental_offset is None:
        return None
    octave = int(octave_text)
    return (octave + 1) * 12 + _PITCH_CLASS_BY_STEP[step] + accidental_offset


def _total_variation_distance(
    generated_counts: Counter[int],
    reference_counts: Counter[int],
) -> float:
    if not generated_counts and not reference_counts:
        return 0.0
    generated_total = sum(generated_counts.values())
    reference_total = sum(reference_counts.values())
    if generated_total <= 0 or reference_total <= 0:
        return 1.0

    all_keys = set(generated_counts) | set(reference_counts)
    return 0.5 * sum(
        abs(
            (generated_counts.get(key, 0) / generated_total)
            - (reference_counts.get(key, 0) / reference_total)
        )
        for key in all_keys
    )


def _token_family(token: str) -> str:
    if token in {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN}:
        return token
    return token.split(SEGMENT_SEPARATOR, maxsplit=1)[0]


def _token_payload(token: str) -> str:
    parts = token.split(SEGMENT_SEPARATOR, maxsplit=1)
    return parts[1] if len(parts) == _TOKEN_PART_COUNT_WITH_PAYLOAD else ""


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _normalize_fraction(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric.")
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{field_name} must satisfy 0.0 <= value <= 1.0.")
    return normalized


__all__ = [
    "DecodedTokenSequence",
    "StructuralCheckReport",
    "StructuralSequenceFailure",
    "coerce_decoded_token_sequences",
    "evaluate_structural_quality",
]
