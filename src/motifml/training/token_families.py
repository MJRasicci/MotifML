"""Canonical token family names and string builders for training."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

SEGMENT_SEPARATOR = ":"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = {
    "pad": PAD_TOKEN,
    "bos": BOS_TOKEN,
    "eos": EOS_TOKEN,
    "unk": UNK_TOKEN,
}
_UNDERSCORE_PATTERN = re.compile(r"_+")


class TokenFamily(StrEnum):
    """Supported v1 token families for the baseline sequence model."""

    TIME_SHIFT = "TIME_SHIFT"
    STRUCTURE = "STRUCTURE"
    NOTE_PITCH = "NOTE_PITCH"
    NOTE_DURATION = "NOTE_DURATION"
    NOTE_STRING = "NOTE_STRING"
    NOTE_VELOCITY = "NOTE_VELOCITY"
    CONTROL_POINT = "CONTROL_POINT"
    CONTROL_SPAN = "CONTROL_SPAN"


def build_time_shift_token(ticks: int) -> str:
    """Build the canonical time-shift token."""
    return _build_family_token(
        TokenFamily.TIME_SHIFT, _normalize_positive_int(ticks, "ticks")
    )


def build_structure_token(kind: str) -> str:
    """Build one structure-marker token."""
    return _build_family_token(
        TokenFamily.STRUCTURE, _normalize_symbolic_segment(kind, "kind")
    )


def build_note_pitch_token(encoded_pitch: str) -> str:
    """Build one note-pitch token from a pre-encoded pitch payload."""
    return _build_family_token(
        TokenFamily.NOTE_PITCH,
        _normalize_symbolic_segment(encoded_pitch, "encoded_pitch"),
    )


def build_note_duration_token(ticks: int) -> str:
    """Build one note-duration token."""
    return _build_family_token(
        TokenFamily.NOTE_DURATION,
        _normalize_positive_int(ticks, "ticks"),
    )


def build_note_string_token(string_number: int) -> str:
    """Build one optional string-number token."""
    return _build_family_token(
        TokenFamily.NOTE_STRING,
        _normalize_positive_int(string_number, "string_number"),
    )


def build_note_velocity_token(value: str | int) -> str:
    """Build one optional note-velocity token."""
    return _build_family_token(
        TokenFamily.NOTE_VELOCITY,
        _normalize_payload_segment(value, "value"),
    )


def build_control_point_token(
    scope: str,
    kind: str,
    value: str | int | float,
) -> str:
    """Build one point-control token."""
    return _build_family_token(
        TokenFamily.CONTROL_POINT,
        _normalize_symbolic_segment(scope, "scope"),
        _normalize_symbolic_segment(kind, "kind"),
        _normalize_payload_segment(value, "value"),
    )


def build_control_span_token(
    scope: str,
    kind: str,
    value: str | int | float,
    duration_ticks: int,
) -> str:
    """Build one span-control token."""
    return _build_family_token(
        TokenFamily.CONTROL_SPAN,
        _normalize_symbolic_segment(scope, "scope"),
        _normalize_symbolic_segment(kind, "kind"),
        _normalize_payload_segment(value, "value"),
        _normalize_positive_int(duration_ticks, "duration_ticks"),
    )


def _build_family_token(family: TokenFamily, *segments: str) -> str:
    return SEGMENT_SEPARATOR.join((family.value, *segments))


def _normalize_positive_int(value: int, field_name: str) -> str:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return str(normalized)


def _normalize_symbolic_segment(value: str, field_name: str) -> str:
    return _normalize_payload_segment(value, field_name)


def _normalize_payload_segment(value: Any, field_name: str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive.")
        return format(value, "g")

    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if SEGMENT_SEPARATOR in normalized:
        raise ValueError(
            f"{field_name} must not contain '{SEGMENT_SEPARATOR}' because token "
            "payloads are colon-delimited."
        )

    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = _UNDERSCORE_PATTERN.sub("_", normalized)
    return normalized.upper()


__all__ = [
    "BOS_TOKEN",
    "EOS_TOKEN",
    "PAD_TOKEN",
    "SEGMENT_SEPARATOR",
    "SPECIAL_TOKENS",
    "TokenFamily",
    "UNK_TOKEN",
    "build_control_point_token",
    "build_control_span_token",
    "build_note_duration_token",
    "build_note_pitch_token",
    "build_note_string_token",
    "build_note_velocity_token",
    "build_structure_token",
    "build_time_shift_token",
]
