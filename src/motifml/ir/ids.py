"""Deterministic IR identifier and sort-key helpers."""

from __future__ import annotations

from typing import Any, Protocol

from motifml.ir.time import ScoreTime

PART_PREFIX = "part"
STAFF_PREFIX = "staff"
BAR_PREFIX = "bar"
VOICE_LANE_PREFIX = "voice"
VOICE_LANE_CHAIN_PREFIX = "voice-chain"
ONSET_PREFIX = "onset"
NOTE_PREFIX = "note"
POINT_CONTROL_PREFIX = "ctrlp"
SPAN_CONTROL_PREFIX = "ctrls"
PHRASE_PREFIX = "phrase"


class SupportsSortKey(Protocol):
    """Protocol for value objects that expose a stable sort key."""

    def sort_key(self) -> tuple[Any, ...]:
        """Return a stable sort key representation."""


__all__ = [
    "BAR_PREFIX",
    "NOTE_PREFIX",
    "ONSET_PREFIX",
    "PART_PREFIX",
    "PHRASE_PREFIX",
    "POINT_CONTROL_PREFIX",
    "SPAN_CONTROL_PREFIX",
    "STAFF_PREFIX",
    "VOICE_LANE_CHAIN_PREFIX",
    "VOICE_LANE_PREFIX",
    "bar_id",
    "bar_sort_key",
    "canonical_sort_ids",
    "edge_sort_key",
    "note_id",
    "note_sort_key",
    "onset_id",
    "onset_sort_key",
    "part_id",
    "part_sort_key",
    "phrase_id",
    "phrase_sort_key",
    "point_control_id",
    "point_control_sort_key",
    "sort_key_for_identifier",
    "span_control_id",
    "span_control_sort_key",
    "staff_id",
    "staff_sort_key",
    "voice_lane_chain_id",
    "voice_lane_id",
    "voice_lane_sort_key",
]


def part_id(track_identity: int | str) -> str:
    """Build a stable document-local part identifier."""
    return _prefixed_identifier(PART_PREFIX, track_identity)


def staff_id(part_identifier: str, staff_index: int) -> str:
    """Build a stable staff identifier scoped to one part."""
    _require_non_negative_index("staff_index", staff_index)
    return _prefixed_identifier(STAFF_PREFIX, part_identifier, staff_index)


def bar_id(bar_index: int) -> str:
    """Build a stable bar identifier."""
    _require_non_negative_index("bar_index", bar_index)
    return _prefixed_identifier(BAR_PREFIX, bar_index)


def voice_lane_id(staff_identifier: str, bar_index: int, voice_index: int) -> str:
    """Build a stable bar-scoped voice lane identifier."""
    _require_non_negative_index("bar_index", bar_index)
    _require_non_negative_index("voice_index", voice_index)
    return _prefixed_identifier(
        VOICE_LANE_PREFIX, staff_identifier, bar_index, voice_index
    )


def voice_lane_chain_id(
    part_identifier: str, staff_identifier: str, voice_index: int
) -> str:
    """Build a deterministic continuity identifier for a voice lane across bars."""
    _require_non_negative_index("voice_index", voice_index)
    return _prefixed_identifier(
        VOICE_LANE_CHAIN_PREFIX, part_identifier, staff_identifier, voice_index
    )


def onset_id(voice_lane_identifier: str, attack_index: int) -> str:
    """Build a stable onset-group identifier."""
    _require_non_negative_index("attack_index", attack_index)
    return _prefixed_identifier(ONSET_PREFIX, voice_lane_identifier, attack_index)


def note_id(onset_identifier: str, note_index: int) -> str:
    """Build a stable note identifier inside an onset group."""
    _require_non_negative_index("note_index", note_index)
    return _prefixed_identifier(NOTE_PREFIX, onset_identifier, note_index)


def point_control_id(scope: str, ordinal: int) -> str:
    """Build a stable point-control identifier."""
    _require_non_negative_index("ordinal", ordinal)
    return _prefixed_identifier(POINT_CONTROL_PREFIX, scope, ordinal)


def span_control_id(scope: str, ordinal: int) -> str:
    """Build a stable span-control identifier."""
    _require_non_negative_index("ordinal", ordinal)
    return _prefixed_identifier(SPAN_CONTROL_PREFIX, scope, ordinal)


def phrase_id(scope_ref: str, ordinal: int) -> str:
    """Build a stable phrase identifier."""
    _require_non_negative_index("ordinal", ordinal)
    return _prefixed_identifier(PHRASE_PREFIX, scope_ref, ordinal)


def part_sort_key(part_identifier: str) -> tuple[str, str]:
    """Return the canonical sort key for a part identifier."""
    return (part_identifier.casefold(), part_identifier)


def staff_sort_key(
    part_identifier: str, staff_index: int, staff_identifier: str
) -> tuple[str, int, str]:
    """Return the canonical sort key for a staff."""
    _require_non_negative_index("staff_index", staff_index)
    return (part_identifier.casefold(), staff_index, staff_identifier)


def bar_sort_key(bar_index: int, bar_identifier: str) -> tuple[int, str]:
    """Return the canonical sort key for a bar."""
    _require_non_negative_index("bar_index", bar_index)
    return (bar_index, bar_identifier)


def voice_lane_sort_key(
    bar_index: int,
    staff_identifier: str,
    voice_index: int,
    voice_lane_identifier: str,
) -> tuple[int, str, int, str]:
    """Return the canonical sort key for a voice lane."""
    _require_non_negative_index("bar_index", bar_index)
    _require_non_negative_index("voice_index", voice_index)
    return (bar_index, staff_identifier, voice_index, voice_lane_identifier)


def onset_sort_key(
    voice_lane_identifier: str,
    time: ScoreTime,
    attack_order_in_voice: int,
    onset_identifier: str,
) -> tuple[str, ScoreTime, int, str]:
    """Return the canonical sort key for an onset group."""
    _require_non_negative_index("attack_order_in_voice", attack_order_in_voice)
    return (voice_lane_identifier, time, attack_order_in_voice, onset_identifier)


def note_sort_key(
    string_number: int | None,
    pitch: SupportsSortKey | int | str | None,
    note_identifier: str,
) -> tuple[int, Any, str]:
    """Return the canonical sort key for notes within one onset."""
    if string_number is not None:
        _require_non_negative_index("string_number", string_number)
        return (0, string_number, note_identifier)

    if pitch is not None:
        if hasattr(pitch, "sort_key"):
            return (1, pitch.sort_key(), note_identifier)

        return (1, _normalized_value_sort_key(pitch), note_identifier)

    return (2, 0, note_identifier)


def point_control_sort_key(
    scope: str,
    target_ref: str,
    time: ScoreTime,
    control_identifier: str,
) -> tuple[str, str, ScoreTime, str]:
    """Return the canonical sort key for a point control."""
    return (scope.casefold(), target_ref, time, control_identifier)


def span_control_sort_key(
    scope: str,
    target_ref: str,
    start_time: ScoreTime,
    end_time: ScoreTime,
    control_identifier: str,
) -> tuple[str, str, ScoreTime, ScoreTime, str]:
    """Return the canonical sort key for a span control."""
    return (scope.casefold(), target_ref, start_time, end_time, control_identifier)


def phrase_sort_key(
    scope_ref: str,
    start_time: ScoreTime,
    end_time: ScoreTime,
    phrase_identifier: str,
) -> tuple[str, ScoreTime, ScoreTime, str]:
    """Return the canonical sort key for a phrase span."""
    return (scope_ref.casefold(), start_time, end_time, phrase_identifier)


def canonical_sort_ids(identifiers: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return identifiers in canonical order.

    The helper is intentionally shallow and deterministic so later serializers can
    sort already-emitted identifiers without re-deriving entity semantics.
    """
    return tuple(sorted(identifiers, key=sort_key_for_identifier))


def sort_key_for_identifier(
    identifier: str,
) -> tuple[str, tuple[tuple[int, int | str], ...]]:
    """Return a stable fallback sort key for any IR identifier."""
    prefix, _, suffix = identifier.partition(":")
    return (prefix.casefold(), _identifier_suffix_sort_key(suffix))


def edge_sort_key(
    source_id: str,
    edge_type: str,
    target_id: str,
) -> tuple[
    tuple[str, tuple[tuple[int, int | str], ...]],
    str,
    tuple[str, tuple[tuple[int, int | str], ...]],
]:
    """Return the canonical sort key for an intrinsic edge."""
    return (
        sort_key_for_identifier(source_id),
        edge_type.casefold(),
        sort_key_for_identifier(target_id),
    )


def _prefixed_identifier(prefix: str, *components: object) -> str:
    normalized_components = tuple(
        _normalize_identifier_component(component) for component in components
    )
    return ":".join((prefix, *normalized_components))


def _normalize_identifier_component(component: object) -> str:
    if isinstance(component, str):
        return component
    return str(component)


def _normalized_value_sort_key(value: int | str) -> tuple[int, int | str]:
    if isinstance(value, int):
        return (0, value)

    return (1, value.casefold())


def _identifier_suffix_sort_key(suffix: str) -> tuple[tuple[int, int | str], ...]:
    if not suffix:
        return ()

    return tuple(
        _normalized_suffix_component_sort_key(component)
        for component in suffix.split(":")
    )


def _normalized_suffix_component_sort_key(component: str) -> tuple[int, int | str]:
    if component.isdigit():
        return (0, int(component))

    return (1, component.casefold())


def _require_non_negative_index(field_name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
