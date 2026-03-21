"""Shared token encode/decode helpers for training, inspection, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from motifml.ir.projections.sequence import SequenceEvent
from motifml.training.sequence_schema import NotePayloadField
from motifml.training.special_token_policy import (
    SpecialTokenPolicy,
    coerce_special_token_policy,
)
from motifml.training.token_ordering import (
    expand_sequence_event_spans,
    flatten_token_spans,
)


@dataclass(frozen=True, slots=True)
class FrozenVocabulary:
    """Immutable view of a frozen token-to-id mapping."""

    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.token_to_id:
            raise ValueError("token_to_id must contain at least one token.")

        normalized_pairs = sorted(
            (
                (_normalize_text(token, "token_to_id"), _normalize_token_id(token_id))
                for token, token_id in self.token_to_id.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
        expected_ids = list(range(len(normalized_pairs)))
        actual_ids = [token_id for _, token_id in normalized_pairs]
        if actual_ids != expected_ids:
            raise ValueError(
                "Frozen vocabulary ids must be unique, contiguous, and zero-based."
            )

        object.__setattr__(
            self,
            "token_to_id",
            {token: token_id for token, token_id in normalized_pairs},
        )
        object.__setattr__(
            self,
            "id_to_token",
            tuple(token for token, _ in normalized_pairs),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the frozen vocabulary for inspection and reporting."""
        return {
            "token_to_id": dict(self.token_to_id),
            "id_to_token": list(self.id_to_token),
        }


def encode_projected_events_to_tokens(
    events: Sequence[SequenceEvent],
    *,
    time_resolution: int,
    note_payload_fields: Sequence[NotePayloadField | str] = (
        NotePayloadField.PITCH,
        NotePayloadField.DURATION,
    ),
    special_token_policy: SpecialTokenPolicy | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Encode projected sequence events into the frozen token-string contract."""
    policy = (
        SpecialTokenPolicy()
        if special_token_policy is None
        else coerce_special_token_policy(special_token_policy)
    )
    spans = expand_sequence_event_spans(
        events,
        time_resolution=time_resolution,
        note_payload_fields=note_payload_fields,
    )
    return policy.apply_to_tokens(
        flatten_token_spans(spans),
    )


def encode_token_strings_to_ids(
    tokens: Sequence[str],
    *,
    vocabulary: FrozenVocabulary | Mapping[str, Any],
    special_token_policy: SpecialTokenPolicy | Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Encode token strings into token ids using one frozen vocabulary."""
    typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
    policy = (
        SpecialTokenPolicy()
        if special_token_policy is None
        else coerce_special_token_policy(special_token_policy)
    )
    return tuple(
        typed_vocabulary.token_to_id[
            policy.resolve_token_surface(
                token,
                known_tokens=typed_vocabulary.token_to_id,
            )
        ]
        for token in tokens
    )


def decode_token_ids_to_strings(
    token_ids: Sequence[int],
    *,
    vocabulary: FrozenVocabulary | Mapping[str, Any],
) -> tuple[str, ...]:
    """Decode token ids back into token strings for inspection and reporting."""
    typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
    decoded_tokens: list[str] = []
    for index, token_id in enumerate(token_ids):
        normalized_token_id = _normalize_token_id(
            token_id, field_name=f"token_ids[{index}]"
        )
        if normalized_token_id >= len(typed_vocabulary.id_to_token):
            raise KeyError(
                "Token id is not present in the frozen vocabulary: "
                f"{normalized_token_id}."
            )
        decoded_tokens.append(typed_vocabulary.id_to_token[normalized_token_id])
    return tuple(decoded_tokens)


def coerce_frozen_vocabulary(
    value: FrozenVocabulary | Mapping[str, Any],
) -> FrozenVocabulary:
    """Coerce JSON-loaded vocabulary payloads into the typed frozen vocabulary."""
    if isinstance(value, FrozenVocabulary):
        return value

    if "token_to_id" in value:
        token_to_id = value["token_to_id"]
        if not isinstance(token_to_id, Mapping):
            raise ValueError("token_to_id must be a mapping when provided.")
        return FrozenVocabulary(
            token_to_id={
                str(token): int(token_id) for token, token_id in token_to_id.items()
            }
        )

    if "id_to_token" in value:
        id_to_token = value["id_to_token"]
        if not isinstance(id_to_token, Sequence) or isinstance(
            id_to_token, str | bytes
        ):
            raise ValueError("id_to_token must be a sequence when provided.")
        return FrozenVocabulary(
            token_to_id={str(token): index for index, token in enumerate(id_to_token)}
        )

    return FrozenVocabulary(
        token_to_id={str(token): int(token_id) for token, token_id in value.items()}
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_token_id(value: int, *, field_name: str = "token_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer token id.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


__all__ = [
    "FrozenVocabulary",
    "coerce_frozen_vocabulary",
    "decode_token_ids_to_strings",
    "encode_projected_events_to_tokens",
    "encode_token_strings_to_ids",
]
