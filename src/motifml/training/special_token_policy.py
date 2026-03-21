"""Frozen special-token policy helpers for MotifML training artifacts."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from motifml.training.token_families import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


class BoundaryPlacement(StrEnum):
    """Supported scopes for BOS/EOS boundary insertion."""

    DOCUMENT = "document"
    WINDOW = "window"


class PaddingInteraction(StrEnum):
    """Supported ways for padding to interact with BOS/EOS tokens."""

    OUTSIDE_BOUNDARIES = "outside_boundaries"
    INSIDE_BOUNDARIES = "inside_boundaries"


class UnknownTokenMapping(StrEnum):
    """Supported policies for handling tokens missing from the frozen vocabulary."""

    MAP_TO_UNK = "map_to_unk"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SpecialTokenPolicy:
    """Frozen BOS/EOS, padding, and unknown-token semantics for training."""

    policy_name: str = "baseline_special_tokens"
    policy_mode: str = "baseline_v1"
    bos_placement: BoundaryPlacement = BoundaryPlacement.DOCUMENT
    eos_placement: BoundaryPlacement = BoundaryPlacement.DOCUMENT
    padding_interaction: PaddingInteraction = PaddingInteraction.OUTSIDE_BOUNDARIES
    unknown_token_mapping: UnknownTokenMapping = UnknownTokenMapping.MAP_TO_UNK

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_name", _normalize_text(self.policy_name, "policy_name")
        )
        object.__setattr__(
            self, "policy_mode", _normalize_text(self.policy_mode, "policy_mode")
        )
        object.__setattr__(
            self,
            "bos_placement",
            BoundaryPlacement(self.bos_placement),
        )
        object.__setattr__(
            self,
            "eos_placement",
            BoundaryPlacement(self.eos_placement),
        )
        object.__setattr__(
            self,
            "padding_interaction",
            PaddingInteraction(self.padding_interaction),
        )
        object.__setattr__(
            self,
            "unknown_token_mapping",
            UnknownTokenMapping(self.unknown_token_mapping),
        )

    def to_version_payload(self) -> dict[str, str]:
        """Return a stable JSON-ready payload for persistence and versioning."""
        return {
            "policy_name": self.policy_name,
            "policy_mode": self.policy_mode,
            "bos": self.bos_placement.value,
            "eos": self.eos_placement.value,
            "padding_interaction": self.padding_interaction.value,
            "unknown_token_mapping": self.unknown_token_mapping.value,
        }

    def boundary_tokens_for_window(
        self,
        *,
        is_first_window: bool = True,
        is_last_window: bool = True,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the BOS/EOS tokens required for one emitted window."""
        prefix = (
            (BOS_TOKEN,)
            if self.bos_placement is BoundaryPlacement.WINDOW or is_first_window
            else ()
        )
        suffix = (
            (EOS_TOKEN,)
            if self.eos_placement is BoundaryPlacement.WINDOW or is_last_window
            else ()
        )
        return prefix, suffix

    def apply_to_tokens(
        self,
        tokens: Sequence[str],
        *,
        is_first_window: bool = True,
        is_last_window: bool = True,
    ) -> tuple[str, ...]:
        """Apply the frozen BOS/EOS policy to a token sequence."""
        normalized_tokens = tuple(_normalize_text(token, "tokens") for token in tokens)
        prefix, suffix = self.boundary_tokens_for_window(
            is_first_window=is_first_window,
            is_last_window=is_last_window,
        )
        return prefix + normalized_tokens + suffix

    def resolve_token_surface(
        self,
        token: str,
        *,
        known_tokens: Collection[str],
    ) -> str:
        """Resolve a token surface according to the frozen unknown-token policy."""
        normalized_token = _normalize_text(token, "token")
        normalized_known_tokens = {
            _normalize_text(known_token, "known_tokens") for known_token in known_tokens
        }
        if normalized_token in normalized_known_tokens:
            return normalized_token
        if self.unknown_token_mapping is UnknownTokenMapping.MAP_TO_UNK:
            if UNK_TOKEN not in normalized_known_tokens:
                raise ValueError(
                    "known_tokens must include <unk> when unknown_token_mapping "
                    "is map_to_unk."
                )
            return UNK_TOKEN
        raise KeyError(
            "Token surface is not present in the frozen vocabulary and the policy "
            "does not allow remapping to <unk>."
        )

    def validate_window_tokens(
        self,
        tokens: Sequence[str],
        *,
        padding_strategy: str = "none",
        is_first_window: bool = True,
        is_last_window: bool = True,
    ) -> tuple[str, ...]:
        """Validate one emitted token sequence against the persisted policy."""
        normalized_tokens = tuple(_normalize_text(token, "tokens") for token in tokens)
        prefix, suffix = self.boundary_tokens_for_window(
            is_first_window=is_first_window,
            is_last_window=is_last_window,
        )
        semantic_tokens = _strip_padding(
            normalized_tokens,
            padding_strategy=_normalize_padding_strategy(padding_strategy),
            expected_prefix=prefix,
            expected_suffix=suffix,
            padding_interaction=self.padding_interaction,
        )
        if prefix:
            if semantic_tokens[: len(prefix)] != prefix:
                raise ValueError(
                    "Window tokens do not match the persisted BOS placement policy."
                )
        elif BOS_TOKEN in semantic_tokens:
            raise ValueError(
                "Window tokens include a BOS token that is not allowed by the "
                "persisted policy."
            )

        if suffix:
            if semantic_tokens[-len(suffix) :] != suffix:
                raise ValueError(
                    "Window tokens do not match the persisted EOS placement policy."
                )
        elif EOS_TOKEN in semantic_tokens:
            raise ValueError(
                "Window tokens include an EOS token that is not allowed by the "
                "persisted policy."
            )

        if semantic_tokens.count(BOS_TOKEN) != len(prefix):
            raise ValueError("Window tokens contain an invalid BOS-token count.")
        if semantic_tokens.count(EOS_TOKEN) != len(suffix):
            raise ValueError("Window tokens contain an invalid EOS-token count.")

        return semantic_tokens


def coerce_special_token_policy(
    value: SpecialTokenPolicy | Mapping[str, Any],
) -> SpecialTokenPolicy:
    """Coerce a Kedro-loaded mapping into the typed special-token policy."""
    if isinstance(value, SpecialTokenPolicy):
        return value

    return SpecialTokenPolicy(
        policy_name=str(value.get("policy_name", "baseline_special_tokens")),
        policy_mode=str(value.get("policy_mode", "baseline_v1")),
        bos_placement=value.get("bos_placement", value.get("bos", "document")),
        eos_placement=value.get("eos_placement", value.get("eos", "document")),
        padding_interaction=value.get(
            "padding_interaction", PaddingInteraction.OUTSIDE_BOUNDARIES
        ),
        unknown_token_mapping=value.get(
            "unknown_token_mapping", UnknownTokenMapping.MAP_TO_UNK
        ),
    )


def _strip_padding(
    tokens: tuple[str, ...],
    *,
    padding_strategy: str,
    expected_prefix: tuple[str, ...],
    expected_suffix: tuple[str, ...],
    padding_interaction: PaddingInteraction,
) -> tuple[str, ...]:
    if padding_strategy == "none":
        if PAD_TOKEN in tokens:
            raise ValueError(
                "Window tokens include padding despite padding_strategy=none."
            )
        return tokens

    if padding_strategy == "left":
        return _strip_left_padding(
            tokens,
            expected_prefix=expected_prefix,
            padding_interaction=padding_interaction,
        )

    return _strip_right_padding(
        tokens,
        expected_suffix=expected_suffix,
        padding_interaction=padding_interaction,
    )


def _strip_left_padding(
    tokens: tuple[str, ...],
    *,
    expected_prefix: tuple[str, ...],
    padding_interaction: PaddingInteraction,
) -> tuple[str, ...]:
    if padding_interaction is PaddingInteraction.INSIDE_BOUNDARIES and expected_prefix:
        if tokens[: len(expected_prefix)] != expected_prefix:
            raise ValueError(
                "Inside-boundary left padding requires the BOS boundary token to stay "
                "at the window edge."
            )
        pad_end = len(expected_prefix)
        while pad_end < len(tokens) and tokens[pad_end] == PAD_TOKEN:
            pad_end += 1
        semantic_tokens = tokens[: len(expected_prefix)] + tokens[pad_end:]
        if PAD_TOKEN in semantic_tokens[len(expected_prefix) :]:
            raise ValueError(
                "Padding tokens must form one contiguous left-padding region."
            )
        return semantic_tokens

    pad_end = 0
    while pad_end < len(tokens) and tokens[pad_end] == PAD_TOKEN:
        pad_end += 1
    semantic_tokens = tokens[pad_end:]
    if PAD_TOKEN in semantic_tokens:
        raise ValueError("Padding tokens must form one contiguous left-padding region.")
    return semantic_tokens


def _strip_right_padding(
    tokens: tuple[str, ...],
    *,
    expected_suffix: tuple[str, ...],
    padding_interaction: PaddingInteraction,
) -> tuple[str, ...]:
    if padding_interaction is PaddingInteraction.INSIDE_BOUNDARIES and expected_suffix:
        if tokens[-len(expected_suffix) :] != expected_suffix:
            raise ValueError(
                "Inside-boundary right padding requires the EOS boundary token to stay "
                "at the window edge."
            )
        pad_start = len(tokens) - len(expected_suffix)
        while pad_start > 0 and tokens[pad_start - 1] == PAD_TOKEN:
            pad_start -= 1
        semantic_tokens = tokens[:pad_start] + tokens[-len(expected_suffix) :]
        if PAD_TOKEN in semantic_tokens[:pad_start]:
            raise ValueError(
                "Padding tokens must form one contiguous right-padding region."
            )
        return semantic_tokens

    pad_start = len(tokens)
    while pad_start > 0 and tokens[pad_start - 1] == PAD_TOKEN:
        pad_start -= 1
    semantic_tokens = tokens[:pad_start]
    if PAD_TOKEN in semantic_tokens:
        raise ValueError(
            "Padding tokens must form one contiguous right-padding region."
        )
    return semantic_tokens


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_padding_strategy(value: str) -> str:
    normalized = _normalize_text(value, "padding_strategy").casefold()
    if normalized not in {"none", "left", "right"}:
        raise ValueError("padding_strategy must be one of: none, left, right.")
    return normalized


__all__ = [
    "BoundaryPlacement",
    "PaddingInteraction",
    "SpecialTokenPolicy",
    "UnknownTokenMapping",
    "coerce_special_token_policy",
]
