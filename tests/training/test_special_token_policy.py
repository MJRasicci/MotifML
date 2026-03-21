"""Tests for the frozen special-token policy contract."""

from __future__ import annotations

import pytest

from motifml.training.special_token_policy import (
    BoundaryPlacement,
    PaddingInteraction,
    SpecialTokenPolicy,
    UnknownTokenMapping,
    coerce_special_token_policy,
)
from motifml.training.token_families import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


def test_special_token_policy_coerces_json_loaded_mappings() -> None:
    policy = coerce_special_token_policy(
        {
            "policy_name": "baseline_special_tokens",
            "policy_mode": "baseline_v1",
            "bos": "document",
            "eos": "window",
            "padding_interaction": "outside_boundaries",
            "unknown_token_mapping": "map_to_unk",
        }
    )

    assert policy == SpecialTokenPolicy(
        bos_placement=BoundaryPlacement.DOCUMENT,
        eos_placement=BoundaryPlacement.WINDOW,
        padding_interaction=PaddingInteraction.OUTSIDE_BOUNDARIES,
        unknown_token_mapping=UnknownTokenMapping.MAP_TO_UNK,
    )
    assert policy.to_version_payload() == {
        "policy_name": "baseline_special_tokens",
        "policy_mode": "baseline_v1",
        "bos": "document",
        "eos": "window",
        "padding_interaction": "outside_boundaries",
        "unknown_token_mapping": "map_to_unk",
    }


def test_apply_to_tokens_uses_document_scope_at_document_edges_only() -> None:
    policy = SpecialTokenPolicy()

    first_window = policy.apply_to_tokens(
        ("NOTE_PITCH:C4",),
        is_first_window=True,
        is_last_window=False,
    )
    middle_window = policy.apply_to_tokens(
        ("NOTE_PITCH:D4",),
        is_first_window=False,
        is_last_window=False,
    )
    final_window = policy.apply_to_tokens(
        ("NOTE_PITCH:E4",),
        is_first_window=False,
        is_last_window=True,
    )

    assert first_window == (BOS_TOKEN, "NOTE_PITCH:C4")
    assert middle_window == ("NOTE_PITCH:D4",)
    assert final_window == ("NOTE_PITCH:E4", EOS_TOKEN)


def test_validate_window_tokens_rejects_window_scope_override_for_document_policy() -> (
    None
):
    policy = SpecialTokenPolicy()

    with pytest.raises(ValueError, match="BOS token"):
        policy.validate_window_tokens(
            (BOS_TOKEN, "NOTE_PITCH:C4"),
            padding_strategy="none",
            is_first_window=False,
            is_last_window=False,
        )


def test_validate_window_tokens_enforces_outside_boundary_padding_contract() -> None:
    policy = SpecialTokenPolicy(
        bos_placement=BoundaryPlacement.WINDOW,
        eos_placement=BoundaryPlacement.WINDOW,
        padding_interaction=PaddingInteraction.OUTSIDE_BOUNDARIES,
    )

    semantic_tokens = policy.validate_window_tokens(
        (BOS_TOKEN, "NOTE_PITCH:C4", "NOTE_DURATION:96", EOS_TOKEN, PAD_TOKEN),
        padding_strategy="right",
    )

    assert semantic_tokens == (
        BOS_TOKEN,
        "NOTE_PITCH:C4",
        "NOTE_DURATION:96",
        EOS_TOKEN,
    )

    with pytest.raises(ValueError, match="right-padding region"):
        policy.validate_window_tokens(
            (BOS_TOKEN, "NOTE_PITCH:C4", "NOTE_DURATION:96", PAD_TOKEN, EOS_TOKEN),
            padding_strategy="right",
        )


def test_validate_window_tokens_supports_inside_boundary_padding_when_persisted() -> (
    None
):
    policy = SpecialTokenPolicy(
        bos_placement=BoundaryPlacement.WINDOW,
        eos_placement=BoundaryPlacement.WINDOW,
        padding_interaction=PaddingInteraction.INSIDE_BOUNDARIES,
    )

    semantic_tokens = policy.validate_window_tokens(
        (BOS_TOKEN, PAD_TOKEN, "NOTE_PITCH:C4", "NOTE_DURATION:96", EOS_TOKEN),
        padding_strategy="left",
    )

    assert semantic_tokens == (
        BOS_TOKEN,
        "NOTE_PITCH:C4",
        "NOTE_DURATION:96",
        EOS_TOKEN,
    )


def test_resolve_token_surface_maps_unknown_tokens_to_unk_when_configured() -> None:
    policy = SpecialTokenPolicy()

    assert (
        policy.resolve_token_surface(
            "NOTE_PITCH:UNKNOWN",
            known_tokens={BOS_TOKEN, EOS_TOKEN, UNK_TOKEN},
        )
        == UNK_TOKEN
    )


def test_resolve_token_surface_can_forbid_unknown_token_mapping() -> None:
    policy = SpecialTokenPolicy(unknown_token_mapping=UnknownTokenMapping.ERROR)

    with pytest.raises(KeyError, match="does not allow remapping"):
        policy.resolve_token_surface(
            "NOTE_PITCH:UNKNOWN",
            known_tokens={BOS_TOKEN, EOS_TOKEN, UNK_TOKEN},
        )
