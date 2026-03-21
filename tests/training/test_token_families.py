"""Tests for canonical token family builders."""

from __future__ import annotations

import pytest

from motifml.training.token_families import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    build_control_point_token,
    build_control_span_token,
    build_note_duration_token,
    build_note_pitch_token,
    build_note_string_token,
    build_note_velocity_token,
    build_structure_token,
    build_time_shift_token,
)


def test_special_tokens_match_the_frozen_v1_contract() -> None:
    assert PAD_TOKEN == "<pad>"
    assert BOS_TOKEN == "<bos>"
    assert EOS_TOKEN == "<eos>"
    assert UNK_TOKEN == "<unk>"
    assert SPECIAL_TOKENS == {
        "pad": "<pad>",
        "bos": "<bos>",
        "eos": "<eos>",
        "unk": "<unk>",
    }


def test_time_and_note_token_builders_emit_canonical_strings() -> None:
    assert build_time_shift_token(96) == "TIME_SHIFT:96"
    assert build_note_pitch_token("c#4") == "NOTE_PITCH:C#4"
    assert build_note_duration_token(48) == "NOTE_DURATION:48"
    assert build_note_string_token(2) == "NOTE_STRING:2"
    assert build_note_velocity_token("mezzo-forte") == "NOTE_VELOCITY:MEZZO_FORTE"


def test_structure_and_control_token_builders_normalize_payload_segments() -> None:
    assert build_structure_token("onset group") == "STRUCTURE:ONSET_GROUP"
    assert (
        build_control_point_token("score", "tempo change", "q=120")
        == "CONTROL_POINT:SCORE:TEMPO_CHANGE:Q=120"
    )
    assert (
        build_control_span_token("voice", "hairpin", "crescendo", 96)
        == "CONTROL_SPAN:VOICE:HAIRPIN:CRESCENDO:96"
    )


def test_token_builders_reject_empty_or_colon_delimited_payload_segments() -> None:
    with pytest.raises(ValueError, match="kind"):
        build_structure_token("   ")

    with pytest.raises(ValueError, match="colon-delimited"):
        build_control_point_token("score", "tempo", "bad:value")
