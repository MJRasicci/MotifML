"""Unit tests for deterministic IR identifier helpers."""

from __future__ import annotations

import pytest

from motifml.ir.ids import (
    BAR_PREFIX,
    NOTE_PREFIX,
    ONSET_PREFIX,
    PART_PREFIX,
    PHRASE_PREFIX,
    POINT_CONTROL_PREFIX,
    SPAN_CONTROL_PREFIX,
    STAFF_PREFIX,
    VOICE_LANE_CHAIN_PREFIX,
    VOICE_LANE_PREFIX,
    bar_id,
    bar_sort_key,
    canonical_sort_ids,
    note_id,
    note_sort_key,
    onset_id,
    onset_sort_key,
    part_id,
    part_sort_key,
    phrase_id,
    phrase_sort_key,
    point_control_id,
    point_control_sort_key,
    sort_key_for_identifier,
    span_control_id,
    span_control_sort_key,
    staff_id,
    staff_sort_key,
    voice_lane_chain_id,
    voice_lane_id,
    voice_lane_sort_key,
)
from motifml.ir.models import Pitch
from motifml.ir.time import ScoreTime

EXPECTED_PREFIXES = (
    PART_PREFIX,
    STAFF_PREFIX,
    BAR_PREFIX,
    VOICE_LANE_PREFIX,
    VOICE_LANE_CHAIN_PREFIX,
    ONSET_PREFIX,
    NOTE_PREFIX,
    POINT_CONTROL_PREFIX,
    SPAN_CONTROL_PREFIX,
    PHRASE_PREFIX,
)


def test_id_builders_are_stable_and_use_expected_prefixes():
    part = part_id("track-7")
    staff = staff_id(part, 2)
    bar = bar_id(5)
    voice_lane = voice_lane_id(staff, 5, 1)
    chain = voice_lane_chain_id(part, staff, 1)
    onset = onset_id(voice_lane, 3)
    note = note_id(onset, 4)
    point_control = point_control_id("score", 0)
    span_control = span_control_id("part:track-7", 1)
    phrase = phrase_id("voice-chain:part:track-7:staff:track-7:2:1", 0)

    assert part == part_id("track-7")
    assert staff == staff_id(part, 2)
    assert bar == bar_id(5)
    assert voice_lane == voice_lane_id(staff, 5, 1)
    assert chain == voice_lane_chain_id(part, staff, 1)
    assert onset == onset_id(voice_lane, 3)
    assert note == note_id(onset, 4)
    assert point_control == point_control_id("score", 0)
    assert span_control == span_control_id("part:track-7", 1)
    assert phrase == phrase_id("voice-chain:part:track-7:staff:track-7:2:1", 0)

    built_ids = (
        part,
        staff,
        bar,
        voice_lane,
        chain,
        onset,
        note,
        point_control,
        span_control,
        phrase,
    )
    assert all(
        identifier.split(":", 1)[0] in EXPECTED_PREFIXES for identifier in built_ids
    )


def test_id_builders_produce_unique_values_within_a_document_scope():
    part = part_id("track-1")
    staff_a = staff_id(part, 0)
    staff_b = staff_id(part, 1)
    bar_a = bar_id(0)
    bar_b = bar_id(1)
    voice_lane_a = voice_lane_id(staff_a, 0, 0)
    voice_lane_b = voice_lane_id(staff_a, 1, 0)
    onset_a = onset_id(voice_lane_a, 0)
    note_a = note_id(onset_a, 0)
    point_control = point_control_id("score", 0)
    span_control = span_control_id("score", 0)
    phrase = phrase_id("voice-chain:track-1", 0)

    identifiers = (
        part,
        staff_a,
        staff_b,
        bar_a,
        bar_b,
        voice_lane_a,
        voice_lane_b,
        onset_a,
        note_a,
        point_control,
        span_control,
        phrase,
    )

    assert len(set(identifiers)) == len(identifiers)


def test_voice_lane_chain_id_is_deterministic_for_the_same_authorial_context():
    assert voice_lane_chain_id("part:1", "staff:1", 0) == voice_lane_chain_id(
        "part:1", "staff:1", 0
    )
    assert voice_lane_chain_id("part:1", "staff:1", 0) != voice_lane_chain_id(
        "part:1", "staff:1", 1
    )
    assert voice_lane_chain_id("part:1", "staff:1", 0) != voice_lane_chain_id(
        "part:2", "staff:1", 0
    )


def test_canonical_sort_helpers_are_stable_under_input_reordering():
    voice_lane_records = [
        {
            "bar_index": 1,
            "staff_id": "staff:b",
            "voice_index": 0,
            "voice_lane_id": "voice:staff:b:1:0",
        },
        {
            "bar_index": 0,
            "staff_id": "staff:a",
            "voice_index": 1,
            "voice_lane_id": "voice:staff:a:0:1",
        },
        {
            "bar_index": 0,
            "staff_id": "staff:a",
            "voice_index": 0,
            "voice_lane_id": "voice:staff:a:0:0",
        },
    ]

    expected_voice_lane_order = [
        "voice:staff:a:0:0",
        "voice:staff:a:0:1",
        "voice:staff:b:1:0",
    ]

    assert [
        record["voice_lane_id"]
        for record in sorted(
            voice_lane_records,
            key=lambda record: voice_lane_sort_key(
                record["bar_index"],
                record["staff_id"],
                record["voice_index"],
                record["voice_lane_id"],
            ),
        )
    ] == expected_voice_lane_order
    assert [
        record["voice_lane_id"]
        for record in sorted(
            reversed(voice_lane_records),
            key=lambda record: voice_lane_sort_key(
                record["bar_index"],
                record["staff_id"],
                record["voice_index"],
                record["voice_lane_id"],
            ),
        )
    ] == expected_voice_lane_order

    onset_records = [
        {
            "voice_lane_id": "voice:staff:a:0:0",
            "time": ScoreTime(1, 4),
            "attack_order_in_voice": 1,
            "onset_id": "onset:voice:staff:a:0:0:1",
        },
        {
            "voice_lane_id": "voice:staff:a:0:0",
            "time": ScoreTime(0, 1),
            "attack_order_in_voice": 0,
            "onset_id": "onset:voice:staff:a:0:0:0",
        },
    ]

    assert [
        record["onset_id"]
        for record in sorted(
            onset_records,
            key=lambda record: onset_sort_key(
                record["voice_lane_id"],
                record["time"],
                record["attack_order_in_voice"],
                record["onset_id"],
            ),
        )
    ] == [
        "onset:voice:staff:a:0:0:0",
        "onset:voice:staff:a:0:0:1",
    ]


def test_note_sort_helpers_respect_string_number_then_pitch_then_identifier():
    note_records = [
        {"string_number": None, "pitch": 72, "note_id": "note:c"},
        {"string_number": 2, "pitch": 60, "note_id": "note:b"},
        {"string_number": 1, "pitch": 64, "note_id": "note:a"},
        {"string_number": None, "pitch": None, "note_id": "note:d"},
    ]

    assert [
        record["note_id"]
        for record in sorted(
            note_records,
            key=lambda record: note_sort_key(
                record["string_number"], record["pitch"], record["note_id"]
            ),
        )
    ] == ["note:a", "note:b", "note:c", "note:d"]
    assert note_sort_key(None, "C4", "note:e") == (1, (1, "c4"), "note:e")
    assert note_sort_key(None, Pitch(step="C", octave=4), "note:f") == (
        1,
        (4, "C", ""),
        "note:f",
    )


def test_family_specific_sort_helpers_return_expected_canonical_keys():
    part_key = part_sort_key("Part B")
    staff_key = staff_sort_key("Part A", 2, "staff:Part A:2")
    bar_key = bar_sort_key(10, "bar:10")
    point_control_key = point_control_sort_key(
        "Score", "score", ScoreTime(1, 4), "ctrlp:score:0"
    )
    span_control_key = span_control_sort_key(
        "Voice", "voice:1", ScoreTime(1, 8), ScoreTime(1, 4), "ctrls:voice:0"
    )
    phrase_key = phrase_sort_key("Part", ScoreTime(0, 1), ScoreTime(1, 2), "phrase:0")

    assert part_key == ("part b", "Part B")
    assert staff_key == ("part a", 2, "staff:Part A:2")
    assert bar_key == (10, "bar:10")
    assert point_control_key == ("score", "score", ScoreTime(1, 4), "ctrlp:score:0")
    assert span_control_key == (
        "voice",
        "voice:1",
        ScoreTime(1, 8),
        ScoreTime(1, 4),
        "ctrls:voice:0",
    )
    assert phrase_key == ("part", ScoreTime(0, 1), ScoreTime(1, 2), "phrase:0")


def test_canonical_sort_ids_uses_identifier_prefix_then_suffix():
    identifiers = [
        "voice:staff:b:1:0",
        "part:track-b",
        "bar:10",
        "bar:2",
        "bar:1",
    ]

    assert canonical_sort_ids(identifiers) == (
        "bar:1",
        "bar:2",
        "bar:10",
        "part:track-b",
        "voice:staff:b:1:0",
    )
    assert sort_key_for_identifier("part:track-b") == ("part", ((1, "track-b"),))
    assert sort_key_for_identifier("part:") == ("part", ())


def test_id_builders_reject_negative_indexes():
    with pytest.raises(ValueError, match="staff_index"):
        staff_id(part_id("track-1"), -1)

    with pytest.raises(ValueError, match="bar_index"):
        bar_id(-1)
