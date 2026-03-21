"""Tests for shared event/token/id codec helpers."""

from __future__ import annotations

import pytest

from motifml.ir.models import (
    ControlScope,
    DynamicChangeValue,
    NoteEvent,
    Pitch,
    PointControlEvent,
    PointControlKind,
)
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    StructureMarkerKind,
    StructureMarkerSequenceEvent,
)
from motifml.ir.time import ScoreTime
from motifml.training.special_token_policy import SpecialTokenPolicy
from motifml.training.token_codec import (
    FrozenVocabulary,
    coerce_frozen_vocabulary,
    decode_token_ids_to_strings,
    encode_projected_events_to_tokens,
    encode_token_strings_to_ids,
)
from motifml.training.token_families import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def test_encode_projected_events_to_tokens_is_deterministic() -> None:
    events = (
        _build_structure_event(time=ScoreTime(0, 1)),
        _build_point_control_event(time=ScoreTime(1, 4)),
        _build_note_event(time=ScoreTime(1, 4)),
    )
    policy = SpecialTokenPolicy().to_version_payload()

    first = encode_projected_events_to_tokens(
        events,
        time_resolution=96,
        special_token_policy=policy,
    )
    repeated = encode_projected_events_to_tokens(
        events,
        time_resolution=96,
        special_token_policy=policy,
    )

    assert first == repeated
    assert first == (
        BOS_TOKEN,
        "STRUCTURE:BAR",
        "TIME_SHIFT:96",
        "CONTROL_POINT:SCORE:DYNAMIC_CHANGE:MF",
        "NOTE_PITCH:C4",
        "NOTE_DURATION:96",
        EOS_TOKEN,
    )


def test_encode_token_strings_to_ids_uses_unk_mapping_from_policy() -> None:
    vocabulary = FrozenVocabulary(
        token_to_id={
            BOS_TOKEN: 0,
            EOS_TOKEN: 1,
            UNK_TOKEN: 2,
            "NOTE_PITCH:C4": 3,
        }
    )

    token_ids = encode_token_strings_to_ids(
        (BOS_TOKEN, "NOTE_PITCH:UNKNOWN", EOS_TOKEN),
        vocabulary=vocabulary,
    )

    assert token_ids == (0, 2, 1)


def test_decode_token_ids_to_strings_is_readable_and_stable() -> None:
    vocabulary = coerce_frozen_vocabulary(
        {
            "token_to_id": {
                BOS_TOKEN: 0,
                "STRUCTURE:BAR": 1,
                "NOTE_PITCH:C4": 2,
                EOS_TOKEN: 3,
            }
        }
    )

    decoded = decode_token_ids_to_strings((0, 1, 2, 3), vocabulary=vocabulary)

    assert decoded == (BOS_TOKEN, "STRUCTURE:BAR", "NOTE_PITCH:C4", EOS_TOKEN)
    assert decode_token_ids_to_strings((0, 1, 2, 3), vocabulary=vocabulary) == decoded


def test_decode_token_ids_to_strings_rejects_unknown_or_malformed_ids() -> None:
    vocabulary = FrozenVocabulary(
        token_to_id={BOS_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2}
    )

    with pytest.raises(KeyError, match="not present"):
        decode_token_ids_to_strings((99,), vocabulary=vocabulary)

    with pytest.raises(ValueError, match="integer token id"):
        decode_token_ids_to_strings(("not-an-int",), vocabulary=vocabulary)


def _build_structure_event(time: ScoreTime) -> StructureMarkerSequenceEvent:
    return StructureMarkerSequenceEvent(
        time=time,
        marker_kind=StructureMarkerKind.BAR,
        entity_id="bar:1",
    )


def _build_note_event(time: ScoreTime) -> NoteSequenceEvent:
    return NoteSequenceEvent(
        time=time,
        note=NoteEvent(
            note_id="note:onset:1:1",
            onset_id="onset:1",
            part_id="part:1",
            staff_id="staff:part:1:1",
            time=time,
            attack_duration=ScoreTime(1, 4),
            sounding_duration=ScoreTime(1, 4),
            pitch=Pitch(step="C", octave=4),
        ),
        part_id="part:1",
        staff_id="staff:part:1:1",
        bar_id="bar:1",
        voice_lane_id="voice:1",
        onset_id="onset:1",
    )


def _build_point_control_event(time: ScoreTime) -> PointControlSequenceEvent:
    return PointControlSequenceEvent(
        time=time,
        control=PointControlEvent(
            control_id="ctrlp:score:1",
            kind=PointControlKind.DYNAMIC_CHANGE,
            scope=ControlScope.SCORE,
            target_ref="score",
            time=time,
            value=DynamicChangeValue(marking="mf"),
        ),
    )
