"""Tests for IR build validation nodes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.models import RhythmBaseValue, TimeSignature
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.models import DiagnosticSeverity
from motifml.pipelines.ir_build.nodes import (
    build_written_time_map,
    emit_bars,
    emit_onset_groups,
    emit_parts_and_staves,
    emit_point_control_events,
    emit_span_control_events,
    emit_voice_lanes,
    validate_canonical_score_surface,
)

EXPECTED_MISSING_FIELD_ERRORS = 6
EXPECTED_INVALID_SCORE_TIME_ERRORS = 2
EXPECTED_TIME_MAP_CONTIGUITY_ERRORS = 1
EXPECTED_TRANSPOSED_CHROMATIC = 2
EXPECTED_CAPO_FRET = 2
EXPECTED_KEY_ACCIDENTALS = -2
EXPECTED_POINT_CONTROLS_IN_FIXTURE = 4
EXPECTED_FIRST_FIXTURE_TEMPO = 120.0
EXPECTED_SECOND_FIXTURE_TEMPO = 132.0
EXPECTED_FERMATA_LENGTH_SCALE = 1.5
EXPECTED_SPAN_CONTROLS_IN_FIXTURE = 2
EXPECTED_VOICE_LANES_WITH_REENTRY = 5
EXPECTED_PRIMARY_TUPLET_NUMERATOR = 3
EXPECTED_PRIMARY_TUPLET_DENOMINATOR = 2
FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "motif_json"


def test_validate_canonical_score_surface_accepts_a_canonical_minimal_document():
    result = validate_canonical_score_surface(
        [
            MotifJsonDocument(
                relative_path="fixtures/canonical.json",
                sha256="abc123",
                file_size_bytes=0,
                score=_minimal_canonical_score(),
            )
        ]
    )[0]

    assert result.passed is True
    assert result.error_count == 0
    assert result.warning_count == 0
    assert result.diagnostics == ()


def test_validate_canonical_score_surface_sorts_results_by_relative_path():
    documents = [
        MotifJsonDocument(
            relative_path="zeta/document.json",
            sha256="zeta",
            file_size_bytes=0,
            score=_minimal_canonical_score(),
        ),
        MotifJsonDocument(
            relative_path="alpha/document.json",
            sha256="alpha",
            file_size_bytes=0,
            score=_minimal_canonical_score(),
        ),
    ]

    results = validate_canonical_score_surface(documents)

    assert [result.relative_path for result in results] == [
        "alpha/document.json",
        "zeta/document.json",
    ]


def test_validate_canonical_score_surface_warns_when_legacy_relation_hints_lack_relations():
    score = _minimal_canonical_score()
    note = score["tracks"][0]["staves"][0]["measures"][0]["voices"][0]["beats"][0][
        "notes"
    ][0]
    note["articulation"] = {"tieDestination": True}

    result = validate_canonical_score_surface(
        [
            MotifJsonDocument(
                relative_path="fixtures/legacy-relations.json",
                sha256="legacy",
                file_size_bytes=0,
                score=score,
            )
        ]
    )[0]

    assert result.passed is True
    assert result.error_count == 0
    assert result.warning_count == 1
    assert result.diagnostics[0].severity is DiagnosticSeverity.WARNING
    assert result.diagnostics[0].path.endswith(".articulation.relations")


def test_validate_canonical_score_surface_rejects_missing_required_canonical_fields():
    score = _minimal_canonical_score()

    del score["pointControls"]
    del score["spanControls"]
    del score["timelineBars"][0]["start"]
    del score["timelineBars"][0]["duration"]
    del score["tracks"][0]["staves"][0]["measures"][0]["voices"][0]["beats"][0][
        "offset"
    ]
    del score["tracks"][0]["staves"][0]["measures"][0]["voices"][0]["beats"][0][
        "duration"
    ]

    result = validate_canonical_score_surface(
        [
            MotifJsonDocument(
                relative_path="fixtures/missing-fields.json",
                sha256="missing",
                file_size_bytes=0,
                score=score,
            )
        ]
    )[0]

    assert result.passed is False
    assert result.error_count == EXPECTED_MISSING_FIELD_ERRORS
    assert [diagnostic.path for diagnostic in result.diagnostics] == [
        "pointControls",
        "spanControls",
        "timelineBars[0].duration",
        "timelineBars[0].start",
        "tracks[0].staves[0].measures[0].voices[0].beats[0].duration",
        "tracks[0].staves[0].measures[0].voices[0].beats[0].offset",
    ]


def test_validate_canonical_score_surface_rejects_invalid_score_time_payloads():
    score = _minimal_canonical_score()
    score["timelineBars"][0]["start"] = {"numerator": 0, "denominator": 0}
    score["tracks"][0]["staves"][0]["measures"][0]["voices"][0]["beats"][0][
        "offset"
    ] = {"numerator": "0", "denominator": 1}

    result = validate_canonical_score_surface(
        [
            MotifJsonDocument(
                relative_path="fixtures/invalid-score-time.json",
                sha256="broken",
                file_size_bytes=0,
                score=score,
            )
        ]
    )[0]

    assert result.passed is False
    assert result.error_count == EXPECTED_INVALID_SCORE_TIME_ERRORS
    assert [diagnostic.path for diagnostic in result.diagnostics] == [
        "timelineBars[0].start",
        "tracks[0].staves[0].measures[0].voices[0].beats[0].offset",
    ]


def test_build_written_time_map_builds_regular_contiguous_bar_geometry():
    score = _minimal_canonical_score()
    score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "4/4",
            "start": {"numerator": 1, "denominator": 1},
            "duration": {"numerator": 1, "denominator": 1},
        }
    )

    result = _build_written_time_map_result(score=score, relative_path="regular.json")

    assert result.passed is True
    assert result.warning_count == 0
    assert result.bar_times == {
        0: (ScoreTime(0, 1), ScoreTime(1, 1)),
        1: (ScoreTime(1, 1), ScoreTime(1, 1)),
    }


def test_build_written_time_map_marks_pickup_bars_when_anacrusis_is_present():
    score = _minimal_canonical_score()
    score["timelineBars"][0]["duration"] = {"numerator": 1, "denominator": 4}
    score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "4/4",
            "start": {"numerator": 1, "denominator": 4},
            "duration": {"numerator": 1, "denominator": 1},
        }
    )
    score["anacrusis"] = True

    result = _build_written_time_map_result(score=score, relative_path="pickup.json")

    assert result.passed is True
    assert result.warning_count == 0
    assert result.bars[0].is_anacrusis is True
    assert result.bars[0].duration == ScoreTime(1, 4)


def test_build_written_time_map_supports_varying_time_signatures():
    score = _minimal_canonical_score()
    score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "6/8",
            "start": {"numerator": 1, "denominator": 1},
            "duration": {"numerator": 3, "denominator": 4},
        }
    )

    result = _build_written_time_map_result(
        score=score,
        relative_path="varying-meter.json",
    )

    assert result.passed is True
    assert result.bar_times[1] == (ScoreTime(1, 1), ScoreTime(3, 4))


def test_build_written_time_map_rejects_non_contiguous_or_overlapping_bars():
    gap_score = _minimal_canonical_score()
    gap_score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "4/4",
            "start": {"numerator": 5, "denominator": 4},
            "duration": {"numerator": 1, "denominator": 1},
        }
    )
    gap_result = _build_written_time_map_result(
        score=gap_score,
        relative_path="gap.json",
    )

    assert gap_result.passed is False
    assert gap_result.error_count == EXPECTED_TIME_MAP_CONTIGUITY_ERRORS
    assert gap_result.diagnostics[0].path == "timelineBars[1].start"

    overlap_score = _minimal_canonical_score()
    overlap_score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "4/4",
            "start": {"numerator": 3, "denominator": 4},
            "duration": {"numerator": 1, "denominator": 1},
        }
    )
    overlap_result = _build_written_time_map_result(
        score=overlap_score,
        relative_path="overlap.json",
    )

    assert overlap_result.passed is False
    assert overlap_result.error_count == EXPECTED_TIME_MAP_CONTIGUITY_ERRORS
    assert overlap_result.diagnostics[0].path == "timelineBars[1].start"


def test_build_written_time_map_warns_for_pickup_like_geometry_without_anacrusis():
    score = _minimal_canonical_score()
    score["timelineBars"][0]["duration"] = {"numerator": 1, "denominator": 4}

    result = _build_written_time_map_result(
        score=score,
        relative_path="pickup-like.json",
    )

    assert result.passed is True
    assert result.warning_count == 1
    assert result.diagnostics[0].severity is DiagnosticSeverity.WARNING
    assert result.diagnostics[0].path == "timelineBars[0].duration"


def test_emit_parts_and_staves_maps_single_staff_tracks_deterministically():
    score = _minimal_canonical_score()

    result = _emit_parts_and_staves_result(score=score, relative_path="single.json")

    assert result.passed is True
    assert len(result.parts) == 1
    assert len(result.staves) == 1
    assert result.parts[0].part_id == "part:1"
    assert result.parts[0].staff_ids == ("staff:part:1:0",)
    assert result.parts[0].transposition.chromatic == 0
    assert result.staves[0].staff_id == "staff:part:1:0"


def test_emit_parts_and_staves_supports_multi_staff_and_transposed_parts():
    score = _load_fixture("ensemble_polyphony_controls.json")

    result = _emit_parts_and_staves_result(
        score=score,
        relative_path="ensemble_polyphony_controls.json",
    )

    assert result.passed is True
    assert [part.part_id for part in result.parts] == ["part:1", "part:2"]
    assert result.parts[0].transposition.chromatic == EXPECTED_TRANSPOSED_CHROMATIC
    assert result.parts[1].staff_ids == ("staff:part:2:0", "staff:part:2:1")
    assert [
        staff.staff_index for staff in result.staves if staff.part_id == "part:2"
    ] == [
        0,
        1,
    ]


def test_emit_parts_and_staves_extracts_tuning_and_capo_context():
    score = _minimal_canonical_score()
    score["tracks"][0]["staves"][0]["tuning"] = {
        "pitches": [64, 59, 55, 50, 45, 40],
        "label": "EADGBE",
    }
    score["tracks"][0]["staves"][0]["capoFret"] = 2

    result = _emit_parts_and_staves_result(
        score=score,
        relative_path="tuning-capo.json",
    )

    assert result.passed is True
    assert result.staves[0].tuning_pitches == (64, 59, 55, 50, 45, 40)
    assert result.staves[0].capo_fret == EXPECTED_CAPO_FRET


def test_emit_bars_maps_standard_bars_from_the_written_time_map():
    score = _minimal_canonical_score()
    score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "4/4",
            "start": {"numerator": 1, "denominator": 1},
            "duration": {"numerator": 1, "denominator": 1},
        }
    )

    result = _emit_bars_result(score=score, relative_path="bars.json")

    assert result.passed is True
    assert [bar.bar_id for bar in result.bars] == ["bar:0", "bar:1"]
    assert result.bars[0].start == ScoreTime(0, 1)
    assert result.bars[1].duration == ScoreTime(1, 1)


def test_emit_bars_extracts_key_and_triplet_feel_metadata():
    score = _minimal_canonical_score()
    score["timelineBars"][0]["keyAccidentalCount"] = 0
    score["timelineBars"][0]["keyMode"] = "major"
    score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "3/4",
            "start": {"numerator": 1, "denominator": 1},
            "duration": {"numerator": 3, "denominator": 4},
            "keyAccidentalCount": EXPECTED_KEY_ACCIDENTALS,
            "keyMode": "minor",
            "tripletFeel": "eighth",
        }
    )

    result = _emit_bars_result(score=score, relative_path="bar-metadata.json")

    assert result.passed is True
    assert result.bars[1].time_signature == TimeSignature(3, 4)
    assert result.bars[1].key_accidental_count == EXPECTED_KEY_ACCIDENTALS
    assert result.bars[1].key_mode == "minor"
    assert result.bars[1].triplet_feel == "eighth"


def test_emit_voice_lanes_maps_single_voice_lanes_deterministically():
    score = _minimal_canonical_score()
    score["timelineBars"].append(
        {
            "index": 1,
            "timeSignature": "4/4",
            "start": {"numerator": 1, "denominator": 1},
            "duration": {"numerator": 1, "denominator": 1},
        }
    )
    score["tracks"][0]["staves"][0]["measures"].append(
        {
            "index": 1,
            "staffIndex": 0,
            "voices": [
                {
                    "voiceIndex": 0,
                    "beats": [
                        {
                            "id": 101,
                            "offset": {"numerator": 0, "denominator": 1},
                            "duration": {"numerator": 1, "denominator": 4},
                            "notes": [],
                        }
                    ],
                }
            ],
        }
    )

    result = _emit_voice_lanes_result(score=score, relative_path="single-voice.json")

    assert result.passed is True
    assert [voice_lane.voice_lane_id for voice_lane in result.voice_lanes] == [
        "voice:staff:part:1:0:0:0",
        "voice:staff:part:1:0:1:0",
    ]
    assert all(
        voice_lane.voice_lane_chain_id == "voice-chain:part:1:staff:part:1:0:0"
        for voice_lane in result.voice_lanes
    )


def test_emit_voice_lanes_keeps_chain_ids_stable_when_voices_reenter():
    score = _load_fixture("voice_reentry.json")

    result = _emit_voice_lanes_result(
        score=score,
        relative_path="voice_reentry.json",
    )

    assert result.passed is True
    assert len(result.voice_lanes) == EXPECTED_VOICE_LANES_WITH_REENTRY

    voice_one_lanes = [
        voice_lane for voice_lane in result.voice_lanes if voice_lane.voice_index == 1
    ]
    assert [voice_lane.bar_id for voice_lane in voice_one_lanes] == ["bar:0", "bar:2"]
    assert {voice_lane.voice_lane_chain_id for voice_lane in voice_one_lanes} == {
        "voice-chain:part:4:staff:part:4:0:1"
    }


def test_emit_voice_lanes_skips_empty_placeholder_voices():
    score = _minimal_canonical_score()
    score["tracks"][0]["staves"][0]["measures"][0]["voices"].append(
        {
            "voiceIndex": 1,
            "beats": [],
        }
    )

    result = _emit_voice_lanes_result(
        score=score,
        relative_path="empty-placeholder-voice.json",
    )

    assert result.passed is True
    assert len(result.voice_lanes) == 1
    assert result.voice_lanes[0].voice_index == 0


def test_emit_onset_groups_maps_standard_beats_to_absolute_onsets():
    score = _minimal_canonical_score()

    result = _emit_onset_groups_result(score=score, relative_path="single-onset.json")

    assert result.passed is True
    assert len(result.onset_groups) == 1
    onset = result.onset_groups[0]
    assert onset.onset_id == "onset:voice:staff:part:1:0:0:0:0"
    assert onset.voice_lane_id == "voice:staff:part:1:0:0:0"
    assert onset.bar_id == "bar:0"
    assert onset.time == ScoreTime(0, 1)
    assert onset.duration_notated == ScoreTime(1, 4)
    assert onset.is_rest is False
    assert onset.attack_order_in_voice == 0
    assert onset.duration_sounding_max == ScoreTime(1, 4)
    assert onset.grace_type is None
    assert onset.rhythm_shape is None


def test_emit_onset_groups_marks_rests_and_caches_sounding_max_duration():
    score = _load_fixture("single_track_monophonic_pickup.json")

    result = _emit_onset_groups_result(
        score=score,
        relative_path="single_track_monophonic_pickup.json",
    )

    assert result.passed is True
    assert [onset.time for onset in result.onset_groups] == [
        ScoreTime(0, 1),
        ScoreTime(1, 4),
        ScoreTime(1, 2),
        ScoreTime(3, 4),
    ]
    assert result.onset_groups[0].duration_sounding_max == ScoreTime(1, 2)
    assert result.onset_groups[1].is_rest is True
    assert result.onset_groups[1].duration_sounding_max is None
    assert [onset.attack_order_in_voice for onset in result.onset_groups] == [
        0,
        0,
        1,
        2,
    ]


def test_emit_onset_groups_preserves_grace_notes_tuplets_and_same_time_ordering():
    score = _load_fixture("guitar_techniques_tuplets.json")

    result = _emit_onset_groups_result(
        score=score,
        relative_path="guitar_techniques_tuplets.json",
    )

    assert result.passed is True
    assert [onset.attack_order_in_voice for onset in result.onset_groups] == [
        0,
        1,
        2,
        3,
    ]
    assert [onset.time for onset in result.onset_groups] == [
        ScoreTime(0, 1),
        ScoreTime(0, 1),
        ScoreTime(1, 12),
        ScoreTime(1, 6),
    ]
    assert result.onset_groups[0].grace_type == "acciaccatura"
    assert result.onset_groups[0].duration_sounding_max == ScoreTime(1, 16)
    assert result.onset_groups[1].rhythm_shape is not None
    assert result.onset_groups[1].rhythm_shape.base_value is RhythmBaseValue.EIGHTH
    assert result.onset_groups[1].rhythm_shape.primary_tuplet is not None
    assert (
        result.onset_groups[1].rhythm_shape.primary_tuplet.numerator
        == EXPECTED_PRIMARY_TUPLET_NUMERATOR
    )
    assert (
        result.onset_groups[1].rhythm_shape.primary_tuplet.denominator
        == EXPECTED_PRIMARY_TUPLET_DENOMINATOR
    )


def test_emit_onset_groups_do_not_create_implicit_onsets_when_voices_reenter():
    score = _load_fixture("voice_reentry.json")

    result = _emit_onset_groups_result(
        score=score,
        relative_path="voice_reentry.json",
    )

    assert result.passed is True
    assert len(result.onset_groups) == EXPECTED_VOICE_LANES_WITH_REENTRY
    voice_one_onsets = [
        onset for onset in result.onset_groups if onset.voice_lane_id.endswith(":1")
    ]
    assert [onset.bar_id for onset in voice_one_onsets] == ["bar:0", "bar:2"]
    assert all(onset.attack_order_in_voice == 0 for onset in voice_one_onsets)


def test_emit_point_control_events_maps_supported_kinds_in_canonical_order():
    score = _load_fixture("ensemble_polyphony_controls.json")

    result = _emit_point_control_events_result(
        score=score,
        relative_path="ensemble_polyphony_controls.json",
    )

    assert result.passed is True
    assert len(result.point_control_events) == EXPECTED_POINT_CONTROLS_IN_FIXTURE
    assert [event.control_id for event in result.point_control_events] == [
        "ctrlp:part:0",
        "ctrlp:score:0",
        "ctrlp:score:1",
        "ctrlp:score:2",
    ]
    assert [event.kind for event in result.point_control_events] == [
        "dynamic_change",
        "tempo_change",
        "tempo_change",
        "fermata",
    ]
    assert [event.target_ref for event in result.point_control_events] == [
        "part:1",
        "score",
        "score",
        "score",
    ]
    assert [event.time for event in result.point_control_events] == [
        ScoreTime(0, 1),
        ScoreTime(0, 1),
        ScoreTime(1, 1),
        ScoreTime(2, 1),
    ]
    assert result.point_control_events[0].value.marking == "mp"
    assert (
        result.point_control_events[1].value.beats_per_minute
        == EXPECTED_FIRST_FIXTURE_TEMPO
    )
    assert (
        result.point_control_events[2].value.beats_per_minute
        == EXPECTED_SECOND_FIXTURE_TEMPO
    )
    assert result.point_control_events[3].value.fermata_type == "normal"
    assert (
        result.point_control_events[3].value.length_scale
        == EXPECTED_FERMATA_LENGTH_SCALE
    )


def test_emit_point_control_events_resolves_staff_and_voice_targets():
    score = _minimal_canonical_score()
    score["pointControls"] = [
        {
            "kind": "Dynamic",
            "scope": "Voice",
            "trackId": 1,
            "staffIndex": 0,
            "voiceIndex": 0,
            "position": {
                "barIndex": 0,
                "offset": {"numerator": 0, "denominator": 1},
            },
            "value": "ff",
        },
        {
            "kind": "Dynamic",
            "scope": "Staff",
            "trackId": 1,
            "staffIndex": 0,
            "position": {
                "barIndex": 0,
                "offset": {"numerator": 0, "denominator": 1},
            },
            "value": "mf",
        },
    ]

    result = _emit_point_control_events_result(
        score=score,
        relative_path="staff-and-voice-point-controls.json",
    )

    assert result.passed is True
    assert [event.control_id for event in result.point_control_events] == [
        "ctrlp:staff:0",
        "ctrlp:voice:0",
    ]
    assert [event.scope for event in result.point_control_events] == [
        "staff",
        "voice",
    ]
    assert [event.target_ref for event in result.point_control_events] == [
        "staff:part:1:0",
        "voice:staff:part:1:0:0:0",
    ]


def test_emit_span_control_events_maps_supported_kinds_in_canonical_order():
    score = _load_fixture("ensemble_polyphony_controls.json")

    result = _emit_span_control_events_result(
        score=score,
        relative_path="ensemble_polyphony_controls.json",
    )

    assert result.passed is True
    assert len(result.span_control_events) == EXPECTED_SPAN_CONTROLS_IN_FIXTURE
    assert [event.control_id for event in result.span_control_events] == [
        "ctrls:part:0",
        "ctrls:staff:0",
    ]
    assert [event.kind for event in result.span_control_events] == [
        "hairpin",
        "ottava",
    ]
    assert [event.target_ref for event in result.span_control_events] == [
        "part:1",
        "staff:part:2:0",
    ]
    assert [event.start_time for event in result.span_control_events] == [
        ScoreTime(1, 4),
        ScoreTime(1, 1),
    ]
    assert [event.end_time for event in result.span_control_events] == [
        ScoreTime(1, 1),
        ScoreTime(2, 1),
    ]
    assert result.span_control_events[0].value.direction == "crescendo"
    assert result.span_control_events[0].value.niente is False
    assert result.span_control_events[0].start_anchor_ref is None
    assert result.span_control_events[0].end_anchor_ref is None
    assert result.span_control_events[1].value.octave_shift == 1


def test_emit_span_control_events_reports_unsupported_span_kinds():
    score = _minimal_canonical_score()
    score["spanControls"] = [
        {
            "kind": "Legato",
            "scope": "Track",
            "trackId": 1,
            "start": {
                "barIndex": 0,
                "offset": {"numerator": 0, "denominator": 1},
            },
            "end": {
                "barIndex": 0,
                "offset": {"numerator": 1, "denominator": 4},
            },
            "value": "slur",
        }
    ]

    result = _emit_span_control_events_result(
        score=score,
        relative_path="unsupported-span-kind.json",
    )

    assert result.passed is True
    assert len(result.span_control_events) == 0
    assert result.warning_count == 1
    assert result.diagnostics[0].code == "unsupported_span_control_kind"
    assert result.diagnostics[0].severity is DiagnosticSeverity.WARNING


def _build_written_time_map_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    return build_written_time_map(documents, validation_results)[0]


def _emit_parts_and_staves_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    return emit_parts_and_staves(documents, validation_results)[0]


def _emit_bars_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    written_time_maps = build_written_time_map(documents, validation_results)
    return emit_bars(documents, written_time_maps)[0]


def _emit_voice_lanes_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    part_staff_emissions = emit_parts_and_staves(documents, validation_results)
    written_time_maps = build_written_time_map(documents, validation_results)
    bar_emissions = emit_bars(documents, written_time_maps)
    return emit_voice_lanes(documents, part_staff_emissions, bar_emissions)[0]


def _emit_onset_groups_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    part_staff_emissions = emit_parts_and_staves(documents, validation_results)
    written_time_maps = build_written_time_map(documents, validation_results)
    bar_emissions = emit_bars(documents, written_time_maps)
    voice_lane_emissions = emit_voice_lanes(
        documents,
        part_staff_emissions,
        bar_emissions,
    )
    return emit_onset_groups(
        documents,
        written_time_maps,
        voice_lane_emissions,
    )[0]


def _emit_point_control_events_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    part_staff_emissions = emit_parts_and_staves(documents, validation_results)
    written_time_maps = build_written_time_map(documents, validation_results)
    bar_emissions = emit_bars(documents, written_time_maps)
    voice_lane_emissions = emit_voice_lanes(
        documents,
        part_staff_emissions,
        bar_emissions,
    )
    return emit_point_control_events(
        documents,
        written_time_maps,
        part_staff_emissions,
        voice_lane_emissions,
    )[0]


def _emit_span_control_events_result(
    score: dict[str, object],
    relative_path: str,
):
    documents = [
        MotifJsonDocument(
            relative_path=relative_path,
            sha256=relative_path,
            file_size_bytes=0,
            score=score,
        )
    ]
    validation_results = validate_canonical_score_surface(documents)
    part_staff_emissions = emit_parts_and_staves(documents, validation_results)
    written_time_maps = build_written_time_map(documents, validation_results)
    bar_emissions = emit_bars(documents, written_time_maps)
    voice_lane_emissions = emit_voice_lanes(
        documents,
        part_staff_emissions,
        bar_emissions,
    )
    return emit_span_control_events(
        documents,
        written_time_maps,
        part_staff_emissions,
        voice_lane_emissions,
    )[0]


def _minimal_canonical_score() -> dict[str, object]:
    score = {
        "tracks": [
            {
                "id": 1,
                "instrument": {
                    "family": 1,
                    "kind": 101,
                    "role": 1,
                },
                "transposition": {
                    "chromatic": 0,
                    "octave": 0,
                    "writtenMinusSoundingSemitones": 0,
                },
                "staves": [
                    {
                        "staffIndex": 0,
                        "measures": [
                            {
                                "index": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 100,
                                                "offset": {
                                                    "numerator": 0,
                                                    "denominator": 1,
                                                },
                                                "duration": {
                                                    "numerator": 1,
                                                    "denominator": 4,
                                                },
                                                "notes": [
                                                    {
                                                        "id": 1000,
                                                        "pitch": {
                                                            "step": "C",
                                                            "octave": 4,
                                                        },
                                                        "duration": {
                                                            "numerator": 1,
                                                            "denominator": 4,
                                                        },
                                                        "soundingDuration": {
                                                            "numerator": 1,
                                                            "denominator": 4,
                                                        },
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        "timelineBars": [
            {
                "index": 0,
                "timeSignature": "4/4",
                "start": {"numerator": 0, "denominator": 1},
                "duration": {"numerator": 1, "denominator": 1},
            }
        ],
        "pointControls": [],
        "spanControls": [],
        "anacrusis": False,
    }
    return deepcopy(score)


def _load_fixture(filename: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / filename).read_text(encoding="utf-8"))
