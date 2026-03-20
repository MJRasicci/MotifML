"""Tests for deterministic IR review table helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from motifml.ir.review_tables import (
    build_control_event_rows,
    build_onset_note_tables,
    build_structure_summary,
    build_voice_lane_onset_tables,
    format_score_time,
    load_ir_document_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "ir" / "golden"


def test_load_ir_document_record_supports_single_file_and_corpus_member():
    single_file_record = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json"
    )
    corpus_member_record = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT,
        "single_track_monophonic_pickup",
    )

    assert single_file_record.relative_path == "single_track_monophonic_pickup"
    assert corpus_member_record.relative_path == "single_track_monophonic_pickup"
    assert single_file_record.document == corpus_member_record.document


def test_load_ir_document_record_rejects_missing_corpus_member():
    with pytest.raises(ValueError, match="was not found"):
        load_ir_document_record(GOLDEN_FIXTURE_ROOT, "missing_fixture")


def test_build_structure_summary_counts_entities_and_rollups():
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "ensemble_polyphony_controls.ir.json"
    ).document

    summary = build_structure_summary(document)

    assert {
        "parts": summary.part_count,
        "staves": summary.staff_count,
        "bars": summary.bar_count,
        "voice_lanes": summary.voice_lane_count,
        "onsets": summary.onset_count,
        "notes": summary.note_count,
        "point_controls": summary.point_control_count,
        "span_controls": summary.span_control_count,
        "edges": summary.edge_count,
    } == {
        "parts": 2,
        "staves": 3,
        "bars": 2,
        "voice_lanes": 6,
        "onsets": 6,
        "notes": 7,
        "point_controls": 4,
        "span_controls": 2,
        "edges": 25,
    }
    assert [(item.name, item.count) for item in summary.edge_counts_by_type] == [
        ("contains", 22),
        ("next_in_voice", 3),
    ]
    assert [
        (item.bar_index, item.onset_count, item.note_count)
        for item in summary.bar_rollups
    ] == [
        (0, 3, 4),
        (1, 3, 3),
    ]
    assert summary.voice_lane_rollups[0].voice_lane_id.startswith("voice:")


def test_build_voice_lane_onset_tables_preserves_canonical_order_and_bar_offsets():
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json"
    ).document

    tables = build_voice_lane_onset_tables(document)

    assert [(table.bar_index, table.voice_index) for table in tables] == [
        (0, 0),
        (1, 0),
    ]
    assert [row.attack_order_in_voice for row in tables[1].rows] == [0, 1, 2]
    assert [row.note_count for row in tables[1].rows] == [0, 1, 1]
    assert [row.is_rest for row in tables[1].rows] == [True, False, False]
    assert [format_score_time(row.bar_offset) for row in tables[1].rows] == [
        "0/1",
        "1/4",
        "1/2",
    ]


def test_build_onset_note_tables_groups_notes_and_summarizes_techniques():
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json"
    ).document

    tables = build_onset_note_tables(document)

    expected_onset_ids = [
        "onset:voice:staff:part:lead-guitar:0:0:0:0",
        "onset:voice:staff:part:lead-guitar:0:1:0:1",
        "onset:voice:staff:part:lead-guitar:0:1:0:2",
    ]
    assert [table.onset_id for table in tables] == expected_onset_ids
    assert tables[0].rows[0].technique_summary == "tie_origin"
    assert tables[1].rows[0].technique_summary == "tie_destination"
    assert tables[2].rows[0].pitch_text == "G4"


def test_build_control_event_rows_normalizes_point_and_span_controls():
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "ensemble_polyphony_controls.ir.json"
    ).document

    rows = build_control_event_rows(document)

    assert [row.kind for row in rows] == [
        "dynamic_change",
        "tempo_change",
        "hairpin",
        "tempo_change",
        "ottava",
        "fermata",
    ]
    assert rows[0].family == "point"
    assert rows[0].value_summary == "marking=mp"
    assert rows[4].family == "span"
    assert rows[4].kind == "ottava"
    assert rows[4].start_bar_index == 1
    assert rows[4].end_bar_index == 1
    assert rows[-1].family == "point"
    assert rows[-1].kind == "fermata"
