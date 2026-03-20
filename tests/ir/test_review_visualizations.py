"""Tests for deterministic IR review visualizations."""

from __future__ import annotations

from pathlib import Path

from motifml.ir.review_tables import load_ir_document_record
from motifml.ir.review_visualizations import (
    render_control_timeline_svg,
    render_note_relations_svg,
    render_timeline_plot_svg,
    render_voice_lane_ladder_svg,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "ir" / "golden"


def test_render_timeline_plot_svg_is_deterministic() -> None:
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "ensemble_polyphony_controls.ir.json"
    ).document

    rendered = render_timeline_plot_svg(document)

    assert rendered == render_timeline_plot_svg(document)
    assert "IR Timeline Plot" in rendered
    assert "bar 0 | onsets=3 | notes=4" in rendered
    assert rendered.startswith('<svg xmlns="http://www.w3.org/2000/svg"')


def test_render_voice_lane_ladder_svg_marks_rests_and_chords() -> None:
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json"
    ).document

    rendered = render_voice_lane_ladder_svg(document)

    assert rendered == render_voice_lane_ladder_svg(document)
    assert "Voice-Lane Onset Ladder" in rendered
    assert "v0 | staff:part:lead-guitar:0" in rendered
    assert 'stroke-dasharray="4 3"' in rendered


def test_render_note_relations_svg_draws_relation_legend() -> None:
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json"
    ).document

    rendered = render_note_relations_svg(document)

    assert rendered == render_note_relations_svg(document)
    assert "Note Relation Graph" in rendered
    assert "tie_to" in rendered
    assert "n0 E4" in rendered


def test_render_control_timeline_svg_includes_point_and_span_controls() -> None:
    document = load_ir_document_record(
        GOLDEN_FIXTURE_ROOT / "ensemble_polyphony_controls.ir.json"
    ).document

    rendered = render_control_timeline_svg(document)

    assert rendered == render_control_timeline_svg(document)
    assert "Control Timeline" in rendered
    assert "tempo_change | score | score" in rendered
    assert "hairpin | part | part:clarinet" in rendered
    assert "beats_per_minute=132" in rendered
