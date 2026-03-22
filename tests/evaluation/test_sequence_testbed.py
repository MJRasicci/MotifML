"""Tests for the sequence-model notebook testbed helpers."""

from __future__ import annotations

from pathlib import Path

from motifml.evaluation.sequence_testbed import (
    GeneratedNote,
    OnsetTemplate,
    ParsedBar,
    ParsedOnset,
    TrackTemplate,
    _build_artifact_paths,
    _build_generated_bar_summaries,
    _parse_bars,
)

LEADING_SHIFT_TICKS = 24
SECOND_ONSET_OFFSET_TICKS = 96


def test_parse_bars_tracks_leading_shift_and_in_bar_onset_offsets() -> None:
    tokens = (
        "<bos>",
        f"TIME_SHIFT:{LEADING_SHIFT_TICKS}",
        "STRUCTURE:PART",
        "STRUCTURE:STAFF",
        "STRUCTURE:BAR",
        "STRUCTURE:VOICE_LANE",
        "STRUCTURE:VOICE_LANE",
        "STRUCTURE:ONSET_GROUP",
        "NOTE_PITCH:C4",
        "NOTE_DURATION:96",
        "STRUCTURE:ONSET_GROUP",
        "NOTE_PITCH:E3",
        "NOTE_DURATION:192",
        f"TIME_SHIFT:{SECOND_ONSET_OFFSET_TICKS}",
        "STRUCTURE:ONSET_GROUP",
        "NOTE_PITCH:D4",
        "NOTE_DURATION:96",
        "<eos>",
    )

    parsed_bars = _parse_bars(tokens)

    assert len(parsed_bars) == 1
    assert parsed_bars[0].leading_time_shift_ticks == LEADING_SHIFT_TICKS
    assert parsed_bars[0].onset_offsets == (0, 0, SECOND_ONSET_OFFSET_TICKS)
    assert parsed_bars[0].raw_note_labels == ("C4", "E3", "D4")
    assert parsed_bars[0].parse_error is None


def test_build_generated_bar_summaries_requires_exact_bar_fill() -> None:
    template_tracks = (
        TrackTemplate(
            track_name="Electric Guitar",
            measure_template={"voices": [{"beats": [{"notes": [{}]}]}]},
            bar_duration_ticks=96,
            onset_templates=(
                OnsetTemplate(
                    beat_index=0,
                    beat_template={
                        "offset": {"numerator": 0, "denominator": 1},
                        "notes": [
                            {"pitch": {"step": "C", "accidental": "", "octave": 4}}
                        ],
                    },
                    offset_ticks=0,
                    json_pitch_labels=("C4",),
                    permutation=(0,),
                ),
            ),
        ),
    )
    parsed_generated_bars = (
        ParsedBar(
            leading_time_shift_ticks=96,
            raw_tokens=("STRUCTURE:BAR",),
            onsets=(
                ParsedOnset(
                    offset_ticks=0,
                    notes=(
                        GeneratedNote(
                            pitch_token="NOTE_PITCH:D4",
                            duration_token="NOTE_DURATION:96",
                        ),
                    ),
                ),
            ),
            parse_error=None,
        ),
        ParsedBar(
            leading_time_shift_ticks=0,
            raw_tokens=("STRUCTURE:BAR",),
            onsets=(
                ParsedOnset(
                    offset_ticks=0,
                    notes=(
                        GeneratedNote(
                            pitch_token="NOTE_PITCH:E4",
                            duration_token="NOTE_DURATION:48",
                        ),
                    ),
                ),
            ),
            parse_error=None,
        ),
    )

    summaries = _build_generated_bar_summaries(
        parsed_generated_bars,
        template_tracks,
        requested_complete_bars=1,
    )

    assert summaries[0].is_structurally_complete is True
    assert summaries[0].exported is True
    assert summaries[0].track_summaries[0].reaches_bar_end is True
    assert summaries[1].is_structurally_complete is False
    assert summaries[1].exported is False
    assert (
        summaries[1].rejection_reason
        == "generated durations do not fill the source bar duration exactly."
    )


def test_build_artifact_paths_derives_names_from_input_path() -> None:
    repo_root = Path("/repo")
    input_score_path = repo_root / "temp" / "SequenceTestFiles" / "ChordTest3.gp"

    paths = _build_artifact_paths(
        repo_root,
        input_score_path,
        requested_complete_bars=2,
    )

    assert paths.artifact_root == (
        repo_root
        / "temp"
        / "roundtrip"
        / "sequence_testbed"
        / "temp__SequenceTestFiles__ChordTest3"
    )
    assert paths.source_json_path.name == "ChordTest3.mapped.json"
    assert paths.source_motif_path.name == "ChordTest3.source.motif"
    assert paths.output_json_path.name == "ChordTest3.2bars.continued.json"
    assert paths.output_gp_path.name == "ChordTest3.2bars.continued.gp"
