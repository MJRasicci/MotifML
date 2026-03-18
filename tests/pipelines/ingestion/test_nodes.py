"""Tests for ingestion node behavior."""

from __future__ import annotations

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.pipelines.ingestion.nodes import (
    build_raw_corpus_manifest,
    summarize_raw_corpus,
)

EXPECTED_TWO = 2
EXPECTED_THREE = 3


def test_build_raw_corpus_manifest_extracts_deterministic_metadata():
    documents = [
        MotifJsonDocument(
            relative_path="b/Beta.json",
            sha256="beta",
            file_size_bytes=200,
            score={
                "Title": "Beta",
                "Artist": "Artist B",
                "Album": "Album Z",
                "Tracks": [{"Name": "Rhythm"}, {"Name": "Bass"}],
                "PlaybackMasterBarSequence": [{}, {}, {}],
            },
        ),
        MotifJsonDocument(
            relative_path="a/Alpha.json",
            sha256="alpha",
            file_size_bytes=100,
            score={
                "Title": "Alpha",
                "Artist": "Artist A",
                "Album": "Album A",
                "Tracks": [{"Name": "Lead"}, {"Name": " "}],
                "PlaybackMasterBarSequence": [{}],
            },
        ),
    ]

    manifest = build_raw_corpus_manifest(documents)

    assert [entry.relative_path for entry in manifest] == [
        "a/Alpha.json",
        "b/Beta.json",
    ]
    assert manifest[0].track_names == ("Lead",)
    assert manifest[1].track_count == EXPECTED_TWO
    assert manifest[1].playback_bar_count == EXPECTED_THREE


def test_summarize_raw_corpus_rolls_up_counts_and_unknowns():
    manifest = build_raw_corpus_manifest(
        [
            MotifJsonDocument(
                relative_path="a/Alpha.json",
                sha256="alpha",
                file_size_bytes=100,
                score={
                    "Title": "Alpha",
                    "Artist": "Artist A",
                    "Album": "Album A",
                    "Tracks": [{"Name": "Lead"}],
                    "PlaybackMasterBarSequence": [{}, {}],
                },
            ),
            MotifJsonDocument(
                relative_path="b/Beta.json",
                sha256="beta",
                file_size_bytes=200,
                score={
                    "Title": "Beta",
                    "Artist": "Artist A",
                    "Album": "",
                    "Tracks": [{"Name": "Rhythm"}, {"Name": "Bass"}],
                    "PlaybackMasterBarSequence": [{}],
                },
            ),
            MotifJsonDocument(
                relative_path="c/Gamma.json",
                sha256="gamma",
                file_size_bytes=300,
                score={
                    "Title": "Gamma",
                    "Tracks": [],
                    "PlaybackMasterBarSequence": [],
                },
            ),
        ]
    )

    summary = summarize_raw_corpus(manifest)

    assert summary.total_files == EXPECTED_THREE
    assert summary.total_tracks == EXPECTED_THREE
    assert summary.total_playback_bars == EXPECTED_THREE
    assert summary.unique_artists == 1
    assert summary.unique_albums == 1
    assert summary.files_without_artist == 1
    assert summary.files_without_album == EXPECTED_TWO
    assert summary.artist_counts[0].name == "Artist A"
    assert summary.artist_counts[0].count == EXPECTED_TWO
    assert summary.album_counts[0].name == "<unknown>"
    assert summary.album_counts[0].count == EXPECTED_TWO
