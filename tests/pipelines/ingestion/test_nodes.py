"""Tests for ingestion node behavior."""

from __future__ import annotations

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.pipelines.ingestion.nodes import (
    build_raw_corpus_manifest,
    build_raw_partition_index,
    build_raw_shard_manifests,
    summarize_raw_corpus,
)

EXPECTED_TWO = 2
EXPECTED_THREE = 3
EXPECTED_SHARD_BETA_BYTES = 120


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


def test_build_raw_partition_index_assigns_deterministic_bounded_shards():
    manifest = build_raw_corpus_manifest(
        [
            MotifJsonDocument(
                relative_path="a/Alpha.json",
                sha256="alpha",
                file_size_bytes=100,
                score={"Title": "Alpha", "Tracks": [], "PlaybackMasterBarSequence": []},
            ),
            MotifJsonDocument(
                relative_path="b/Beta.json",
                sha256="beta",
                file_size_bytes=120,
                score={"Title": "Beta", "Tracks": [], "PlaybackMasterBarSequence": []},
            ),
            MotifJsonDocument(
                relative_path="c/Gamma.json",
                sha256="gamma",
                file_size_bytes=140,
                score={"Title": "Gamma", "Tracks": [], "PlaybackMasterBarSequence": []},
            ),
        ]
    )

    partition_index = build_raw_partition_index(
        manifest,
        {
            "max_documents_per_shard": 2,
            "max_raw_bytes_per_shard": 500,
        },
    )

    assert [entry.relative_path for entry in partition_index] == [
        "a/Alpha.json",
        "b/Beta.json",
        "c/Gamma.json",
    ]
    assert [entry.shard_id for entry in partition_index] == [
        "shard-00000",
        "shard-00000",
        "shard-00001",
    ]


def test_build_raw_shard_manifests_rolls_up_partition_index_membership():
    shard_manifests = build_raw_shard_manifests(
        [
            {
                "relative_path": "a/Alpha.json",
                "sha256": "alpha",
                "file_size_bytes": 100,
                "shard_id": "shard-00000",
            },
            {
                "relative_path": "b/Beta.json",
                "sha256": "beta",
                "file_size_bytes": 120,
                "shard_id": "shard-00001",
            },
        ]
    )

    assert [manifest.shard_id for manifest in shard_manifests] == [
        "shard-00000",
        "shard-00001",
    ]
    assert shard_manifests[0].document_relative_paths == ("a/Alpha.json",)
    assert shard_manifests[1].total_file_size_bytes == EXPECTED_SHARD_BETA_BYTES
