"""Typed models used by the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedCount:
    """A stable name/count pair for corpus summary rollups."""

    name: str
    count: int


@dataclass(frozen=True)
class RawCorpusManifestEntry:
    """Metadata extracted from one raw Motif JSON score."""

    relative_path: str
    sha256: str
    file_size_bytes: int
    title: str
    artist: str | None
    album: str | None
    track_count: int
    playback_bar_count: int
    track_names: tuple[str, ...]


@dataclass(frozen=True)
class RawCorpusSummary:
    """Aggregate summary information for the raw Motif JSON corpus."""

    total_files: int
    total_tracks: int
    total_playback_bars: int
    unique_artists: int
    unique_albums: int
    files_without_artist: int
    files_without_album: int
    max_track_count: int
    max_playback_bars: int
    artist_counts: tuple[NamedCount, ...]
    album_counts: tuple[NamedCount, ...]
