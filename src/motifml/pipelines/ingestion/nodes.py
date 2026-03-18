"""Nodes for indexing and summarizing the raw Motif JSON corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import (
    MotifJsonDocument,
    RawMotifScore,
    RawMotifTrack,
)
from motifml.pipelines.ingestion.models import (
    NamedCount,
    RawCorpusManifestEntry,
    RawCorpusSummary,
)

UNKNOWN_VALUE = "<unknown>"


def build_raw_corpus_manifest(
    documents: list[MotifJsonDocument],
) -> list[RawCorpusManifestEntry]:
    """Build a deterministic manifest from the raw Motif JSON corpus.

    Args:
        documents: Raw Motif JSON documents loaded from the `01_raw` data stage.

    Returns:
        A path-sorted manifest with stable metadata for each score.
    """
    manifest = [
        RawCorpusManifestEntry(
            relative_path=document.relative_path,
            sha256=document.sha256,
            file_size_bytes=document.file_size_bytes,
            title=_normalize_optional_text(document.score.get("Title"))
            or UNKNOWN_VALUE,
            artist=_normalize_optional_text(document.score.get("Artist")),
            album=_normalize_optional_text(document.score.get("Album")),
            track_count=len(_get_tracks(document.score)),
            playback_bar_count=len(_get_playback_bars(document.score)),
            track_names=_get_track_names(_get_tracks(document.score)),
        )
        for document in documents
    ]

    return sorted(manifest, key=lambda entry: entry.relative_path.casefold())


def summarize_raw_corpus(
    manifest: list[RawCorpusManifestEntry] | list[dict[str, Any]],
) -> RawCorpusSummary:
    """Aggregate manifest entries into a compact corpus summary.

    Args:
        manifest: File-level manifest entries for the raw corpus.

    Returns:
        Aggregate counts and per-artist/per-album rollups.
    """
    manifest_entries = _coerce_manifest_entries(manifest)

    artist_counter = Counter(
        entry.artist if entry.artist is not None else UNKNOWN_VALUE
        for entry in manifest_entries
    )
    album_counter = Counter(
        entry.album if entry.album is not None else UNKNOWN_VALUE
        for entry in manifest_entries
    )

    return RawCorpusSummary(
        total_files=len(manifest_entries),
        total_tracks=sum(entry.track_count for entry in manifest_entries),
        total_playback_bars=sum(entry.playback_bar_count for entry in manifest_entries),
        unique_artists=len(
            {entry.artist for entry in manifest_entries if entry.artist is not None}
        ),
        unique_albums=len(
            {entry.album for entry in manifest_entries if entry.album is not None}
        ),
        files_without_artist=sum(
            1 for entry in manifest_entries if entry.artist is None
        ),
        files_without_album=sum(1 for entry in manifest_entries if entry.album is None),
        max_track_count=max(
            (entry.track_count for entry in manifest_entries), default=0
        ),
        max_playback_bars=max(
            (entry.playback_bar_count for entry in manifest_entries), default=0
        ),
        artist_counts=_to_named_counts(artist_counter.items()),
        album_counts=_to_named_counts(album_counter.items()),
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _get_tracks(score: RawMotifScore) -> list[RawMotifTrack]:
    tracks = score.get("Tracks")
    return tracks if tracks is not None else []


def _get_playback_bars(score: RawMotifScore) -> list[object]:
    playback_bars = score.get("PlaybackMasterBarSequence")
    return playback_bars if playback_bars is not None else []


def _get_track_names(tracks: list[RawMotifTrack]) -> tuple[str, ...]:
    return tuple(
        normalized_name
        for track in tracks
        if (normalized_name := _normalize_optional_text(track.get("Name"))) is not None
    )


def _to_named_counts(items: Iterable[tuple[str, int]]) -> tuple[NamedCount, ...]:
    return tuple(
        NamedCount(name=name, count=count)
        for name, count in sorted(
            items, key=lambda item: (-item[1], item[0].casefold())
        )
    )


def _coerce_manifest_entries(
    manifest: list[RawCorpusManifestEntry] | list[dict[str, Any]],
) -> list[RawCorpusManifestEntry]:
    entries: list[RawCorpusManifestEntry] = []
    for entry in manifest:
        if isinstance(entry, RawCorpusManifestEntry):
            entries.append(entry)
            continue

        track_names = entry.get("track_names", [])
        entries.append(
            RawCorpusManifestEntry(
                relative_path=str(entry["relative_path"]),
                sha256=str(entry["sha256"]),
                file_size_bytes=int(entry["file_size_bytes"]),
                title=str(entry["title"]),
                artist=_normalize_optional_text(_optional_string(entry.get("artist"))),
                album=_normalize_optional_text(_optional_string(entry.get("album"))),
                track_count=int(entry["track_count"]),
                playback_bar_count=int(entry["playback_bar_count"]),
                track_names=tuple(str(name) for name in track_names),
            )
        )

    return entries


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)
