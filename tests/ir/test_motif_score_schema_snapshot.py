"""Regression coverage for the checked-in Motif score schema snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "temp" / "motif-score.schema.json"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ir" / "motif-score.schema.json"
)


def test_motif_score_schema_snapshot_matches_the_current_export():
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert snapshot == fixture
    assert snapshot["title"] == "Motif Score"
    assert _schema_path(snapshot, "properties", "pointControls", "type") == "array"
    assert _schema_path(snapshot, "properties", "spanControls", "type") == "array"
    assert _schema_path(
        snapshot,
        "properties",
        "timelineBars",
        "items",
        "properties",
        "start",
        "required",
    ) == ["numerator", "denominator"]
    assert _schema_path(
        snapshot,
        "properties",
        "timelineBars",
        "items",
        "properties",
        "duration",
        "required",
    ) == ["numerator", "denominator"]
    assert _schema_path(
        snapshot,
        "properties",
        "tracks",
        "items",
        "properties",
        "staves",
        "items",
        "properties",
        "measures",
        "items",
        "properties",
        "voices",
        "items",
        "properties",
        "beats",
        "items",
        "properties",
        "offset",
        "required",
    ) == ["numerator", "denominator"]
    assert _schema_path(
        snapshot,
        "properties",
        "tracks",
        "items",
        "properties",
        "staves",
        "items",
        "properties",
        "measures",
        "items",
        "properties",
        "voices",
        "items",
        "properties",
        "beats",
        "items",
        "properties",
        "duration",
        "required",
    ) == ["numerator", "denominator"]
    assert (
        _schema_path(
            snapshot,
            "properties",
            "tracks",
            "items",
            "properties",
            "staves",
            "items",
            "properties",
            "measures",
            "items",
            "properties",
            "voices",
            "items",
            "properties",
            "beats",
            "items",
            "properties",
            "notes",
            "items",
            "properties",
            "articulation",
            "properties",
            "relations",
            "type",
        )
        == "array"
    )

    assert "tempoChanges" not in snapshot["properties"]
    assert "isSpecified" not in _schema_path(
        snapshot,
        "properties",
        "tracks",
        "items",
        "properties",
        "transposition",
        "properties",
    )
    assert "dynamic" not in _schema_path(
        snapshot,
        "properties",
        "tracks",
        "items",
        "properties",
        "staves",
        "items",
        "properties",
        "measures",
        "items",
        "properties",
        "voices",
        "items",
        "properties",
        "beats",
        "items",
        "properties",
    )
    assert "repeatStartAttributePresent" not in _schema_path(
        snapshot,
        "properties",
        "timelineBars",
        "items",
        "properties",
    )


def _schema_path(document: dict[str, Any], *segments: str) -> Any:
    value: Any = document
    for segment in segments:
        value = value[segment]

    return value
