"""Regression coverage for the checked-in Motif score schema snapshot."""

from __future__ import annotations

import json
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "temp" / "motif-score.schema.json"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "ir" / "motif-score.schema.json"
)


def test_motif_score_schema_snapshot_matches_the_current_export():
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert snapshot == fixture
    assert snapshot["title"] == "Motif Score"
    assert snapshot["properties"]["timelineBars"]["items"]["type"] == "object"
    beat_properties = snapshot["properties"]["tracks"]["items"]["properties"]["staves"][
        "items"
    ]["properties"]["measures"]["items"]["properties"]["voices"]["items"]["properties"][
        "beats"
    ]["items"]["properties"]
    assert beat_properties["offset"]["type"] == "number"
    assert beat_properties["duration"]["type"] == "number"
    assert (
        snapshot["properties"]["tracks"]["items"]["properties"]["transposition"][
            "properties"
        ]["isSpecified"]["type"]
        == "boolean"
    )
