"""Tests for IR build validation nodes."""

from __future__ import annotations

from copy import deepcopy

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.pipelines.ir_build.models import DiagnosticSeverity
from motifml.pipelines.ir_build.nodes import validate_canonical_score_surface

EXPECTED_MISSING_FIELD_ERRORS = 6
EXPECTED_INVALID_SCORE_TIME_ERRORS = 2


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


def _minimal_canonical_score() -> dict[str, object]:
    score = {
        "tracks": [
            {
                "id": 1,
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
