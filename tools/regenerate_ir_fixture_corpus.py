#!/usr/bin/env python3
"""Regenerate the tracked raw Motif JSON fixtures and golden IR artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motifml.ir.ids import (
    bar_id,
    note_id,
    onset_id,
    part_id,
    point_control_id,
    span_control_id,
    staff_id,
    voice_lane_chain_id,
    voice_lane_id,
)
from motifml.ir.models import (
    Bar,
    ControlScope,
    DynamicChangeValue,
    Edge,
    EdgeType,
    FermataValue,
    GenericTechniqueFlags,
    HairpinDirection,
    HairpinValue,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    OttavaValue,
    Part,
    Pitch,
    PitchStep,
    PointControlEvent,
    PointControlKind,
    SpanControlEvent,
    SpanControlKind,
    Staff,
    TechniquePayload,
    TempoChangeValue,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.serialization import serialize_document
from motifml.ir.time import ScoreTime

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
RAW_FIXTURE_SUBDIR = Path("motif_json")
GOLDEN_FIXTURE_SUBDIR = Path("ir") / "golden"
CATALOG_FILENAME = "ir_fixture_catalog.json"
CATALOG_VERSION = "1.0.0"
REGENERATION_COMMAND = "uv run python tools/regenerate_ir_fixture_corpus.py"
RAW_SCHEMA_PATH = "ir/motif-score.schema.json"
IR_SCHEMA_PATH = "../../src/motifml/ir/schema/motifml-ir-document.schema.json"
REQUIRED_COVERAGE = (
    "single-track monophonic",
    "multi-track polyphonic",
    "rests",
    "tuplets",
    "ties",
    "hammer-on/pull-off",
    "slide links",
    "bends",
    "harmonics",
    "grace notes",
    "tempo changes",
    "dynamics",
    "fermatas",
    "hairpins",
    "ottava",
    "pickup bars",
    "transposed instruments",
    "multi-staff parts",
    "disappearing/reappearing voices",
)


@dataclass(frozen=True)
class FixtureSpec:
    """One curated raw-score fixture and its optional golden IR counterpart."""

    fixture_id: str
    description: str
    covers: tuple[str, ...]
    raw_builder: Callable[[], dict[str, Any]]
    golden_builder: Callable[[], MotifMlIrDocument] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the tracked Motif JSON fixture corpus and golden IR subset."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="Directory that should receive the generated tests/fixtures layout.",
    )
    args = parser.parse_args()

    generate_fixture_corpus(args.output_root)


def generate_fixture_corpus(output_root: Path = DEFAULT_FIXTURE_ROOT) -> None:
    fixture_root = output_root.resolve()
    raw_fixture_root = fixture_root / RAW_FIXTURE_SUBDIR
    golden_fixture_root = fixture_root / GOLDEN_FIXTURE_SUBDIR
    catalog_path = fixture_root / CATALOG_FILENAME

    raw_fixture_root.mkdir(parents=True, exist_ok=True)
    golden_fixture_root.mkdir(parents=True, exist_ok=True)
    _remove_managed_json_files(raw_fixture_root)
    _remove_managed_json_files(golden_fixture_root)

    catalog_entries: list[dict[str, Any]] = []
    for spec in _fixture_specs():
        raw_path = raw_fixture_root / f"{spec.fixture_id}.json"
        raw_path.write_text(_dump_json(spec.raw_builder()), encoding="utf-8")

        golden_relative_path: str | None = None
        if spec.golden_builder is not None:
            golden_path = golden_fixture_root / f"{spec.fixture_id}.ir.json"
            golden_path.write_text(
                serialize_document(spec.golden_builder()),
                encoding="utf-8",
            )
            golden_relative_path = golden_path.relative_to(fixture_root).as_posix()

        catalog_entries.append(
            {
                "fixture_id": spec.fixture_id,
                "description": spec.description,
                "covers": list(spec.covers),
                "raw_motif_json_path": raw_path.relative_to(fixture_root).as_posix(),
                "golden_ir_path": golden_relative_path,
            }
        )

    catalog = {
        "catalog_version": CATALOG_VERSION,
        "regeneration_command": REGENERATION_COMMAND,
        "required_coverage": list(REQUIRED_COVERAGE),
        "raw_schema_path": RAW_SCHEMA_PATH,
        "ir_schema_path": IR_SCHEMA_PATH,
        "fixtures": catalog_entries,
    }
    catalog_path.write_text(_dump_json(catalog), encoding="utf-8")


def _fixture_specs() -> tuple[FixtureSpec, ...]:
    return (
        FixtureSpec(
            fixture_id="single_track_monophonic_pickup",
            description=(
                "Lead-guitar pickup phrase with a tied continuation and an explicit rest."
            ),
            covers=(
                "single-track monophonic",
                "rests",
                "ties",
                "pickup bars",
            ),
            raw_builder=_single_track_monophonic_pickup_raw,
            golden_builder=_single_track_monophonic_pickup_ir,
        ),
        FixtureSpec(
            fixture_id="ensemble_polyphony_controls",
            description=(
                "Clarinet-plus-piano excerpt with transposition, multi-staff writing, "
                "polyphony, and authored point/span controls."
            ),
            covers=(
                "multi-track polyphonic",
                "tempo changes",
                "dynamics",
                "fermatas",
                "hairpins",
                "ottava",
                "transposed instruments",
                "multi-staff parts",
            ),
            raw_builder=_ensemble_polyphony_controls_raw,
            golden_builder=_ensemble_polyphony_controls_ir,
        ),
        FixtureSpec(
            fixture_id="guitar_techniques_tuplets",
            description=(
                "Single-staff guitar line with grace notes, triplets, linked-note "
                "techniques, bends, and harmonics."
            ),
            covers=(
                "tuplets",
                "hammer-on/pull-off",
                "slide links",
                "bends",
                "harmonics",
                "grace notes",
            ),
            raw_builder=_guitar_techniques_tuplets_raw,
        ),
        FixtureSpec(
            fixture_id="voice_reentry",
            description=(
                "Keyboard texture where the second written voice drops out for a bar "
                "and later reappears."
            ),
            covers=("disappearing/reappearing voices",),
            raw_builder=_voice_reentry_raw,
        ),
    )


def _single_track_monophonic_pickup_raw() -> dict[str, Any]:
    return {
        "title": "Pickup Study",
        "artist": "Fixture Suite",
        "album": "IR Corpus",
        "tracks": [
            {
                "id": 1,
                "name": "Lead Guitar",
                "instrument": {"family": 1, "kind": 101, "role": 1},
                "transposition": _raw_transposition(),
                "staves": [
                    {
                        "staffIndex": 0,
                        "tuning": {
                            "pitches": [64, 59, 55, 50, 45, 40],
                            "label": "EADGBE",
                        },
                        "measures": [
                            {
                                "index": 0,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 100,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 4),
                                                "notes": [
                                                    {
                                                        "id": 1000,
                                                        "velocity": 84,
                                                        "pitch": {
                                                            "step": "E",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 1,
                                                        "duration": _raw_score_time(
                                                            1, 4
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "articulation": {
                                                            "tieOrigin": True,
                                                            "relations": [
                                                                {
                                                                    "kind": "Tie",
                                                                    "targetNoteId": 1001,
                                                                }
                                                            ],
                                                        },
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "index": 1,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 101,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 4),
                                                "notes": [],
                                            },
                                            {
                                                "id": 102,
                                                "offset": _raw_score_time(1, 4),
                                                "duration": _raw_score_time(1, 4),
                                                "notes": [
                                                    {
                                                        "id": 1001,
                                                        "velocity": 80,
                                                        "pitch": {
                                                            "step": "E",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 1,
                                                        "duration": _raw_score_time(
                                                            1, 4
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 4
                                                        ),
                                                        "articulation": {
                                                            "tieDestination": True,
                                                        },
                                                    }
                                                ],
                                            },
                                            {
                                                "id": 103,
                                                "offset": _raw_score_time(1, 2),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 1002,
                                                        "velocity": 88,
                                                        "pitch": {
                                                            "step": "G",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 1,
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    }
                                                ],
                                            },
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
        "timelineBars": [
            {
                "index": 0,
                "timeSignature": "4/4",
                "start": _raw_score_time(0, 1),
                "duration": _raw_score_time(1, 4),
            },
            {
                "index": 1,
                "timeSignature": "4/4",
                "start": _raw_score_time(1, 4),
                "duration": _raw_score_time(1, 1),
            },
        ],
        "pointControls": [
            {
                "kind": "Tempo",
                "scope": "Score",
                "position": {"barIndex": 0, "offset": _raw_score_time(0, 1)},
                "numericValue": 96.0,
            },
            {
                "kind": "Dynamic",
                "scope": "Track",
                "trackId": 1,
                "position": {"barIndex": 0, "offset": _raw_score_time(0, 1)},
                "value": "mp",
            },
        ],
        "spanControls": [],
        "anacrusis": True,
        "playbackMasterBarSequence": [0, 1],
    }


def _ensemble_polyphony_controls_raw() -> dict[str, Any]:
    return {
        "title": "Ensemble Control Study",
        "artist": "Fixture Suite",
        "album": "IR Corpus",
        "tracks": [
            {
                "id": 1,
                "name": "Clarinet in Bb",
                "instrument": {"family": 5, "kind": 502, "role": 1},
                "transposition": _raw_transposition(chromatic=2),
                "staves": [
                    {
                        "staffIndex": 0,
                        "measures": [
                            {
                                "index": 0,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 200,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 2000,
                                                        "velocity": 76,
                                                        "pitch": {
                                                            "step": "D",
                                                            "octave": 5,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "index": 1,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 201,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 1),
                                                "notes": [
                                                    {
                                                        "id": 2001,
                                                        "velocity": 82,
                                                        "pitch": {
                                                            "step": "G",
                                                            "octave": 5,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "id": 2,
                "name": "Piano",
                "instrument": {"family": 8, "kind": 801, "role": 2},
                "transposition": _raw_transposition(),
                "staves": [
                    {
                        "staffIndex": 0,
                        "measures": [
                            {
                                "index": 0,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 210,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 2100,
                                                        "velocity": 72,
                                                        "pitch": {
                                                            "step": "C",
                                                            "octave": 5,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    },
                                                    {
                                                        "id": 2101,
                                                        "velocity": 72,
                                                        "pitch": {
                                                            "step": "E",
                                                            "octave": 5,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    },
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "index": 1,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 211,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 1),
                                                "notes": [
                                                    {
                                                        "id": 2102,
                                                        "velocity": 78,
                                                        "pitch": {
                                                            "step": "D",
                                                            "octave": 6,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                        ],
                    },
                    {
                        "staffIndex": 1,
                        "measures": [
                            {
                                "index": 0,
                                "staffIndex": 1,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 220,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 1),
                                                "notes": [
                                                    {
                                                        "id": 2200,
                                                        "velocity": 68,
                                                        "pitch": {
                                                            "step": "C",
                                                            "octave": 3,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "index": 1,
                                "staffIndex": 1,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 221,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 1),
                                                "notes": [
                                                    {
                                                        "id": 2201,
                                                        "velocity": 68,
                                                        "pitch": {
                                                            "step": "F",
                                                            "octave": 2,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
        "timelineBars": [
            {
                "index": 0,
                "timeSignature": "4/4",
                "start": _raw_score_time(0, 1),
                "duration": _raw_score_time(1, 1),
            },
            {
                "index": 1,
                "timeSignature": "4/4",
                "start": _raw_score_time(1, 1),
                "duration": _raw_score_time(1, 1),
            },
        ],
        "pointControls": [
            {
                "kind": "Tempo",
                "scope": "Score",
                "position": {"barIndex": 0, "offset": _raw_score_time(0, 1)},
                "numericValue": 120.0,
            },
            {
                "kind": "Tempo",
                "scope": "Score",
                "position": {"barIndex": 1, "offset": _raw_score_time(0, 1)},
                "numericValue": 132.0,
            },
            {
                "kind": "Dynamic",
                "scope": "Track",
                "trackId": 1,
                "position": {"barIndex": 0, "offset": _raw_score_time(0, 1)},
                "value": "mp",
            },
            {
                "kind": "Fermata",
                "scope": "Score",
                "position": {"barIndex": 1, "offset": _raw_score_time(1, 1)},
                "value": "normal",
                "length": 1.5,
                "placement": "end",
            },
        ],
        "spanControls": [
            {
                "kind": "Hairpin",
                "scope": "Track",
                "trackId": 1,
                "start": {"barIndex": 0, "offset": _raw_score_time(1, 4)},
                "end": {"barIndex": 1, "offset": _raw_score_time(0, 1)},
                "value": "crescendo",
            },
            {
                "kind": "Ottava",
                "scope": "Staff",
                "trackId": 2,
                "staffIndex": 0,
                "start": {"barIndex": 1, "offset": _raw_score_time(0, 1)},
                "end": {"barIndex": 1, "offset": _raw_score_time(1, 1)},
                "value": "8va",
            },
        ],
        "anacrusis": False,
        "playbackMasterBarSequence": [0, 1],
    }


def _guitar_techniques_tuplets_raw() -> dict[str, Any]:
    triplet_rhythm = {
        "baseValue": "Eighth",
        "augmentationDots": 0,
        "primaryTuplet": {"numerator": 3, "denominator": 2},
    }
    return {
        "title": "Technique Triplets",
        "artist": "Fixture Suite",
        "album": "IR Corpus",
        "tracks": [
            {
                "id": 3,
                "name": "Technique Guitar",
                "instrument": {"family": 1, "kind": 102, "role": 1},
                "transposition": _raw_transposition(),
                "staves": [
                    {
                        "staffIndex": 0,
                        "tuning": {
                            "pitches": [64, 59, 55, 50, 45, 40],
                            "label": "EADGBE",
                        },
                        "measures": [
                            {
                                "index": 0,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 300,
                                                "graceType": "acciaccatura",
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 16),
                                                "notes": [
                                                    {
                                                        "id": 3000,
                                                        "velocity": 72,
                                                        "pitch": {
                                                            "step": "D",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 3,
                                                        "duration": _raw_score_time(
                                                            1, 16
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 16
                                                        ),
                                                    }
                                                ],
                                            },
                                            {
                                                "id": 301,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 12),
                                                "rhythm": triplet_rhythm,
                                                "notes": [
                                                    {
                                                        "id": 3001,
                                                        "velocity": 88,
                                                        "pitch": {
                                                            "step": "E",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 2,
                                                        "duration": _raw_score_time(
                                                            1, 12
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 12
                                                        ),
                                                        "articulation": {
                                                            "hopoOrigin": True,
                                                            "hopoType": 1,
                                                            "slides": [1],
                                                            "harmonic": {
                                                                "type": 1,
                                                                "typeName": "natural",
                                                                "kind": 1,
                                                                "fret": 5.0,
                                                                "enabled": True,
                                                            },
                                                            "bend": {
                                                                "enabled": True,
                                                                "type": 1,
                                                                "originValue": 0.0,
                                                                "destinationValue": 1.0,
                                                            },
                                                            "relations": [
                                                                {
                                                                    "kind": "HammerOn",
                                                                    "targetNoteId": 3002,
                                                                },
                                                                {
                                                                    "kind": "Slide",
                                                                    "targetNoteId": 3003,
                                                                },
                                                            ],
                                                        },
                                                    }
                                                ],
                                            },
                                            {
                                                "id": 302,
                                                "offset": _raw_score_time(1, 12),
                                                "duration": _raw_score_time(1, 12),
                                                "rhythm": triplet_rhythm,
                                                "notes": [
                                                    {
                                                        "id": 3002,
                                                        "velocity": 90,
                                                        "pitch": {
                                                            "step": "F",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 2,
                                                        "duration": _raw_score_time(
                                                            1, 12
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 12
                                                        ),
                                                        "articulation": {
                                                            "hopoOrigin": True,
                                                            "hopoDestination": True,
                                                            "hopoType": 2,
                                                            "relations": [
                                                                {
                                                                    "kind": "PullOff",
                                                                    "targetNoteId": 3003,
                                                                }
                                                            ],
                                                        },
                                                    }
                                                ],
                                            },
                                            {
                                                "id": 303,
                                                "offset": _raw_score_time(1, 6),
                                                "duration": _raw_score_time(1, 12),
                                                "rhythm": triplet_rhythm,
                                                "notes": [
                                                    {
                                                        "id": 3003,
                                                        "velocity": 86,
                                                        "pitch": {
                                                            "step": "G",
                                                            "octave": 4,
                                                        },
                                                        "showStringNumber": True,
                                                        "stringNumber": 2,
                                                        "duration": _raw_score_time(
                                                            1, 12
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 12
                                                        ),
                                                        "articulation": {
                                                            "hopoDestination": True,
                                                        },
                                                    }
                                                ],
                                            },
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
                "start": _raw_score_time(0, 1),
                "duration": _raw_score_time(1, 1),
            }
        ],
        "pointControls": [],
        "spanControls": [],
        "anacrusis": False,
        "playbackMasterBarSequence": [0],
    }


def _voice_reentry_raw() -> dict[str, Any]:
    return {
        "title": "Voice Reentry Etude",
        "artist": "Fixture Suite",
        "album": "IR Corpus",
        "tracks": [
            {
                "id": 4,
                "name": "Keyboard",
                "instrument": {"family": 8, "kind": 802, "role": 2},
                "transposition": _raw_transposition(),
                "staves": [
                    {
                        "staffIndex": 0,
                        "measures": [
                            {
                                "index": 0,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 400,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 4000,
                                                        "velocity": 72,
                                                        "pitch": {
                                                            "step": "C",
                                                            "octave": 4,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                    {
                                        "voiceIndex": 1,
                                        "beats": [
                                            {
                                                "id": 401,
                                                "offset": _raw_score_time(1, 2),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 4001,
                                                        "velocity": 68,
                                                        "pitch": {
                                                            "step": "G",
                                                            "octave": 5,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            },
                            {
                                "index": 1,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 402,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 1),
                                                "notes": [
                                                    {
                                                        "id": 4002,
                                                        "velocity": 70,
                                                        "pitch": {
                                                            "step": "D",
                                                            "octave": 4,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 1
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "index": 2,
                                "staffIndex": 0,
                                "voices": [
                                    {
                                        "voiceIndex": 0,
                                        "beats": [
                                            {
                                                "id": 403,
                                                "offset": _raw_score_time(0, 1),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 4003,
                                                        "velocity": 72,
                                                        "pitch": {
                                                            "step": "E",
                                                            "octave": 4,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                    {
                                        "voiceIndex": 1,
                                        "beats": [
                                            {
                                                "id": 404,
                                                "offset": _raw_score_time(1, 2),
                                                "duration": _raw_score_time(1, 2),
                                                "notes": [
                                                    {
                                                        "id": 4004,
                                                        "velocity": 68,
                                                        "pitch": {
                                                            "step": "A",
                                                            "octave": 5,
                                                        },
                                                        "duration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                        "soundingDuration": _raw_score_time(
                                                            1, 2
                                                        ),
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
        "timelineBars": [
            {
                "index": 0,
                "timeSignature": "4/4",
                "start": _raw_score_time(0, 1),
                "duration": _raw_score_time(1, 1),
            },
            {
                "index": 1,
                "timeSignature": "4/4",
                "start": _raw_score_time(1, 1),
                "duration": _raw_score_time(1, 1),
            },
            {
                "index": 2,
                "timeSignature": "4/4",
                "start": _raw_score_time(2, 1),
                "duration": _raw_score_time(1, 1),
            },
        ],
        "pointControls": [],
        "spanControls": [],
        "anacrusis": False,
        "playbackMasterBarSequence": [0, 1, 2],
    }


def _single_track_monophonic_pickup_ir() -> MotifMlIrDocument:
    part = part_id("lead-guitar")
    staff = staff_id(part, 0)
    bar_zero = bar_id(0)
    bar_one = bar_id(1)
    lane_zero = voice_lane_id(staff, 0, 0)
    lane_one = voice_lane_id(staff, 1, 0)
    chain = voice_lane_chain_id(part, staff, 0)
    onset_zero = onset_id(lane_zero, 0)
    onset_one = onset_id(lane_one, 0)
    onset_two = onset_id(lane_one, 1)
    onset_three = onset_id(lane_one, 2)
    note_zero = note_id(onset_zero, 0)
    note_one = note_id(onset_two, 0)
    note_two = note_id(onset_three, 0)

    return MotifMlIrDocument(
        metadata=IrDocumentMetadata(
            ir_schema_version="1.0.0",
            corpus_build_version="fixture-corpus-v1",
            generator_version="tools/regenerate_ir_fixture_corpus.py",
            source_document_hash="fixture:single_track_monophonic_pickup",
        ),
        parts=(
            Part(
                part_id=part,
                instrument_family=1,
                instrument_kind=101,
                role=1,
                transposition=Transposition(),
                staff_ids=(staff,),
            ),
        ),
        staves=(
            Staff(
                staff_id=staff,
                part_id=part,
                staff_index=0,
                tuning_pitches=(64, 59, 55, 50, 45, 40),
            ),
        ),
        bars=(
            Bar(
                bar_id=bar_zero,
                bar_index=0,
                start=ScoreTime(0, 1),
                duration=ScoreTime(1, 4),
                time_signature=TimeSignature(4, 4),
            ),
            Bar(
                bar_id=bar_one,
                bar_index=1,
                start=ScoreTime(1, 4),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
                anacrusis_context="pickup-continuation",
            ),
        ),
        voice_lanes=(
            VoiceLane(
                voice_lane_id=lane_zero,
                voice_lane_chain_id=chain,
                part_id=part,
                staff_id=staff,
                bar_id=bar_zero,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=lane_one,
                voice_lane_chain_id=chain,
                part_id=part,
                staff_id=staff,
                bar_id=bar_one,
                voice_index=0,
            ),
        ),
        point_control_events=(
            PointControlEvent(
                control_id=point_control_id("score", 0),
                kind=PointControlKind.TEMPO_CHANGE,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(0, 1),
                value=TempoChangeValue(beats_per_minute=96.0),
            ),
            PointControlEvent(
                control_id=point_control_id("part", 0),
                kind=PointControlKind.DYNAMIC_CHANGE,
                scope=ControlScope.PART,
                target_ref=part,
                time=ScoreTime(0, 1),
                value=DynamicChangeValue(marking="mp"),
            ),
        ),
        onset_groups=(
            OnsetGroup(
                onset_id=onset_zero,
                voice_lane_id=lane_zero,
                bar_id=bar_zero,
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 2),
            ),
            OnsetGroup(
                onset_id=onset_one,
                voice_lane_id=lane_one,
                bar_id=bar_one,
                time=ScoreTime(1, 4),
                duration_notated=ScoreTime(1, 4),
                is_rest=True,
                attack_order_in_voice=0,
            ),
            OnsetGroup(
                onset_id=onset_two,
                voice_lane_id=lane_one,
                bar_id=bar_one,
                time=ScoreTime(1, 2),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=1,
                duration_sounding_max=ScoreTime(1, 4),
            ),
            OnsetGroup(
                onset_id=onset_three,
                voice_lane_id=lane_one,
                bar_id=bar_one,
                time=ScoreTime(3, 4),
                duration_notated=ScoreTime(1, 2),
                is_rest=False,
                attack_order_in_voice=2,
                duration_sounding_max=ScoreTime(1, 2),
            ),
        ),
        note_events=(
            NoteEvent(
                note_id=note_zero,
                onset_id=onset_zero,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 2),
                pitch=Pitch(step=PitchStep.E, octave=4),
                velocity=84,
                string_number=1,
                show_string_number=True,
                techniques=TechniquePayload(
                    generic=GenericTechniqueFlags(tie_origin=True),
                ),
            ),
            NoteEvent(
                note_id=note_one,
                onset_id=onset_two,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(1, 2),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step=PitchStep.E, octave=4),
                velocity=80,
                string_number=1,
                show_string_number=True,
                techniques=TechniquePayload(
                    generic=GenericTechniqueFlags(tie_destination=True),
                ),
            ),
            NoteEvent(
                note_id=note_two,
                onset_id=onset_three,
                part_id=part,
                staff_id=staff,
                time=ScoreTime(3, 4),
                attack_duration=ScoreTime(1, 2),
                sounding_duration=ScoreTime(1, 2),
                pitch=Pitch(step=PitchStep.G, octave=4),
                velocity=88,
                string_number=1,
                show_string_number=True,
            ),
        ),
        edges=_contains_edges(
            (part, staff),
            (bar_zero, lane_zero),
            (bar_one, lane_one),
            (lane_zero, onset_zero),
            (lane_one, onset_one),
            (lane_one, onset_two),
            (lane_one, onset_three),
            (onset_zero, note_zero),
            (onset_two, note_one),
            (onset_three, note_two),
        )
        + (
            Edge(
                source_id=onset_zero,
                target_id=onset_one,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
            Edge(
                source_id=onset_one,
                target_id=onset_two,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
            Edge(
                source_id=onset_two,
                target_id=onset_three,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
            Edge(
                source_id=note_zero,
                target_id=note_one,
                edge_type=EdgeType.TIE_TO,
            ),
        ),
    )


def _ensemble_polyphony_controls_ir() -> MotifMlIrDocument:
    clarinet_part = part_id("clarinet")
    piano_part = part_id("piano")
    clarinet_staff = staff_id(clarinet_part, 0)
    piano_rh_staff = staff_id(piano_part, 0)
    piano_lh_staff = staff_id(piano_part, 1)
    bar_zero = bar_id(0)
    bar_one = bar_id(1)
    clarinet_lane_zero = voice_lane_id(clarinet_staff, 0, 0)
    clarinet_lane_one = voice_lane_id(clarinet_staff, 1, 0)
    piano_rh_lane_zero = voice_lane_id(piano_rh_staff, 0, 0)
    piano_rh_lane_one = voice_lane_id(piano_rh_staff, 1, 0)
    piano_lh_lane_zero = voice_lane_id(piano_lh_staff, 0, 0)
    piano_lh_lane_one = voice_lane_id(piano_lh_staff, 1, 0)
    clarinet_chain = voice_lane_chain_id(clarinet_part, clarinet_staff, 0)
    piano_rh_chain = voice_lane_chain_id(piano_part, piano_rh_staff, 0)
    piano_lh_chain = voice_lane_chain_id(piano_part, piano_lh_staff, 0)
    clarinet_onset_zero = onset_id(clarinet_lane_zero, 0)
    clarinet_onset_one = onset_id(clarinet_lane_one, 0)
    piano_rh_onset_zero = onset_id(piano_rh_lane_zero, 0)
    piano_rh_onset_one = onset_id(piano_rh_lane_one, 0)
    piano_lh_onset_zero = onset_id(piano_lh_lane_zero, 0)
    piano_lh_onset_one = onset_id(piano_lh_lane_one, 0)
    clarinet_note_zero = note_id(clarinet_onset_zero, 0)
    clarinet_note_one = note_id(clarinet_onset_one, 0)
    piano_rh_note_zero = note_id(piano_rh_onset_zero, 0)
    piano_rh_note_one = note_id(piano_rh_onset_zero, 1)
    piano_rh_note_two = note_id(piano_rh_onset_one, 0)
    piano_lh_note_zero = note_id(piano_lh_onset_zero, 0)
    piano_lh_note_one = note_id(piano_lh_onset_one, 0)

    return MotifMlIrDocument(
        metadata=IrDocumentMetadata(
            ir_schema_version="1.0.0",
            corpus_build_version="fixture-corpus-v1",
            generator_version="tools/regenerate_ir_fixture_corpus.py",
            source_document_hash="fixture:ensemble_polyphony_controls",
        ),
        parts=(
            Part(
                part_id=clarinet_part,
                instrument_family=5,
                instrument_kind=502,
                role=1,
                transposition=Transposition(chromatic=2),
                staff_ids=(clarinet_staff,),
            ),
            Part(
                part_id=piano_part,
                instrument_family=8,
                instrument_kind=801,
                role=2,
                transposition=Transposition(),
                staff_ids=(piano_rh_staff, piano_lh_staff),
            ),
        ),
        staves=(
            Staff(
                staff_id=clarinet_staff,
                part_id=clarinet_part,
                staff_index=0,
            ),
            Staff(
                staff_id=piano_rh_staff,
                part_id=piano_part,
                staff_index=0,
            ),
            Staff(
                staff_id=piano_lh_staff,
                part_id=piano_part,
                staff_index=1,
            ),
        ),
        bars=(
            Bar(
                bar_id=bar_zero,
                bar_index=0,
                start=ScoreTime(0, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            ),
            Bar(
                bar_id=bar_one,
                bar_index=1,
                start=ScoreTime(1, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            ),
        ),
        voice_lanes=(
            VoiceLane(
                voice_lane_id=clarinet_lane_zero,
                voice_lane_chain_id=clarinet_chain,
                part_id=clarinet_part,
                staff_id=clarinet_staff,
                bar_id=bar_zero,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=clarinet_lane_one,
                voice_lane_chain_id=clarinet_chain,
                part_id=clarinet_part,
                staff_id=clarinet_staff,
                bar_id=bar_one,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=piano_rh_lane_zero,
                voice_lane_chain_id=piano_rh_chain,
                part_id=piano_part,
                staff_id=piano_rh_staff,
                bar_id=bar_zero,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=piano_rh_lane_one,
                voice_lane_chain_id=piano_rh_chain,
                part_id=piano_part,
                staff_id=piano_rh_staff,
                bar_id=bar_one,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=piano_lh_lane_zero,
                voice_lane_chain_id=piano_lh_chain,
                part_id=piano_part,
                staff_id=piano_lh_staff,
                bar_id=bar_zero,
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=piano_lh_lane_one,
                voice_lane_chain_id=piano_lh_chain,
                part_id=piano_part,
                staff_id=piano_lh_staff,
                bar_id=bar_one,
                voice_index=0,
            ),
        ),
        point_control_events=(
            PointControlEvent(
                control_id=point_control_id("score", 0),
                kind=PointControlKind.TEMPO_CHANGE,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(0, 1),
                value=TempoChangeValue(beats_per_minute=120.0),
            ),
            PointControlEvent(
                control_id=point_control_id("score", 1),
                kind=PointControlKind.TEMPO_CHANGE,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(1, 1),
                value=TempoChangeValue(beats_per_minute=132.0),
            ),
            PointControlEvent(
                control_id=point_control_id("part", 0),
                kind=PointControlKind.DYNAMIC_CHANGE,
                scope=ControlScope.PART,
                target_ref=clarinet_part,
                time=ScoreTime(0, 1),
                value=DynamicChangeValue(marking="mp"),
            ),
            PointControlEvent(
                control_id=point_control_id("score", 2),
                kind=PointControlKind.FERMATA,
                scope=ControlScope.SCORE,
                target_ref="score",
                time=ScoreTime(2, 1),
                value=FermataValue(fermata_type="normal", length_scale=1.5),
            ),
        ),
        span_control_events=(
            SpanControlEvent(
                control_id=span_control_id("part", 0),
                kind=SpanControlKind.HAIRPIN,
                scope=ControlScope.PART,
                target_ref=clarinet_part,
                start_time=ScoreTime(1, 4),
                end_time=ScoreTime(1, 1),
                value=HairpinValue(direction=HairpinDirection.CRESCENDO),
            ),
            SpanControlEvent(
                control_id=span_control_id("staff", 0),
                kind=SpanControlKind.OTTAVA,
                scope=ControlScope.STAFF,
                target_ref=piano_rh_staff,
                start_time=ScoreTime(1, 1),
                end_time=ScoreTime(2, 1),
                value=OttavaValue(octave_shift=1),
            ),
        ),
        onset_groups=(
            OnsetGroup(
                onset_id=clarinet_onset_zero,
                voice_lane_id=clarinet_lane_zero,
                bar_id=bar_zero,
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 2),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 2),
            ),
            OnsetGroup(
                onset_id=clarinet_onset_one,
                voice_lane_id=clarinet_lane_one,
                bar_id=bar_one,
                time=ScoreTime(1, 1),
                duration_notated=ScoreTime(1, 1),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 1),
            ),
            OnsetGroup(
                onset_id=piano_rh_onset_zero,
                voice_lane_id=piano_rh_lane_zero,
                bar_id=bar_zero,
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 2),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 2),
            ),
            OnsetGroup(
                onset_id=piano_rh_onset_one,
                voice_lane_id=piano_rh_lane_one,
                bar_id=bar_one,
                time=ScoreTime(1, 1),
                duration_notated=ScoreTime(1, 1),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 1),
            ),
            OnsetGroup(
                onset_id=piano_lh_onset_zero,
                voice_lane_id=piano_lh_lane_zero,
                bar_id=bar_zero,
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 1),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 1),
            ),
            OnsetGroup(
                onset_id=piano_lh_onset_one,
                voice_lane_id=piano_lh_lane_one,
                bar_id=bar_one,
                time=ScoreTime(1, 1),
                duration_notated=ScoreTime(1, 1),
                is_rest=False,
                attack_order_in_voice=0,
                duration_sounding_max=ScoreTime(1, 1),
            ),
        ),
        note_events=(
            NoteEvent(
                note_id=clarinet_note_zero,
                onset_id=clarinet_onset_zero,
                part_id=clarinet_part,
                staff_id=clarinet_staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 2),
                sounding_duration=ScoreTime(1, 2),
                pitch=Pitch(step=PitchStep.D, octave=5),
                velocity=76,
            ),
            NoteEvent(
                note_id=clarinet_note_one,
                onset_id=clarinet_onset_one,
                part_id=clarinet_part,
                staff_id=clarinet_staff,
                time=ScoreTime(1, 1),
                attack_duration=ScoreTime(1, 1),
                sounding_duration=ScoreTime(1, 1),
                pitch=Pitch(step=PitchStep.G, octave=5),
                velocity=82,
            ),
            NoteEvent(
                note_id=piano_rh_note_zero,
                onset_id=piano_rh_onset_zero,
                part_id=piano_part,
                staff_id=piano_rh_staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 2),
                sounding_duration=ScoreTime(1, 2),
                pitch=Pitch(step=PitchStep.C, octave=5),
                velocity=72,
            ),
            NoteEvent(
                note_id=piano_rh_note_one,
                onset_id=piano_rh_onset_zero,
                part_id=piano_part,
                staff_id=piano_rh_staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 2),
                sounding_duration=ScoreTime(1, 2),
                pitch=Pitch(step=PitchStep.E, octave=5),
                velocity=72,
            ),
            NoteEvent(
                note_id=piano_rh_note_two,
                onset_id=piano_rh_onset_one,
                part_id=piano_part,
                staff_id=piano_rh_staff,
                time=ScoreTime(1, 1),
                attack_duration=ScoreTime(1, 1),
                sounding_duration=ScoreTime(1, 1),
                pitch=Pitch(step=PitchStep.D, octave=6),
                velocity=78,
            ),
            NoteEvent(
                note_id=piano_lh_note_zero,
                onset_id=piano_lh_onset_zero,
                part_id=piano_part,
                staff_id=piano_lh_staff,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 1),
                sounding_duration=ScoreTime(1, 1),
                pitch=Pitch(step=PitchStep.C, octave=3),
                velocity=68,
            ),
            NoteEvent(
                note_id=piano_lh_note_one,
                onset_id=piano_lh_onset_one,
                part_id=piano_part,
                staff_id=piano_lh_staff,
                time=ScoreTime(1, 1),
                attack_duration=ScoreTime(1, 1),
                sounding_duration=ScoreTime(1, 1),
                pitch=Pitch(step=PitchStep.F, octave=2),
                velocity=68,
            ),
        ),
        edges=_contains_edges(
            (clarinet_part, clarinet_staff),
            (piano_part, piano_rh_staff),
            (piano_part, piano_lh_staff),
            (bar_zero, clarinet_lane_zero),
            (bar_one, clarinet_lane_one),
            (bar_zero, piano_rh_lane_zero),
            (bar_one, piano_rh_lane_one),
            (bar_zero, piano_lh_lane_zero),
            (bar_one, piano_lh_lane_one),
            (clarinet_lane_zero, clarinet_onset_zero),
            (clarinet_lane_one, clarinet_onset_one),
            (piano_rh_lane_zero, piano_rh_onset_zero),
            (piano_rh_lane_one, piano_rh_onset_one),
            (piano_lh_lane_zero, piano_lh_onset_zero),
            (piano_lh_lane_one, piano_lh_onset_one),
            (clarinet_onset_zero, clarinet_note_zero),
            (clarinet_onset_one, clarinet_note_one),
            (piano_rh_onset_zero, piano_rh_note_zero),
            (piano_rh_onset_zero, piano_rh_note_one),
            (piano_rh_onset_one, piano_rh_note_two),
            (piano_lh_onset_zero, piano_lh_note_zero),
            (piano_lh_onset_one, piano_lh_note_one),
        )
        + (
            Edge(
                source_id=clarinet_onset_zero,
                target_id=clarinet_onset_one,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
            Edge(
                source_id=piano_rh_onset_zero,
                target_id=piano_rh_onset_one,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
            Edge(
                source_id=piano_lh_onset_zero,
                target_id=piano_lh_onset_one,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
        ),
    )


def _contains_edges(*pairs: tuple[str, str]) -> tuple[Edge, ...]:
    return tuple(
        Edge(source_id=source, target_id=target, edge_type=EdgeType.CONTAINS)
        for source, target in pairs
    )


def _remove_managed_json_files(directory: Path) -> None:
    for json_path in directory.glob("*.json"):
        json_path.unlink()


def _raw_score_time(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _raw_transposition(chromatic: int = 0, octave: int = 0) -> dict[str, int]:
    return {
        "chromatic": chromatic,
        "octave": octave,
        "writtenMinusSoundingSemitones": chromatic + (octave * 12),
    }


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


if __name__ == "__main__":
    main()
