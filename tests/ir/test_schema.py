"""Contract tests for the persisted IR JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from motifml.ir.models import (
    DerivedEdge,
    DerivedEdgeSet,
    DerivedEdgeType,
    MotifMlIrDocument,
    OptionalOverlays,
    OptionalViews,
    PhraseKind,
    PhraseSource,
    PhraseSpan,
    PlaybackInstance,
)
from motifml.ir.serialization import deserialize_document, serialize_document

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "motifml"
    / "ir"
    / "schema"
    / "motifml-ir-document.schema.json"
)
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "ir"
    / "representative_document.ir.json"
)


def test_ir_schema_is_a_valid_draft_2020_12_schema():
    schema = _load_json(SCHEMA_PATH)

    Draft202012Validator.check_schema(schema)


def test_serialized_ir_documents_validate_against_the_checked_in_schema():
    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)

    fixture_payload = _load_json(FIXTURE_PATH)
    canonical_document = deserialize_document(FIXTURE_PATH.read_text(encoding="utf-8"))
    serialized_payload = json.loads(serialize_document(canonical_document))

    assert list(validator.iter_errors(fixture_payload)) == []
    assert list(validator.iter_errors(serialized_payload)) == []


def test_phrase_overlay_documents_validate_against_the_checked_in_schema():
    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)

    base_document = deserialize_document(FIXTURE_PATH.read_text(encoding="utf-8"))
    overlaid_document = MotifMlIrDocument(
        metadata=base_document.metadata,
        parts=base_document.parts,
        staves=base_document.staves,
        bars=base_document.bars,
        voice_lanes=base_document.voice_lanes,
        point_control_events=base_document.point_control_events,
        span_control_events=base_document.span_control_events,
        onset_groups=base_document.onset_groups,
        note_events=base_document.note_events,
        edges=base_document.edges,
        optional_overlays=OptionalOverlays(
            phrase_spans=(
                PhraseSpan(
                    phrase_id="phrase:part:track-a:0",
                    scope_ref="part:track-a",
                    start_time=base_document.bars[0].start,
                    end_time=base_document.bars[-1].start
                    + base_document.bars[-1].duration,
                    phrase_kind=PhraseKind.MELODIC,
                    source=PhraseSource.MANUAL_ANNOTATION,
                    confidence="reviewed",
                ),
            )
        ),
        optional_views=OptionalViews(
            playback_instances=(
                PlaybackInstance(
                    instance_id="playback:part:track-a:0",
                    source_ref="note:onset:voice:staff:part:track-a:0:0:0:0:0",
                    start_time=base_document.note_events[0].time,
                    end_time=base_document.note_events[0].time
                    + base_document.note_events[0].attack_duration,
                ),
            ),
            derived_edge_sets=(
                DerivedEdgeSet(
                    name="playback-links",
                    kind="traversal",
                    edges=(
                        DerivedEdge(
                            source_id="note:onset:voice:staff:part:track-a:0:0:0:0:0",
                            target_id="note:onset:voice:staff:part:track-a:0:0:0:0:1",
                            edge_type=DerivedEdgeType.PLAYBACK_NEXT,
                        ),
                    ),
                ),
            ),
        ),
    )

    errors = list(validator.iter_errors(_load_payload(overlaid_document)))

    assert errors == []


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_payload(document: MotifMlIrDocument) -> dict[str, object]:
    return json.loads(serialize_document(document))
