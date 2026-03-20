"""Unit tests for the feature extraction pipeline skeleton."""

from __future__ import annotations

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.models import (
    Bar,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    Part,
    Pitch,
    Staff,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.projections.graph import GraphAdjacency, GraphProjection
from motifml.ir.projections.hierarchical import HierarchicalProjection
from motifml.ir.projections.sequence import SequenceProjection, SequenceProjectionMode
from motifml.ir.time import ScoreTime
from motifml.pipelines.feature_extraction.models import FeatureExtractionParameters
from motifml.pipelines.feature_extraction.nodes import extract_features


def test_extract_features_uses_the_sequence_projection(monkeypatch) -> None:
    record = _build_record()
    captured: dict[str, object] = {}

    def fake_project_sequence(document, config):
        captured["document"] = document
        captured["mode"] = config.mode
        return SequenceProjection(mode=config.mode, events=())

    monkeypatch.setattr(
        "motifml.pipelines.feature_extraction.nodes.project_sequence",
        fake_project_sequence,
    )

    features = extract_features(
        [record],
        FeatureExtractionParameters(
            projection_type="sequence",
            event_types_included=("notes", "controls"),
        ),
    )

    assert features.parameters.projection_type.value == "sequence"
    assert isinstance(features.records[0].projection, SequenceProjection)
    assert (
        features.records[0].projection.mode is SequenceProjectionMode.NOTES_AND_CONTROLS
    )
    assert captured["document"] == record.document
    assert captured["mode"] is SequenceProjectionMode.NOTES_AND_CONTROLS


def test_extract_features_uses_the_graph_projection(monkeypatch) -> None:
    record = _build_record()
    captured: dict[str, object] = {}

    def fake_project_graph(document, parameters):
        captured["document"] = document
        captured["derived_edge_types"] = parameters.derived_edge_types
        return GraphProjection(
            nodes=(),
            edges=(),
            adjacency=GraphAdjacency(
                node_ids=(),
                edge_index=((), ()),
                edge_types=(),
                outgoing_by_node=(),
                incoming_by_node=(),
            ),
        )

    monkeypatch.setattr(
        "motifml.pipelines.feature_extraction.nodes.project_graph",
        fake_project_graph,
    )

    features = extract_features(
        [record],
        {
            "projection_type": "graph",
            "event_types_included": ["notes"],
            "derived_edge_families_included": ["playback_next"],
        },
    )

    assert features.parameters.projection_type.value == "graph"
    assert isinstance(features.records[0].projection, GraphProjection)
    assert captured["document"] == record.document
    assert captured["derived_edge_types"] == ("playback_next",)


def test_extract_features_uses_the_hierarchical_projection(monkeypatch) -> None:
    record = _build_record()
    captured: dict[str, object] = {}

    def fake_project_hierarchical(document):
        captured["document"] = document
        return HierarchicalProjection()

    monkeypatch.setattr(
        "motifml.pipelines.feature_extraction.nodes.project_hierarchical",
        fake_project_hierarchical,
    )

    features = extract_features(
        [record],
        {
            "projection_type": "hierarchical",
            "event_types_included": ["notes"],
            "derived_edge_families_included": [],
        },
    )

    assert features.parameters.projection_type.value == "hierarchical"
    assert isinstance(features.records[0].projection, HierarchicalProjection)
    assert captured["document"] == record.document


def _build_record() -> MotifIrDocumentRecord:
    part_id = "part:track-a"
    staff_id = "staff:part:track-a:0"
    bar_id = "bar:0"
    voice_lane_id = "voice:staff:part:track-a:0:0:0"
    onset_id = "onset:voice:staff:part:track-a:0:0:0:0"
    note_id = "note:onset:voice:staff:part:track-a:0:0:0:0:0"

    return MotifIrDocumentRecord(
        relative_path="fixtures/example.json",
        document=MotifMlIrDocument(
            metadata=IrDocumentMetadata(
                ir_schema_version="1.0.0",
                corpus_build_version="build-1",
                generator_version="0.1.0",
                source_document_hash="abc123",
            ),
            parts=(
                Part(
                    part_id=part_id,
                    instrument_family=1,
                    instrument_kind=2,
                    role=3,
                    transposition=Transposition(),
                    staff_ids=(staff_id,),
                ),
            ),
            staves=(Staff(staff_id=staff_id, part_id=part_id, staff_index=0),),
            bars=(
                Bar(
                    bar_id=bar_id,
                    bar_index=0,
                    start=ScoreTime(0, 1),
                    duration=ScoreTime(1, 1),
                    time_signature=TimeSignature(4, 4),
                ),
            ),
            voice_lanes=(
                VoiceLane(
                    voice_lane_id=voice_lane_id,
                    voice_lane_chain_id="voice-chain:part:track-a:staff:part:track-a:0:0",
                    part_id=part_id,
                    staff_id=staff_id,
                    bar_id=bar_id,
                    voice_index=0,
                ),
            ),
            onset_groups=(
                OnsetGroup(
                    onset_id=onset_id,
                    voice_lane_id=voice_lane_id,
                    bar_id=bar_id,
                    time=ScoreTime(0, 1),
                    duration_notated=ScoreTime(1, 4),
                    is_rest=False,
                    attack_order_in_voice=0,
                ),
            ),
            note_events=(
                NoteEvent(
                    note_id=note_id,
                    onset_id=onset_id,
                    part_id=part_id,
                    staff_id=staff_id,
                    time=ScoreTime(0, 1),
                    attack_duration=ScoreTime(1, 4),
                    sounding_duration=ScoreTime(1, 4),
                    pitch=Pitch(step="C", octave=4),
                ),
            ),
        ),
    )
