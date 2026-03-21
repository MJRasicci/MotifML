"""Unit tests for the feature extraction pipeline skeleton."""

from __future__ import annotations

import pytest

import motifml.pipelines.feature_extraction.nodes as feature_extraction_nodes
from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.ids import point_control_id, span_control_id
from motifml.ir.models import (
    Bar,
    ControlScope,
    HairpinDirection,
    HairpinValue,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    Part,
    Pitch,
    PointControlEvent,
    PointControlKind,
    SpanControlEvent,
    SpanControlKind,
    Staff,
    TempoChangeValue,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.projections.graph import GraphAdjacency, GraphProjection
from motifml.ir.projections.hierarchical import HierarchicalProjection
from motifml.ir.projections.sequence import (
    NoteSequenceEvent,
    PointControlSequenceEvent,
    SequenceProjection,
    SequenceProjectionMode,
    SpanControlSequenceEvent,
    StructureMarkerKind,
    StructureMarkerSequenceEvent,
)
from motifml.ir.time import ScoreTime
from motifml.pipelines.feature_extraction.models import (
    BASELINE_SEQUENCE_MODE,
    FeatureExtractionParameters,
)
from motifml.pipelines.feature_extraction.nodes import (
    extract_features,
    merge_feature_shards,
)
from motifml.pipelines.normalization.models import NormalizedIrVersionMetadata
from motifml.training.sequence_schema import NotePayloadField, SequenceSchemaContract


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
        _build_normalized_ir_version_metadata(),
        FeatureExtractionParameters(
            projection_type="sequence",
            sequence_mode=BASELINE_SEQUENCE_MODE,
        ),
        SequenceSchemaContract(),
    )

    assert features.parameters.projection_type.value == "sequence"
    assert isinstance(features.records[0].projection, SequenceProjection)
    assert (
        features.records[0].projection.mode
        is SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS
    )
    assert captured["document"] == record.document
    assert (
        captured["mode"]
        is SequenceProjectionMode.NOTES_AND_CONTROLS_AND_STRUCTURE_MARKERS
    )


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
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "graph",
            "derived_edge_families_included": ["playback_next"],
        },
        SequenceSchemaContract(),
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
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "hierarchical",
            "derived_edge_families_included": [],
        },
        SequenceSchemaContract(),
    )

    assert features.parameters.projection_type.value == "hierarchical"
    assert isinstance(features.records[0].projection, HierarchicalProjection)
    assert captured["document"] == record.document


def test_merge_feature_shards_preserves_parameter_contract_and_order() -> None:
    merged = merge_feature_shards(
        [
            {
                "parameters": {"projection_type": "sequence"},
                "records": [
                    {
                        "relative_path": "fixtures/b.json",
                        "projection_type": "sequence",
                        "projection": {"mode": "notes_only", "events": []},
                    }
                ],
            },
            {
                "parameters": {"projection_type": "sequence"},
                "records": [
                    {
                        "relative_path": "fixtures/a.json",
                        "projection_type": "sequence",
                        "projection": {"mode": "notes_only", "events": []},
                    }
                ],
            },
        ]
    )

    assert merged.parameters.projection_type.value == "sequence"
    assert [record.relative_path for record in merged.records] == [
        "fixtures/a.json",
        "fixtures/b.json",
    ]


def test_extract_features_excludes_structure_markers_when_schema_disables_them() -> (
    None
):
    features = extract_features(
        [_build_record()],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        {
            "structure_markers": {"enabled": False},
            "controls": {
                "include_point_controls": False,
                "include_span_controls": False,
            },
        },
    )

    projection = features.records[0].projection

    assert all(
        not isinstance(event, StructureMarkerSequenceEvent)
        for event in projection.events
    )


def test_extract_features_excludes_control_families_when_schema_disables_them() -> None:
    features = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        {
            "controls": {
                "include_point_controls": False,
                "include_span_controls": False,
            }
        },
    )

    projection = features.records[0].projection

    assert all(
        not isinstance(event, PointControlSequenceEvent) for event in projection.events
    )
    assert all(
        not isinstance(event, SpanControlSequenceEvent) for event in projection.events
    )


def test_extract_features_persists_feature_and_schema_versions() -> None:
    schema = SequenceSchemaContract()

    features = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        schema,
    )

    assert features.parameters.normalized_ir_version == "normalized-v1"
    assert features.parameters.sequence_schema_version == schema.sequence_schema_version
    assert features.parameters.feature_version


def test_extract_features_feature_version_is_stable_for_unchanged_inputs() -> None:
    first = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )
    second = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )

    assert first.parameters.feature_version == second.parameters.feature_version


def test_extract_features_feature_version_changes_when_normalized_ir_version_changes() -> (
    None
):
    first = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata("normalized-v1"),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )
    second = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata("normalized-v2"),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )

    assert first.parameters.feature_version != second.parameters.feature_version


def test_extract_features_feature_version_changes_when_sequence_ordering_changes(
    monkeypatch,
) -> None:
    first = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )
    monkeypatch.setattr(
        feature_extraction_nodes,
        "SEQUENCE_EVENT_ORDERING_VERSION",
        "time_then_family_then_entity_v2",
    )

    second = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )

    assert first.parameters.feature_version != second.parameters.feature_version


def test_extract_features_feature_version_changes_when_sequence_schema_changes() -> (
    None
):
    first = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )
    second = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(
            note_payload_fields=(
                NotePayloadField.PITCH,
                NotePayloadField.DURATION,
                NotePayloadField.STRING_NUMBER,
            )
        ),
    )

    assert first.parameters.sequence_schema_version != (
        second.parameters.sequence_schema_version
    )
    assert first.parameters.feature_version != second.parameters.feature_version


def test_extract_features_rejects_out_of_order_sequence_events(monkeypatch) -> None:
    original_project_sequence = feature_extraction_nodes.project_sequence

    def fake_project_sequence(document, config):
        projection = original_project_sequence(document, config)
        return SequenceProjection(
            mode=projection.mode,
            events=tuple(reversed(projection.events)),
        )

    monkeypatch.setattr(
        feature_extraction_nodes,
        "project_sequence",
        fake_project_sequence,
    )

    with pytest.raises(
        ValueError,
        match="Sequence projection contract violation",
    ):
        extract_features(
            [_build_record(include_controls=True)],
            _build_normalized_ir_version_metadata(),
            {
                "projection_type": "sequence",
                "sequence_mode": BASELINE_SEQUENCE_MODE,
            },
            SequenceSchemaContract(),
        )


def test_extract_features_preserves_multi_voice_note_interleaving() -> None:
    features = extract_features(
        [_build_multi_voice_record()],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        {
            "structure_markers": {"enabled": False},
            "controls": {
                "include_point_controls": False,
                "include_span_controls": False,
            },
        },
    )

    projection = features.records[0].projection

    assert all(isinstance(event, NoteSequenceEvent) for event in projection.events)
    assert [event.time for event in projection.events] == [
        ScoreTime(0, 1),
        ScoreTime(1, 8),
        ScoreTime(1, 4),
    ]
    assert [event.voice_lane_id for event in projection.events] == [
        "voice:staff:part:track-b:0:0:0",
        "voice:staff:part:track-b:0:0:1",
        "voice:staff:part:track-b:0:0:0",
    ]


def test_extract_features_places_structure_markers_before_controls_and_notes() -> None:
    features = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )

    projection = features.records[0].projection
    time_zero_events = [
        event for event in projection.events if event.time == ScoreTime(0, 1)
    ]

    assert [type(event) for event in time_zero_events[:5]] == [
        StructureMarkerSequenceEvent,
        StructureMarkerSequenceEvent,
        StructureMarkerSequenceEvent,
        StructureMarkerSequenceEvent,
        StructureMarkerSequenceEvent,
    ]
    assert [event.marker_kind for event in time_zero_events[:5]] == [
        StructureMarkerKind.PART,
        StructureMarkerKind.STAFF,
        StructureMarkerKind.BAR,
        StructureMarkerKind.VOICE_LANE,
        StructureMarkerKind.ONSET_GROUP,
    ]
    assert isinstance(time_zero_events[5], PointControlSequenceEvent)
    assert isinstance(time_zero_events[6], SpanControlSequenceEvent)
    assert isinstance(time_zero_events[7], NoteSequenceEvent)


def test_extract_features_orders_control_events_before_notes() -> None:
    features = extract_features(
        [_build_record(include_controls=True)],
        _build_normalized_ir_version_metadata(),
        {
            "projection_type": "sequence",
            "sequence_mode": BASELINE_SEQUENCE_MODE,
        },
        SequenceSchemaContract(),
    )

    projection = features.records[0].projection
    non_marker_events = [
        event
        for event in projection.events
        if event.time == ScoreTime(0, 1)
        and not isinstance(event, StructureMarkerSequenceEvent)
    ]

    assert [type(event) for event in non_marker_events] == [
        PointControlSequenceEvent,
        SpanControlSequenceEvent,
        NoteSequenceEvent,
    ]


def _build_normalized_ir_version_metadata(
    normalized_ir_version: str = "normalized-v1",
) -> NormalizedIrVersionMetadata:
    return NormalizedIrVersionMetadata(
        normalized_ir_version=normalized_ir_version,
        contract_name="motifml.normalized_ir",
        contract_version="1.0.0",
        serialized_document_format="motifml.ir.document",
        normalization_strategy="passthrough_v1",
        upstream_ir_schema_version="1.0.0",
        task_agnostic_guarantees=(
            "stable_source_relative_identity",
            "task_agnostic_domain_truth",
        ),
    )


def _build_multi_voice_record() -> MotifIrDocumentRecord:
    part_id = "part:track-b"
    staff_id = "staff:part:track-b:0"
    bar_id = "bar:0"
    voice_lane_id_0 = "voice:staff:part:track-b:0:0:0"
    voice_lane_id_1 = "voice:staff:part:track-b:0:0:1"
    onset_id_0 = "onset:voice:staff:part:track-b:0:0:0:0"
    onset_id_1 = "onset:voice:staff:part:track-b:0:0:0:1"
    onset_id_2 = "onset:voice:staff:part:track-b:0:0:1:0"

    return MotifIrDocumentRecord(
        relative_path="fixtures/multi_voice.json",
        document=MotifMlIrDocument(
            metadata=IrDocumentMetadata(
                ir_schema_version="1.0.0",
                corpus_build_version="build-1",
                generator_version="0.1.0",
                source_document_hash="multi-voice",
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
                    voice_lane_id=voice_lane_id_0,
                    voice_lane_chain_id="voice-chain:part:track-b:staff:part:track-b:0:0",
                    part_id=part_id,
                    staff_id=staff_id,
                    bar_id=bar_id,
                    voice_index=0,
                ),
                VoiceLane(
                    voice_lane_id=voice_lane_id_1,
                    voice_lane_chain_id="voice-chain:part:track-b:staff:part:track-b:0:1",
                    part_id=part_id,
                    staff_id=staff_id,
                    bar_id=bar_id,
                    voice_index=1,
                ),
            ),
            onset_groups=(
                OnsetGroup(
                    onset_id=onset_id_0,
                    voice_lane_id=voice_lane_id_0,
                    bar_id=bar_id,
                    time=ScoreTime(0, 1),
                    duration_notated=ScoreTime(1, 4),
                    is_rest=False,
                    attack_order_in_voice=0,
                ),
                OnsetGroup(
                    onset_id=onset_id_1,
                    voice_lane_id=voice_lane_id_0,
                    bar_id=bar_id,
                    time=ScoreTime(1, 4),
                    duration_notated=ScoreTime(1, 4),
                    is_rest=False,
                    attack_order_in_voice=1,
                ),
                OnsetGroup(
                    onset_id=onset_id_2,
                    voice_lane_id=voice_lane_id_1,
                    bar_id=bar_id,
                    time=ScoreTime(1, 8),
                    duration_notated=ScoreTime(1, 8),
                    is_rest=False,
                    attack_order_in_voice=0,
                ),
            ),
            note_events=(
                NoteEvent(
                    note_id="note:onset:voice:staff:part:track-b:0:0:0:0:0",
                    onset_id=onset_id_0,
                    part_id=part_id,
                    staff_id=staff_id,
                    time=ScoreTime(0, 1),
                    attack_duration=ScoreTime(1, 4),
                    sounding_duration=ScoreTime(1, 4),
                    pitch=Pitch(step="C", octave=4),
                ),
                NoteEvent(
                    note_id="note:onset:voice:staff:part:track-b:0:0:1:0:0",
                    onset_id=onset_id_2,
                    part_id=part_id,
                    staff_id=staff_id,
                    time=ScoreTime(1, 8),
                    attack_duration=ScoreTime(1, 8),
                    sounding_duration=ScoreTime(1, 8),
                    pitch=Pitch(step="D", octave=4),
                ),
                NoteEvent(
                    note_id="note:onset:voice:staff:part:track-b:0:0:0:1:0",
                    onset_id=onset_id_1,
                    part_id=part_id,
                    staff_id=staff_id,
                    time=ScoreTime(1, 4),
                    attack_duration=ScoreTime(1, 4),
                    sounding_duration=ScoreTime(1, 4),
                    pitch=Pitch(step="E", octave=4),
                ),
            ),
        ),
    )


def _build_record(include_controls: bool = False) -> MotifIrDocumentRecord:
    part_id = "part:track-a"
    staff_id = "staff:part:track-a:0"
    bar_id = "bar:0"
    voice_lane_id = "voice:staff:part:track-a:0:0:0"
    onset_id = "onset:voice:staff:part:track-a:0:0:0:0"
    note_id = "note:onset:voice:staff:part:track-a:0:0:0:0:0"
    point_control_event_id = point_control_id("score", 0)
    span_control_event_id = span_control_id(staff_id, 0)

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
            point_control_events=(
                (
                    PointControlEvent(
                        control_id=point_control_event_id,
                        kind=PointControlKind.TEMPO_CHANGE,
                        scope=ControlScope.SCORE,
                        target_ref="score",
                        time=ScoreTime(0, 1),
                        value=TempoChangeValue(beats_per_minute=120.0),
                    ),
                )
                if include_controls
                else ()
            ),
            span_control_events=(
                (
                    SpanControlEvent(
                        control_id=span_control_event_id,
                        kind=SpanControlKind.HAIRPIN,
                        scope=ControlScope.STAFF,
                        target_ref=staff_id,
                        start_time=ScoreTime(0, 1),
                        end_time=ScoreTime(1, 4),
                        value=HairpinValue(direction=HairpinDirection.CRESCENDO),
                    ),
                )
                if include_controls
                else ()
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
