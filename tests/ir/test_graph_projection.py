"""Tests for typed graph projections over MotifML IR documents."""

from __future__ import annotations

from motifml.ir.ids import (
    bar_id,
    note_id,
    onset_id,
    part_id,
    phrase_id,
    point_control_id,
    span_control_id,
    staff_id,
    voice_lane_chain_id,
    voice_lane_id,
)
from motifml.ir.models import (
    Bar,
    ControlScope,
    DerivedEdge,
    DerivedEdgeSet,
    DerivedEdgeType,
    Edge,
    EdgeType,
    HairpinDirection,
    HairpinValue,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    OptionalOverlays,
    OptionalViews,
    Part,
    PhraseKind,
    PhraseSource,
    PhraseSpan,
    Pitch,
    PitchStep,
    PlaybackInstance,
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
from motifml.ir.projections.graph import (
    GraphEdgeFamily,
    GraphNodeFamily,
    GraphProjectionParameters,
    project_graph,
)
from motifml.ir.time import ScoreTime

PART_ID = part_id("track-1")
STAFF_ID = staff_id(PART_ID, 0)
BAR_0_ID = bar_id(0)
BAR_1_ID = bar_id(1)
VOICE_LANE_0_ID = voice_lane_id(STAFF_ID, 0, 0)
VOICE_LANE_1_ID = voice_lane_id(STAFF_ID, 0, 1)
VOICE_LANE_2_ID = voice_lane_id(STAFF_ID, 1, 0)
VOICE_LANE_0_CHAIN_ID = voice_lane_chain_id(PART_ID, STAFF_ID, 0)
VOICE_LANE_1_CHAIN_ID = voice_lane_chain_id(PART_ID, STAFF_ID, 1)
ONSET_0_ID = onset_id(VOICE_LANE_0_ID, 0)
ONSET_1_ID = onset_id(VOICE_LANE_1_ID, 0)
ONSET_2_ID = onset_id(VOICE_LANE_2_ID, 0)
NOTE_0_ID = note_id(ONSET_0_ID, 0)
NOTE_1_ID = note_id(ONSET_0_ID, 1)
NOTE_2_ID = note_id(ONSET_1_ID, 0)
NOTE_3_ID = note_id(ONSET_2_ID, 0)
POINT_CONTROL_ID = point_control_id("score", 0)
SPAN_CONTROL_ID = span_control_id(f"staff:{PART_ID}:0", 0)
PHRASE_0_ID = phrase_id(PART_ID, 0)
PHRASE_1_ID = phrase_id(PART_ID, 1)
EXPECTED_NODE_COUNT = 20
EXPECTED_INTRINSIC_EDGE_COUNT = 12
EXPECTED_DERIVED_EDGE_COUNT = 2
EXPECTED_TOTAL_EDGE_COUNT = 14
EXPECTED_ONSET_AND_NOTE_PREFIX_COUNT = 12


def test_project_graph_preserves_intrinsic_only_structure():
    projection = project_graph(_build_graph_document())

    assert projection.node_count == EXPECTED_NODE_COUNT
    assert projection.edge_count == EXPECTED_INTRINSIC_EDGE_COUNT
    assert all(edge.family is GraphEdgeFamily.INTRINSIC for edge in projection.edges)
    assert [
        node.family for node in projection.nodes[:EXPECTED_ONSET_AND_NOTE_PREFIX_COUNT]
    ] == [
        GraphNodeFamily.PART,
        GraphNodeFamily.STAFF,
        GraphNodeFamily.BAR,
        GraphNodeFamily.BAR,
        GraphNodeFamily.VOICE_LANE,
        GraphNodeFamily.VOICE_LANE,
        GraphNodeFamily.VOICE_LANE,
        GraphNodeFamily.POINT_CONTROL,
        GraphNodeFamily.SPAN_CONTROL,
        GraphNodeFamily.ONSET_GROUP,
        GraphNodeFamily.ONSET_GROUP,
        GraphNodeFamily.ONSET_GROUP,
    ]
    assert [node.node_id for node in projection.nodes[4:7]] == [
        VOICE_LANE_0_ID,
        VOICE_LANE_1_ID,
        VOICE_LANE_2_ID,
    ]
    assert [node.node_id for node in projection.nodes[12:16]] == [
        NOTE_1_ID,
        NOTE_0_ID,
        NOTE_2_ID,
        NOTE_3_ID,
    ]
    assert projection.edges[0].edge_type is EdgeType.CONTAINS


def test_project_graph_includes_selected_derived_edges_only():
    projection = project_graph(
        _build_graph_document(),
        GraphProjectionParameters(
            derived_edge_types=(DerivedEdgeType.PLAYBACK_NEXT, "repeats")
        ),
    )

    derived_edges = [
        edge for edge in projection.edges if edge.family is GraphEdgeFamily.DERIVED
    ]

    assert len(derived_edges) == EXPECTED_DERIVED_EDGE_COUNT
    assert {edge.edge_type for edge in derived_edges} == {
        DerivedEdgeType.PLAYBACK_NEXT,
        DerivedEdgeType.REPEATS,
    }
    assert [edge.derived_set_name for edge in derived_edges] == ["analysis", "analysis"]
    assert projection.edge_count == EXPECTED_TOTAL_EDGE_COUNT


def test_project_graph_builds_consistent_adjacency_structures():
    projection = project_graph(
        _build_graph_document(),
        GraphProjectionParameters(
            derived_edge_types=(DerivedEdgeType.PLAYBACK_NEXT, DerivedEdgeType.REPEATS)
        ),
    )

    assert projection.adjacency.node_ids == tuple(
        node.node_id for node in projection.nodes
    )
    assert projection.adjacency.edge_index == (
        tuple(edge.source_index for edge in projection.edges),
        tuple(edge.target_index for edge in projection.edges),
    )
    assert len(projection.adjacency.outgoing_by_node) == projection.node_count
    assert len(projection.adjacency.incoming_by_node) == projection.node_count
    assert (
        sum(len(indices) for indices in projection.adjacency.outgoing_by_node)
        == projection.edge_count
    )
    assert (
        sum(len(indices) for indices in projection.adjacency.incoming_by_node)
        == projection.edge_count
    )
    assert projection.node_index_by_id[BAR_1_ID] > projection.node_index_by_id[BAR_0_ID]


def _build_graph_document() -> MotifMlIrDocument:
    metadata = IrDocumentMetadata(
        ir_schema_version="1.0.0",
        corpus_build_version="build-1",
        generator_version="0.1.0",
        source_document_hash="abc123",
    )
    part = Part(
        part_id=PART_ID,
        instrument_family=1,
        instrument_kind=2,
        role=3,
        transposition=Transposition(),
        staff_ids=(STAFF_ID,),
    )
    staff = Staff(staff_id=STAFF_ID, part_id=PART_ID, staff_index=0)
    bar_0 = Bar(
        bar_id=BAR_0_ID,
        bar_index=0,
        start=ScoreTime(0, 1),
        duration=ScoreTime(1, 1),
        time_signature=TimeSignature(4, 4),
    )
    bar_1 = Bar(
        bar_id=BAR_1_ID,
        bar_index=1,
        start=ScoreTime(1, 1),
        duration=ScoreTime(1, 1),
        time_signature=TimeSignature(4, 4),
    )
    voice_lane_0 = VoiceLane(
        voice_lane_id=VOICE_LANE_0_ID,
        voice_lane_chain_id=VOICE_LANE_0_CHAIN_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        bar_id=BAR_0_ID,
        voice_index=0,
    )
    voice_lane_1 = VoiceLane(
        voice_lane_id=VOICE_LANE_1_ID,
        voice_lane_chain_id=VOICE_LANE_1_CHAIN_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        bar_id=BAR_0_ID,
        voice_index=1,
    )
    voice_lane_2 = VoiceLane(
        voice_lane_id=VOICE_LANE_2_ID,
        voice_lane_chain_id=VOICE_LANE_0_CHAIN_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        bar_id=BAR_1_ID,
        voice_index=0,
    )
    onset_0 = OnsetGroup(
        onset_id=ONSET_0_ID,
        voice_lane_id=VOICE_LANE_0_ID,
        bar_id=BAR_0_ID,
        time=ScoreTime(0, 1),
        duration_notated=ScoreTime(1, 4),
        is_rest=False,
        attack_order_in_voice=0,
        techniques=None,
    )
    onset_1 = OnsetGroup(
        onset_id=ONSET_1_ID,
        voice_lane_id=VOICE_LANE_1_ID,
        bar_id=BAR_0_ID,
        time=ScoreTime(1, 4),
        duration_notated=ScoreTime(1, 4),
        is_rest=False,
        attack_order_in_voice=0,
        techniques=None,
    )
    onset_2 = OnsetGroup(
        onset_id=ONSET_2_ID,
        voice_lane_id=VOICE_LANE_2_ID,
        bar_id=BAR_1_ID,
        time=ScoreTime(0, 1),
        duration_notated=ScoreTime(1, 2),
        is_rest=False,
        attack_order_in_voice=0,
        techniques=None,
    )
    note_0 = NoteEvent(
        note_id=NOTE_0_ID,
        onset_id=ONSET_0_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        time=ScoreTime(0, 1),
        attack_duration=ScoreTime(1, 4),
        sounding_duration=ScoreTime(1, 4),
        pitch=Pitch(step=PitchStep.C, octave=4),
        string_number=2,
    )
    note_1 = NoteEvent(
        note_id=NOTE_1_ID,
        onset_id=ONSET_0_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        time=ScoreTime(0, 1),
        attack_duration=ScoreTime(1, 4),
        sounding_duration=ScoreTime(1, 4),
        pitch=Pitch(step=PitchStep.E, octave=4),
        string_number=1,
    )
    note_2 = NoteEvent(
        note_id=NOTE_2_ID,
        onset_id=ONSET_1_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        time=ScoreTime(1, 4),
        attack_duration=ScoreTime(1, 4),
        sounding_duration=ScoreTime(1, 2),
        pitch=Pitch(step=PitchStep.G, octave=4),
    )
    note_3 = NoteEvent(
        note_id=NOTE_3_ID,
        onset_id=ONSET_2_ID,
        part_id=PART_ID,
        staff_id=STAFF_ID,
        time=ScoreTime(0, 1),
        attack_duration=ScoreTime(1, 4),
        sounding_duration=ScoreTime(1, 2),
        pitch=Pitch(step=PitchStep.B, octave=4),
    )
    point_control = PointControlEvent(
        control_id=POINT_CONTROL_ID,
        kind=PointControlKind.TEMPO_CHANGE,
        scope=ControlScope.SCORE,
        target_ref="score",
        time=ScoreTime(0, 1),
        value=TempoChangeValue(beats_per_minute=120.0),
    )
    span_control = SpanControlEvent(
        control_id=SPAN_CONTROL_ID,
        kind=SpanControlKind.HAIRPIN,
        scope=ControlScope.STAFF,
        target_ref=f"staff:{PART_ID}:0",
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 2),
        value=HairpinValue(direction=HairpinDirection.CRESCENDO),
    )
    phrase_span_0 = PhraseSpan(
        phrase_id=PHRASE_0_ID,
        scope_ref=PART_ID,
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 1),
        phrase_kind=PhraseKind.MELODIC,
        source=PhraseSource.MANUAL_ANNOTATION,
        confidence="0.9",
    )
    phrase_span_1 = PhraseSpan(
        phrase_id=PHRASE_1_ID,
        scope_ref=PART_ID,
        start_time=ScoreTime(1, 1),
        end_time=ScoreTime(2, 1),
        phrase_kind=PhraseKind.RIFF,
        source=PhraseSource.DERIVED_RULE_BASED,
        confidence=0.7,
    )
    playback_instance_0 = PlaybackInstance(
        instance_id="playback:0",
        source_ref=NOTE_0_ID,
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 4),
        voice_lane_chain_id=VOICE_LANE_0_CHAIN_ID,
    )
    playback_instance_1 = PlaybackInstance(
        instance_id="playback:1",
        source_ref=NOTE_2_ID,
        start_time=ScoreTime(1, 4),
        end_time=ScoreTime(1, 2),
        voice_lane_chain_id=VOICE_LANE_0_CHAIN_ID,
    )
    derived_edge_set = DerivedEdgeSet(
        name="analysis",
        kind="graph",
        edges=(
            DerivedEdge(
                source_id=playback_instance_0.instance_id,
                target_id=playback_instance_1.instance_id,
                edge_type=DerivedEdgeType.PLAYBACK_NEXT,
            ),
            DerivedEdge(
                source_id=phrase_span_0.phrase_id,
                target_id=phrase_span_1.phrase_id,
                edge_type=DerivedEdgeType.REPEATS,
            ),
            DerivedEdge(
                source_id=phrase_span_1.phrase_id,
                target_id=ONSET_0_ID,
                edge_type=DerivedEdgeType.ALIGNS_WITH,
            ),
        ),
    )

    return MotifMlIrDocument(
        metadata=metadata,
        parts=(part,),
        staves=(staff,),
        bars=(bar_1, bar_0),
        voice_lanes=(voice_lane_1, voice_lane_2, voice_lane_0),
        point_control_events=(point_control,),
        span_control_events=(span_control,),
        onset_groups=(onset_2, onset_1, onset_0),
        note_events=(note_3, note_2, note_1, note_0),
        edges=(
            Edge(source_id=PART_ID, target_id=STAFF_ID, edge_type=EdgeType.CONTAINS),
            Edge(
                source_id=BAR_0_ID,
                target_id=VOICE_LANE_0_ID,
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id=BAR_0_ID,
                target_id=VOICE_LANE_1_ID,
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id=BAR_1_ID,
                target_id=VOICE_LANE_2_ID,
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id=VOICE_LANE_0_ID,
                target_id=ONSET_0_ID,
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id=VOICE_LANE_1_ID,
                target_id=ONSET_1_ID,
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id=VOICE_LANE_2_ID,
                target_id=ONSET_2_ID,
                edge_type=EdgeType.CONTAINS,
            ),
            Edge(
                source_id=ONSET_0_ID, target_id=NOTE_1_ID, edge_type=EdgeType.CONTAINS
            ),
            Edge(
                source_id=ONSET_0_ID, target_id=NOTE_0_ID, edge_type=EdgeType.CONTAINS
            ),
            Edge(
                source_id=ONSET_1_ID, target_id=NOTE_2_ID, edge_type=EdgeType.CONTAINS
            ),
            Edge(
                source_id=ONSET_2_ID, target_id=NOTE_3_ID, edge_type=EdgeType.CONTAINS
            ),
            Edge(
                source_id=ONSET_0_ID,
                target_id=ONSET_2_ID,
                edge_type=EdgeType.NEXT_IN_VOICE,
            ),
        ),
        optional_overlays=OptionalOverlays(phrase_spans=(phrase_span_0, phrase_span_1)),
        optional_views=OptionalViews(
            playback_instances=(playback_instance_0, playback_instance_1),
            derived_edge_sets=(derived_edge_set,),
        ),
    )
