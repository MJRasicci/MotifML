"""Tests for typed hierarchical IR projections."""

from __future__ import annotations

from motifml.ir.ids import (
    bar_id,
    note_id,
    onset_id,
    part_id,
    staff_id,
    voice_lane_chain_id,
    voice_lane_id,
)
from motifml.ir.models import (
    Bar,
    ControlScope,
    DynamicChangeValue,
    FermataValue,
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
    TempoChangeValue,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.projections.hierarchical import (
    BarProjection,
    HierarchicalProjection,
    NoteProjection,
    OnsetProjection,
    PartProjection,
    VoiceLaneProjection,
    project_hierarchical,
)
from motifml.ir.time import ScoreTime

EXPECTED_PART_ID = part_id("lead")
EXPECTED_STAFF_ID = staff_id(EXPECTED_PART_ID, 0)
EXPECTED_BAR_IDS = (bar_id(0), bar_id(1))
EXPECTED_TOP_LEVEL_BRANCH_COUNT = 3
EXPECTED_VOICE_LANE_IDS = (
    voice_lane_id(EXPECTED_STAFF_ID, 0, 0),
    voice_lane_id(EXPECTED_STAFF_ID, 1, 0),
)
EXPECTED_CHAIN_ID = voice_lane_chain_id(EXPECTED_PART_ID, EXPECTED_STAFF_ID, 0)
EXPECTED_ONSET_IDS = (
    onset_id(EXPECTED_VOICE_LANE_IDS[0], 0),
    onset_id(EXPECTED_VOICE_LANE_IDS[0], 1),
    onset_id(EXPECTED_VOICE_LANE_IDS[1], 0),
)
EXPECTED_NOTE_IDS = (
    note_id(EXPECTED_ONSET_IDS[0], 0),
    note_id(EXPECTED_ONSET_IDS[1], 0),
    note_id(EXPECTED_ONSET_IDS[2], 0),
)


def test_project_hierarchical_builds_a_complete_containment_tree():
    document = _build_document()
    projection = project_hierarchical(document)

    assert isinstance(projection, HierarchicalProjection)
    assert [branch.part.part_id for branch in projection.parts] == [EXPECTED_PART_ID]
    assert [branch.bar.bar_id for branch in projection.bars] == list(EXPECTED_BAR_IDS)
    assert len(projection.children) == EXPECTED_TOP_LEVEL_BRANCH_COUNT
    assert isinstance(projection.parts[0], PartProjection)
    assert isinstance(projection.bars[0], BarProjection)
    assert projection.parts[0].children[0].staff.staff_id == EXPECTED_STAFF_ID
    assert (
        projection.bars[0].children[0].voice_lane.voice_lane_id
        == EXPECTED_VOICE_LANE_IDS[0]
    )
    assert (
        projection.bars[1].children[0].voice_lane.voice_lane_id
        == EXPECTED_VOICE_LANE_IDS[1]
    )
    assert (
        projection.bars[0].children[0].children[0].onset_group.onset_id
        == EXPECTED_ONSET_IDS[0]
    )
    assert (
        projection.bars[0].children[0].children[1].onset_group.onset_id
        == EXPECTED_ONSET_IDS[1]
    )
    assert (
        projection.bars[1].children[0].children[0].onset_group.onset_id
        == EXPECTED_ONSET_IDS[2]
    )


def test_project_hierarchical_preserves_the_expected_nesting_depth():
    projection = project_hierarchical(_build_document())

    voice_lane_projection = projection.bars[0].children[0]
    onset_projection = voice_lane_projection.children[0]
    note_projection = onset_projection.children[0]

    assert isinstance(voice_lane_projection, VoiceLaneProjection)
    assert isinstance(onset_projection, OnsetProjection)
    assert isinstance(note_projection, NoteProjection)
    assert note_projection.note_event.note_id == EXPECTED_NOTE_IDS[0]
    assert note_projection.note_event.pitch == Pitch(step=PitchStep.C, octave=4)


def test_project_hierarchical_attaches_controls_at_their_scoped_level():
    projection = project_hierarchical(_build_document())

    assert [
        control.control_id for control in projection.score_controls.point_controls
    ] == [
        "ctrlp:score:0",
    ]
    assert [
        control.control_id for control in projection.score_controls.span_controls
    ] == [
        "ctrls:score:0",
    ]
    assert [
        control.control_id for control in projection.parts[0].controls.point_controls
    ] == [
        "ctrlp:part:0",
    ]
    assert [
        control.control_id for control in projection.parts[0].controls.span_controls
    ] == [
        "ctrls:part:0",
    ]
    assert [
        control.control_id
        for control in projection.parts[0].children[0].controls.point_controls
    ] == [
        "ctrlp:staff:0",
    ]
    assert [
        control.control_id
        for control in projection.parts[0].children[0].controls.span_controls
    ] == [
        "ctrls:staff:0",
    ]
    assert [
        control.control_id
        for control in projection.bars[0].children[0].controls.point_controls
    ] == [
        "ctrlp:voice:0",
    ]
    assert [
        control.control_id
        for control in projection.bars[0].children[0].controls.span_controls
    ] == [
        "ctrls:voice:0",
    ]


def _build_document() -> MotifMlIrDocument:
    score_point_control = PointControlEvent(
        control_id="ctrlp:score:0",
        kind=PointControlKind.TEMPO_CHANGE,
        scope=ControlScope.SCORE,
        target_ref="score",
        time=ScoreTime(0, 1),
        value=TempoChangeValue(beats_per_minute=120.0),
    )
    part_point_control = PointControlEvent(
        control_id="ctrlp:part:0",
        kind=PointControlKind.DYNAMIC_CHANGE,
        scope=ControlScope.PART,
        target_ref=EXPECTED_PART_ID,
        time=ScoreTime(0, 1),
        value=DynamicChangeValue(marking="mf"),
    )
    staff_point_control = PointControlEvent(
        control_id="ctrlp:staff:0",
        kind=PointControlKind.FERMATA,
        scope=ControlScope.STAFF,
        target_ref=EXPECTED_STAFF_ID,
        time=ScoreTime(0, 1),
        value=FermataValue(fermata_type="upright"),
    )
    voice_point_control = PointControlEvent(
        control_id="ctrlp:voice:0",
        kind=PointControlKind.DYNAMIC_CHANGE,
        scope=ControlScope.VOICE,
        target_ref=EXPECTED_VOICE_LANE_IDS[0],
        time=ScoreTime(0, 1),
        value=DynamicChangeValue(marking="p"),
    )
    score_span_control = SpanControlEvent(
        control_id="ctrls:score:0",
        kind=SpanControlKind.HAIRPIN,
        scope=ControlScope.SCORE,
        target_ref="score",
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 1),
        value=HairpinValue(direction=HairpinDirection.CRESCENDO),
    )
    part_span_control = SpanControlEvent(
        control_id="ctrls:part:0",
        kind=SpanControlKind.OTTAVA,
        scope=ControlScope.PART,
        target_ref=EXPECTED_PART_ID,
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 2),
        value=OttavaValue(octave_shift=1),
    )
    staff_span_control = SpanControlEvent(
        control_id="ctrls:staff:0",
        kind=SpanControlKind.HAIRPIN,
        scope=ControlScope.STAFF,
        target_ref=EXPECTED_STAFF_ID,
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 2),
        value=HairpinValue(direction=HairpinDirection.DECRESCENDO),
    )
    voice_span_control = SpanControlEvent(
        control_id="ctrls:voice:0",
        kind=SpanControlKind.OTTAVA,
        scope=ControlScope.VOICE,
        target_ref=EXPECTED_VOICE_LANE_IDS[0],
        start_time=ScoreTime(0, 1),
        end_time=ScoreTime(1, 2),
        value=OttavaValue(octave_shift=-1),
    )

    return MotifMlIrDocument(
        metadata=IrDocumentMetadata(
            ir_schema_version="1.0.0",
            corpus_build_version="build-1",
            generator_version="0.1.0",
            source_document_hash="abc123",
        ),
        parts=(
            Part(
                part_id=EXPECTED_PART_ID,
                instrument_family=1,
                instrument_kind=2,
                role=3,
                transposition=Transposition(),
                staff_ids=(EXPECTED_STAFF_ID,),
            ),
        ),
        staves=(
            Staff(
                staff_id=EXPECTED_STAFF_ID,
                part_id=EXPECTED_PART_ID,
                staff_index=0,
            ),
        ),
        bars=(
            Bar(
                bar_id=EXPECTED_BAR_IDS[1],
                bar_index=1,
                start=ScoreTime(1, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            ),
            Bar(
                bar_id=EXPECTED_BAR_IDS[0],
                bar_index=0,
                start=ScoreTime(0, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            ),
        ),
        voice_lanes=(
            VoiceLane(
                voice_lane_id=EXPECTED_VOICE_LANE_IDS[1],
                voice_lane_chain_id=EXPECTED_CHAIN_ID,
                part_id=EXPECTED_PART_ID,
                staff_id=EXPECTED_STAFF_ID,
                bar_id=EXPECTED_BAR_IDS[1],
                voice_index=0,
            ),
            VoiceLane(
                voice_lane_id=EXPECTED_VOICE_LANE_IDS[0],
                voice_lane_chain_id=EXPECTED_CHAIN_ID,
                part_id=EXPECTED_PART_ID,
                staff_id=EXPECTED_STAFF_ID,
                bar_id=EXPECTED_BAR_IDS[0],
                voice_index=0,
            ),
        ),
        point_control_events=(
            score_point_control,
            part_point_control,
            staff_point_control,
            voice_point_control,
        ),
        span_control_events=(
            score_span_control,
            part_span_control,
            staff_span_control,
            voice_span_control,
        ),
        onset_groups=(
            OnsetGroup(
                onset_id=EXPECTED_ONSET_IDS[2],
                voice_lane_id=EXPECTED_VOICE_LANE_IDS[1],
                bar_id=EXPECTED_BAR_IDS[1],
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=0,
            ),
            OnsetGroup(
                onset_id=EXPECTED_ONSET_IDS[1],
                voice_lane_id=EXPECTED_VOICE_LANE_IDS[0],
                bar_id=EXPECTED_BAR_IDS[0],
                time=ScoreTime(1, 4),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=1,
            ),
            OnsetGroup(
                onset_id=EXPECTED_ONSET_IDS[0],
                voice_lane_id=EXPECTED_VOICE_LANE_IDS[0],
                bar_id=EXPECTED_BAR_IDS[0],
                time=ScoreTime(0, 1),
                duration_notated=ScoreTime(1, 4),
                is_rest=False,
                attack_order_in_voice=0,
            ),
        ),
        note_events=(
            NoteEvent(
                note_id=EXPECTED_NOTE_IDS[2],
                onset_id=EXPECTED_ONSET_IDS[2],
                part_id=EXPECTED_PART_ID,
                staff_id=EXPECTED_STAFF_ID,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step=PitchStep.E, octave=4),
            ),
            NoteEvent(
                note_id=EXPECTED_NOTE_IDS[1],
                onset_id=EXPECTED_ONSET_IDS[1],
                part_id=EXPECTED_PART_ID,
                staff_id=EXPECTED_STAFF_ID,
                time=ScoreTime(1, 4),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step=PitchStep.D, octave=4),
                string_number=2,
            ),
            NoteEvent(
                note_id=EXPECTED_NOTE_IDS[0],
                onset_id=EXPECTED_ONSET_IDS[0],
                part_id=EXPECTED_PART_ID,
                staff_id=EXPECTED_STAFF_ID,
                time=ScoreTime(0, 1),
                attack_duration=ScoreTime(1, 4),
                sounding_duration=ScoreTime(1, 4),
                pitch=Pitch(step=PitchStep.C, octave=4),
                string_number=1,
            ),
        ),
    )
