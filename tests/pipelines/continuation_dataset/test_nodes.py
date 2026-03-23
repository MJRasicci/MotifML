"""Focused tests for the continuation dataset extraction surface."""

from __future__ import annotations

from typing import Any

import pytest

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
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
    Edge,
    EdgeType,
    IrDocumentMetadata,
    MotifMlIrDocument,
    NoteEvent,
    OnsetGroup,
    Part,
    Pitch,
    PitchStep,
    Staff,
    TimeSignature,
    Transposition,
    VoiceLane,
)
from motifml.ir.time import ScoreTime
from motifml.pipelines.continuation_dataset.models import ContinuationDatasetParameters
from motifml.pipelines.continuation_dataset.nodes import extract_continuation_examples
from motifml.training.contracts import DatasetSplit, SplitManifestEntry

PROMPT_BAR_COUNT = 4
EXPECTED_TARGET_BAR_INDEX = 4
NODE_OUTPUT_COUNT = 2


def test_extract_continuation_examples_emits_one_eligible_next_bar_example() -> None:
    record = _build_document("collections/eligible_song.json", bar_count=5)
    manifest_entry = _build_split_entry(record.relative_path)

    examples, report = _extract_examples_and_report([record], [manifest_entry])

    assert report is not None
    assert len(examples) >= 1

    example = examples[0]

    assert _prompt_bar_count(example) == PROMPT_BAR_COUNT
    assert _target_scaffold(example) is not None
    assert _target_output(example) is not None
    assert _target_scaffold(example) != _target_output(example)
    assert _example_relative_path(example) == record.relative_path


@pytest.mark.parametrize(
    ("builder_name", "expected_reason_fragment"),
    [
        ("multi_part", "multi-part"),
        ("multi_voice", "multi-voice"),
    ],
)
def test_extract_continuation_examples_reports_structural_rejections(
    builder_name: str,
    expected_reason_fragment: str,
) -> None:
    builder = {
        "multi_part": _build_multi_part_document,
        "multi_voice": _build_multi_voice_document,
    }[builder_name]
    record = builder("collections/rejected_song.json")
    manifest_entry = _build_split_entry(record.relative_path)

    examples, report = _extract_examples_and_report([record], [manifest_entry])
    rejections = _report_rejections(report)

    assert examples == ()
    assert len(rejections) == 1
    assert _rejection_relative_path(rejections[0]) == record.relative_path
    assert expected_reason_fragment in _rejection_reason(rejections[0]).casefold()


def test_extract_continuation_examples_is_deterministic_and_sorted() -> None:
    alpha_record = _build_document("alpha/song.json", bar_count=6)
    zeta_record = _build_document("zeta/song.json", bar_count=6)
    alpha_manifest = _build_split_entry(alpha_record.relative_path)
    zeta_manifest = _build_split_entry(zeta_record.relative_path)

    first_examples, first_report = _extract_examples_and_report(
        [zeta_record, alpha_record],
        [zeta_manifest, alpha_manifest],
    )
    second_examples, second_report = _extract_examples_and_report(
        [alpha_record, zeta_record],
        [alpha_manifest, zeta_manifest],
    )

    assert first_report == second_report
    assert first_examples == second_examples

    first_trace_ids = [_trace_id(example) for example in first_examples]
    second_trace_ids = [_trace_id(example) for example in second_examples]

    assert first_trace_ids == second_trace_ids
    assert len(first_trace_ids) == len(set(first_trace_ids))
    assert [
        (_example_relative_path(example), _target_bar_index(example))
        for example in first_examples
    ] == [
        ("alpha/song.json", EXPECTED_TARGET_BAR_INDEX),
        ("alpha/song.json", EXPECTED_TARGET_BAR_INDEX + 1),
        ("zeta/song.json", EXPECTED_TARGET_BAR_INDEX),
        ("zeta/song.json", EXPECTED_TARGET_BAR_INDEX + 1),
    ]


def _extract_examples_and_report(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    split_manifest: list[SplitManifestEntry],
) -> tuple[tuple[Any, ...], Any]:
    result = extract_continuation_examples(
        normalized_ir_corpus,
        split_manifest,
        ContinuationDatasetParameters(),
    )

    if isinstance(result, tuple) and len(result) == NODE_OUTPUT_COUNT:
        return _coerce_examples(result[0]), result[1]

    if hasattr(result, "records") and hasattr(result, "report"):
        return _coerce_examples(getattr(result, "records")), getattr(result, "report")

    if hasattr(result, "examples") and hasattr(result, "report"):
        return _coerce_examples(getattr(result, "examples")), getattr(result, "report")

    raise AssertionError("extract_continuation_examples returned an unsupported shape.")


def _coerce_examples(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    if hasattr(value, "records"):
        return tuple(getattr(value, "records"))

    if hasattr(value, "examples"):
        return tuple(getattr(value, "examples"))

    raise AssertionError("example collection did not expose a stable sequence.")


def _build_split_entry(
    relative_path: str,
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> SplitManifestEntry:
    return SplitManifestEntry(
        document_id=relative_path,
        relative_path=relative_path,
        split=split,
        group_key=relative_path,
        split_version="split-v1",
    )


def _build_document(
    relative_path: str,
    *,
    bar_count: int,
    part_count: int = 1,
    voices_per_part: int = 1,
) -> MotifIrDocumentRecord:
    parts: list[Part] = []
    staves: list[Staff] = []
    bars: list[Bar] = []
    voice_lanes: list[VoiceLane] = []
    onset_groups: list[OnsetGroup] = []
    note_events: list[NoteEvent] = []
    edges: list[Edge] = []

    for bar_index in range(bar_count):
        bars.append(
            Bar(
                bar_id=bar_id(bar_index),
                bar_index=bar_index,
                start=ScoreTime(bar_index, 1),
                duration=ScoreTime(1, 1),
                time_signature=TimeSignature(4, 4),
            )
        )

    for part_index in range(part_count):
        track_name = f"track-{part_index}"
        owning_part_id = part_id(track_name)
        owning_staff_id = staff_id(owning_part_id, 0)
        parts.append(
            Part(
                part_id=owning_part_id,
                instrument_family=1 + part_index,
                instrument_kind=10 + part_index,
                role=part_index,
                transposition=Transposition(),
                staff_ids=(owning_staff_id,),
            )
        )
        staves.append(
            Staff(
                staff_id=owning_staff_id,
                part_id=owning_part_id,
                staff_index=0,
            )
        )

        for voice_index in range(voices_per_part):
            voice_chain_id = voice_lane_chain_id(
                owning_part_id,
                owning_staff_id,
                voice_index,
            )
            previous_onset_id: str | None = None

            for bar_index in range(bar_count):
                lane_id = voice_lane_id(owning_staff_id, bar_index, voice_index)
                voice_lanes.append(
                    VoiceLane(
                        voice_lane_id=lane_id,
                        voice_lane_chain_id=voice_chain_id,
                        part_id=owning_part_id,
                        staff_id=owning_staff_id,
                        bar_id=bar_id(bar_index),
                        voice_index=voice_index,
                    )
                )

                onset_identifier = onset_id(lane_id, 0)
                onset_groups.append(
                    OnsetGroup(
                        onset_id=onset_identifier,
                        voice_lane_id=lane_id,
                        bar_id=bar_id(bar_index),
                        time=ScoreTime(0, 1),
                        duration_notated=ScoreTime(1, 4),
                        is_rest=False,
                        attack_order_in_voice=0,
                        duration_sounding_max=ScoreTime(1, 4),
                    )
                )
                note_identifier = note_id(onset_identifier, 0)
                note_events.append(
                    NoteEvent(
                        note_id=note_identifier,
                        onset_id=onset_identifier,
                        part_id=owning_part_id,
                        staff_id=owning_staff_id,
                        time=ScoreTime(0, 1),
                        attack_duration=ScoreTime(1, 4),
                        sounding_duration=ScoreTime(1, 4),
                        pitch=Pitch(
                            step=_pitch_step_for(bar_index + voice_index),
                            octave=4 + part_index,
                        ),
                    )
                )

                edges.append(
                    Edge(
                        source_id=bar_id(bar_index),
                        target_id=lane_id,
                        edge_type=EdgeType.CONTAINS,
                    )
                )
                edges.append(
                    Edge(
                        source_id=lane_id,
                        target_id=onset_identifier,
                        edge_type=EdgeType.CONTAINS,
                    )
                )
                edges.append(
                    Edge(
                        source_id=onset_identifier,
                        target_id=note_identifier,
                        edge_type=EdgeType.CONTAINS,
                    )
                )
                if previous_onset_id is not None:
                    edges.append(
                        Edge(
                            source_id=previous_onset_id,
                            target_id=onset_identifier,
                            edge_type=EdgeType.NEXT_IN_VOICE,
                        )
                    )

                previous_onset_id = onset_identifier

        edges.append(
            Edge(
                source_id=owning_part_id,
                target_id=owning_staff_id,
                edge_type=EdgeType.CONTAINS,
            )
        )

    document = MotifMlIrDocument(
        metadata=IrDocumentMetadata(
            ir_schema_version="1.0.0",
            corpus_build_version="normalized-v1",
            generator_version="tests",
            source_document_hash=f"hash:{relative_path}",
        ),
        parts=tuple(parts),
        staves=tuple(staves),
        bars=tuple(bars),
        voice_lanes=tuple(voice_lanes),
        onset_groups=tuple(onset_groups),
        note_events=tuple(note_events),
        edges=tuple(edges),
    )
    return MotifIrDocumentRecord(relative_path=relative_path, document=document)


def _build_multi_part_document(relative_path: str) -> MotifIrDocumentRecord:
    return _build_document(relative_path, bar_count=5, part_count=2)


def _build_multi_voice_document(relative_path: str) -> MotifIrDocumentRecord:
    return _build_document(relative_path, bar_count=5, voices_per_part=2)


def _prompt_bar_count(example: Any) -> int:
    prompt_bars = _resolve_candidate_collection(
        example,
        ("prompt_bars", "prompt", "prompt_context"),
    )

    if hasattr(prompt_bars, "bars"):
        return len(tuple(getattr(prompt_bars, "bars")))

    return len(tuple(prompt_bars))


def _target_scaffold(example: Any) -> Any:
    return _resolve_candidate_value(example, ("target_scaffold", "scaffold"))


def _target_output(example: Any) -> Any:
    return _resolve_candidate_value(example, ("target_output", "target_fill", "target"))


def _example_relative_path(example: Any) -> str:
    return str(
        _resolve_candidate_value(
            example,
            ("source_relative_path", "document_relative_path", "relative_path"),
        )
    )


def _target_bar_index(example: Any) -> int:
    direct = _resolve_optional_candidate_value(
        example,
        ("target_bar_index", "bar_index"),
    )
    if direct is not None:
        return int(direct)

    nested = _resolve_optional_candidate_value(example, ("target_bar",))
    if nested is not None and hasattr(nested, "bar_index"):
        return int(getattr(nested, "bar_index"))

    raise AssertionError("example did not expose a stable target bar index.")


def _trace_id(example: Any) -> str:
    value = _resolve_candidate_value(
        example,
        ("trace_id", "trace_identifier", "trace_key"),
    )
    return str(value)


def _report_rejections(report: Any) -> tuple[Any, ...]:
    if isinstance(report, tuple):
        return report

    if isinstance(report, list):
        return tuple(report)

    return _coerce_examples(
        _resolve_candidate_value(report, ("rejections", "rejected_documents"))
    )


def _rejection_relative_path(rejection: Any) -> str:
    return str(
        _resolve_candidate_value(
            rejection,
            ("relative_path", "source_relative_path", "document_relative_path"),
        )
    )


def _rejection_reason(rejection: Any) -> str:
    return str(
        _resolve_candidate_value(
            rejection,
            ("rejection_reason", "reason", "message"),
        )
    )


def _resolve_candidate_collection(example: Any, names: tuple[str, ...]) -> Any:
    value = _resolve_optional_candidate_value(example, names)
    if value is None:
        raise AssertionError(f"example did not expose any of {names!r}.")
    return value


def _resolve_candidate_value(example: Any, names: tuple[str, ...]) -> Any:
    value = _resolve_optional_candidate_value(example, names)
    if value is None:
        raise AssertionError(f"object did not expose any of {names!r}.")
    return value


def _resolve_optional_candidate_value(example: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(example, name):
            return getattr(example, name)
    return None


def _pitch_step_for(index: int) -> PitchStep:
    steps = tuple(PitchStep)
    return steps[index % len(steps)]
