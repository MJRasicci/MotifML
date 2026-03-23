"""Nodes for extracting the V1 continuation-example dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.inspection_models import OnsetNoteTable, VoiceLaneOnsetTable
from motifml.ir.inspection_tables import (
    build_onset_note_tables,
    build_voice_lane_onset_tables,
)
from motifml.ir.models import Bar, MotifMlIrDocument
from motifml.ir.time import ScoreTime
from motifml.pipelines.continuation_dataset.models import (
    ContinuationDatasetArtifacts,
    ContinuationDatasetMetadata,
    ContinuationDatasetParameters,
    ContinuationDatasetReport,
    ContinuationDocumentRejection,
    ContinuationExample,
    PromptBar,
    PromptNote,
    PromptOnset,
    TargetBarFill,
    TargetBarScaffold,
    TargetNoteSlot,
    TargetOnsetFill,
    TargetOnsetScaffold,
    coerce_continuation_dataset_parameters,
)
from motifml.pipelines.normalization.models import (
    NormalizedIrVersionMetadata,
    coerce_normalized_ir_version_metadata,
)
from motifml.training.contracts import (
    SplitManifestEntry,
    coerce_split_manifest_entries,
)
from motifml.training.versioning import build_contract_version

_RECORD_SCHEMA_VERSION = "motifml.continuation_dataset.record.v1"
_EXTRACTION_STRATEGY = "normalized_ir_single_voice_next_bar_v1"
_UNVERSIONED_NORMALIZED_IR = "normalized_ir_unversioned"


@dataclass(frozen=True, slots=True)
class _ContinuationDocumentContext:
    source_relative_path: str
    document_id: str
    split: Any
    split_version: str
    normalized_ir_version_key: str
    continuation_dataset_version: str
    parameters: ContinuationDatasetParameters
    part_id: str
    staff_id: str
    voice_lane_chain_id: str
    onset_tables_by_bar_id: Mapping[str, VoiceLaneOnsetTable]
    onset_note_tables_by_onset_id: Mapping[str, OnsetNoteTable]


def extract_continuation_examples(
    normalized_ir_corpus: list[MotifIrDocumentRecord],
    split_manifest: tuple[SplitManifestEntry, ...] | list[Mapping[str, Any]],
    parameters: ContinuationDatasetParameters | Mapping[str, Any],
    normalized_ir_version: NormalizedIrVersionMetadata
    | Mapping[str, Any]
    | None = None,
) -> tuple[ContinuationDatasetArtifacts, ContinuationDatasetReport]:
    """Extract deterministic V1 prompt/scaffold/target continuation examples."""
    typed_parameters = coerce_continuation_dataset_parameters(parameters)
    manifest_entries = coerce_split_manifest_entries(split_manifest)
    if not manifest_entries:
        raise ValueError("split_manifest must contain at least one entry.")

    split_versions = {entry.split_version for entry in manifest_entries}
    if len(split_versions) != 1:
        raise ValueError("split_manifest entries must share one split_version.")
    split_version = manifest_entries[0].split_version

    manifest_by_relative_path: dict[str, SplitManifestEntry] = {}
    for entry in manifest_entries:
        if entry.relative_path in manifest_by_relative_path:
            raise ValueError("split_manifest must not repeat relative_path values.")
        manifest_by_relative_path[entry.relative_path] = entry

    normalized_ir_version_key = _normalized_ir_version_key(normalized_ir_version)
    continuation_dataset_version = build_contract_version(
        namespace="continuation_dataset_version",
        payload={
            "normalized_ir_version": normalized_ir_version_key,
            "split_version": split_version,
            "task_contract": typed_parameters.to_version_payload(),
            "record_schema_version": _RECORD_SCHEMA_VERSION,
            "extraction_strategy": _EXTRACTION_STRATEGY,
        },
    )

    examples: list[ContinuationExample] = []
    rejections: list[ContinuationDocumentRejection] = []
    split_example_counts: Counter[str] = Counter()
    used_relative_paths: set[str] = set()

    for record in sorted(
        normalized_ir_corpus,
        key=lambda item: item.relative_path.casefold(),
    ):
        manifest_entry = manifest_by_relative_path.get(record.relative_path)
        if manifest_entry is None:
            raise ValueError(
                "split_manifest is missing an entry for normalized IR record "
                f"'{record.relative_path}'."
            )

        document_examples, rejection = _extract_document_examples(
            record,
            manifest_entry,
            typed_parameters,
            normalized_ir_version_key=normalized_ir_version_key,
            continuation_dataset_version=continuation_dataset_version,
        )
        if rejection is not None:
            rejections.append(rejection)
            continue

        if document_examples:
            used_relative_paths.add(record.relative_path)
            split_example_counts[manifest_entry.split.value] += len(document_examples)
            examples.extend(document_examples)

    metadata = ContinuationDatasetMetadata(
        continuation_dataset_version=continuation_dataset_version,
        normalized_ir_version=normalized_ir_version_key,
        split_version=split_version,
        task_contract_name=typed_parameters.contract_name,
        task_contract_version=typed_parameters.contract_version,
        prompt_bar_count=typed_parameters.prompt_bar_count,
    )
    report = ContinuationDatasetReport(
        continuation_dataset_version=continuation_dataset_version,
        normalized_ir_version=normalized_ir_version_key,
        split_version=split_version,
        source_document_count=len(normalized_ir_corpus),
        source_document_used_count=len(used_relative_paths),
        emitted_example_count=len(examples),
        split_example_counts=dict(split_example_counts),
        rejection_counts_by_reason=dict(
            Counter(rejection.rejection_reason for rejection in rejections)
        ),
        rejections=tuple(rejections),
    )
    artifacts = ContinuationDatasetArtifacts(
        parameters=metadata,
        records=tuple(examples),
    )
    return artifacts, report


def _extract_document_examples(
    record: MotifIrDocumentRecord,
    manifest_entry: SplitManifestEntry,
    parameters: ContinuationDatasetParameters,
    *,
    normalized_ir_version_key: str,
    continuation_dataset_version: str,
) -> tuple[tuple[ContinuationExample, ...], ContinuationDocumentRejection | None]:
    voice_lane_tables = build_voice_lane_onset_tables(record.document)
    onset_note_tables = build_onset_note_tables(record.document)
    rejection_reason = _document_rejection_reason(
        record.document,
        voice_lane_tables,
        onset_note_tables,
        parameters,
    )
    if rejection_reason is not None:
        return (), _build_rejection(record, manifest_entry, rejection_reason)

    sorted_bars = _sorted_bars(record.document)
    if len(sorted_bars) <= parameters.prompt_bar_count:
        return (), _build_rejection(
            record,
            manifest_entry,
            f"insufficient bars for {parameters.prompt_bar_count}-bar prompt",
        )

    part_id = record.document.parts[0].part_id
    staff_id = record.document.staves[0].staff_id
    voice_lane_chain_id = record.document.voice_lanes[0].voice_lane_chain_id
    onset_tables_by_bar_id = _onset_tables_by_bar_id(
        voice_lane_tables,
        voice_lane_chain_id=voice_lane_chain_id,
    )
    onset_note_tables_by_onset_id = _onset_note_tables_by_onset_id(
        onset_note_tables,
        voice_lane_chain_id=voice_lane_chain_id,
    )

    context = _ContinuationDocumentContext(
        source_relative_path=record.relative_path,
        document_id=manifest_entry.document_id,
        split=manifest_entry.split,
        split_version=manifest_entry.split_version,
        normalized_ir_version_key=normalized_ir_version_key,
        continuation_dataset_version=continuation_dataset_version,
        parameters=parameters,
        part_id=part_id,
        staff_id=staff_id,
        voice_lane_chain_id=voice_lane_chain_id,
        onset_tables_by_bar_id=onset_tables_by_bar_id,
        onset_note_tables_by_onset_id=onset_note_tables_by_onset_id,
    )
    examples: list[ContinuationExample] = []
    for target_position in range(parameters.prompt_bar_count, len(sorted_bars)):
        prompt_source_bars = tuple(
            sorted_bars[index]
            for index in range(
                target_position - parameters.prompt_bar_count,
                target_position,
            )
        )
        target_bar = sorted_bars[target_position]
        example = _build_continuation_example(
            context=context,
            prompt_source_bars=prompt_source_bars,
            target_bar=target_bar,
        )
        _validate_example_contract(
            example, prompt_bar_count=parameters.prompt_bar_count
        )
        examples.append(example)

    return tuple(examples), None


def _document_rejection_reason(
    document: MotifMlIrDocument,
    voice_lane_tables: Sequence[VoiceLaneOnsetTable],
    onset_note_tables: Sequence[OnsetNoteTable],
    parameters: ContinuationDatasetParameters,
) -> str | None:
    voice_lane_chain_ids = {
        voice_lane.voice_lane_chain_id for voice_lane in document.voice_lanes
    }

    simple_checks = (
        (not document.bars, "document contains no bars"),
        (not document.voice_lanes, "document contains no voice lanes"),
        (
            parameters.require_single_part and len(document.parts) != 1,
            "multi-part document",
        ),
        (
            parameters.require_single_staff and len(document.staves) != 1,
            "multi-staff document",
        ),
        (
            parameters.require_single_voice_lane_chain
            and len(voice_lane_chain_ids) != 1,
            "multi-voice document",
        ),
        (
            parameters.reject_control_events
            and (document.point_control_events or document.span_control_events),
            "control events present",
        ),
        (
            parameters.reject_grace_notes
            and any(
                row.grace_type is not None
                for table in voice_lane_tables
                for row in table.rows
            ),
            "grace notes present",
        ),
        (
            parameters.reject_techniques
            and (
                any(
                    row.technique_summary is not None
                    for table in voice_lane_tables
                    for row in table.rows
                )
                or any(
                    row.technique_summary is not None
                    for table in onset_note_tables
                    for row in table.rows
                )
            ),
            "techniques present",
        ),
        (
            parameters.require_attack_equals_sounding
            and any(
                row.attack_duration != row.sounding_duration
                for table in onset_note_tables
                for row in table.rows
            ),
            "attack and sounding durations differ",
        ),
        (
            parameters.require_pitched_notes
            and any(
                row.pitch_text.casefold() == "unpitched"
                for table in onset_note_tables
                for row in table.rows
            ),
            "unpitched notes present",
        ),
    )
    for condition, reason in simple_checks:
        if condition:
            return reason

    if len(voice_lane_chain_ids) == 1:
        target_chain_id = next(iter(voice_lane_chain_ids))
        chain_bar_ids = {
            voice_lane.bar_id
            for voice_lane in document.voice_lanes
            if voice_lane.voice_lane_chain_id == target_chain_id
        }
        if len(chain_bar_ids) != len(document.bars):
            return "voice lane chain does not span every bar"

    return None


def _build_continuation_example(
    *,
    context: _ContinuationDocumentContext,
    prompt_source_bars: Sequence[Bar],
    target_bar: Bar,
) -> ContinuationExample:
    trace_id = f"{context.source_relative_path}::{context.voice_lane_chain_id}::bar:{target_bar.bar_index}"
    example_id = build_contract_version(
        namespace="continuation_example_id",
        payload={"trace_id": trace_id},
    )
    artifact_relative_path = _artifact_relative_path(
        split=str(context.split),
        source_relative_path=context.source_relative_path,
        voice_lane_chain_id=context.voice_lane_chain_id,
        target_bar_index=target_bar.bar_index,
    )
    prompt_bars = tuple(
        _build_prompt_bar(
            bar,
            context.onset_tables_by_bar_id.get(bar.bar_id),
            context.onset_note_tables_by_onset_id,
        )
        for bar in prompt_source_bars
    )
    target_scaffold, target_fill = _build_target_bar_payloads(
        target_bar,
        context.onset_tables_by_bar_id.get(target_bar.bar_id),
        context.onset_note_tables_by_onset_id,
    )
    return ContinuationExample(
        relative_path=artifact_relative_path,
        trace_id=trace_id,
        example_id=example_id,
        source_relative_path=context.source_relative_path,
        document_id=context.document_id,
        split=context.split,
        split_version=context.split_version,
        normalized_ir_version=context.normalized_ir_version_key,
        continuation_dataset_version=context.continuation_dataset_version,
        task_contract_name=context.parameters.contract_name,
        task_contract_version=context.parameters.contract_version,
        part_id=context.part_id,
        staff_id=context.staff_id,
        voice_lane_chain_id=context.voice_lane_chain_id,
        target_bar_id=target_bar.bar_id,
        target_bar_index=target_bar.bar_index,
        prompt_bars=prompt_bars,
        target_scaffold=target_scaffold,
        target_fill=target_fill,
    )


def _build_prompt_bar(
    bar: Bar,
    onset_table: VoiceLaneOnsetTable | None,
    onset_note_tables_by_onset_id: Mapping[str, OnsetNoteTable],
) -> PromptBar:
    onsets = ()
    if onset_table is not None:
        onsets = tuple(
            PromptOnset(
                onset_id=row.onset_id,
                onset_slot_index=index,
                attack_order_in_voice=row.attack_order_in_voice,
                bar_offset=_effective_bar_offset(
                    preferred=row.bar_offset,
                    fallback=row.time,
                ),
                duration_notated=row.duration_notated,
                is_rest=row.is_rest,
                notes=tuple(
                    PromptNote(
                        note_id=note_row.note_id,
                        note_slot_index=note_index,
                        pitch_text=note_row.pitch_text,
                        attack_duration=note_row.attack_duration,
                    )
                    for note_index, note_row in enumerate(
                        _note_rows_for_onset(
                            onset_note_tables_by_onset_id,
                            onset_id=row.onset_id,
                        )
                    )
                ),
            )
            for index, row in enumerate(onset_table.rows)
        )
    return PromptBar(
        bar_id=bar.bar_id,
        bar_index=bar.bar_index,
        duration=bar.duration,
        time_signature=bar.time_signature,
        onsets=onsets,
    )


def _build_target_bar_payloads(
    bar: Bar,
    onset_table: VoiceLaneOnsetTable | None,
    onset_note_tables_by_onset_id: Mapping[str, OnsetNoteTable],
) -> tuple[TargetBarScaffold, TargetBarFill]:
    scaffold_onsets = []
    fill_onsets = []
    if onset_table is not None:
        for index, row in enumerate(onset_table.rows):
            note_rows = tuple(
                _note_rows_for_onset(
                    onset_note_tables_by_onset_id,
                    onset_id=row.onset_id,
                )
            )
            scaffold_onsets.append(
                TargetOnsetScaffold(
                    onset_id=row.onset_id,
                    onset_slot_index=index,
                    attack_order_in_voice=row.attack_order_in_voice,
                    bar_offset=_effective_bar_offset(
                        preferred=row.bar_offset,
                        fallback=row.time,
                    ),
                    duration_notated=row.duration_notated,
                    is_rest=row.is_rest,
                    note_slot_count=len(note_rows),
                )
            )
            fill_onsets.append(
                TargetOnsetFill(
                    onset_id=row.onset_id,
                    onset_slot_index=index,
                    note_slots=tuple(
                        TargetNoteSlot(
                            note_id=note_row.note_id,
                            note_slot_index=note_index,
                            pitch_text=note_row.pitch_text,
                            attack_duration=note_row.attack_duration,
                        )
                        for note_index, note_row in enumerate(note_rows)
                    ),
                )
            )
    return (
        TargetBarScaffold(
            bar_id=bar.bar_id,
            bar_index=bar.bar_index,
            duration=bar.duration,
            time_signature=bar.time_signature,
            onsets=tuple(scaffold_onsets),
        ),
        TargetBarFill(
            bar_id=bar.bar_id,
            bar_index=bar.bar_index,
            onsets=tuple(fill_onsets),
        ),
    )


def _build_rejection(
    record: MotifIrDocumentRecord,
    manifest_entry: SplitManifestEntry,
    rejection_reason: str,
) -> ContinuationDocumentRejection:
    return ContinuationDocumentRejection(
        relative_path=record.relative_path,
        document_id=manifest_entry.document_id,
        split=manifest_entry.split,
        split_version=manifest_entry.split_version,
        rejection_reason=rejection_reason,
    )


def _validate_example_contract(
    example: ContinuationExample,
    *,
    prompt_bar_count: int,
) -> None:
    if len(example.prompt_bars) != prompt_bar_count:
        raise ValueError(
            "Continuation examples must contain the configured prompt bars."
        )

    for prompt_bar in example.prompt_bars:
        for onset in prompt_bar.onsets:
            if onset.bar_offset >= prompt_bar.duration:
                raise ValueError("Prompt onset bar_offset must fall within the bar.")

    scaffold_onset_ids = tuple(
        onset.onset_id for onset in example.target_scaffold.onsets
    )
    fill_onset_ids = tuple(onset.onset_id for onset in example.target_fill.onsets)
    if scaffold_onset_ids != fill_onset_ids:
        raise ValueError("Target scaffold and fill must align onset-for-onset.")

    for scaffold_onset, fill_onset in zip(
        example.target_scaffold.onsets,
        example.target_fill.onsets,
        strict=True,
    ):
        if scaffold_onset.onset_slot_index != fill_onset.onset_slot_index:
            raise ValueError("Target scaffold and fill must align onset slot indexes.")
        if scaffold_onset.note_slot_count != len(fill_onset.note_slots):
            raise ValueError("Target fill note slots must match the scaffold count.")
        if scaffold_onset.bar_offset >= example.target_scaffold.duration:
            raise ValueError("Target onset bar_offset must fall within the bar.")


def _normalized_ir_version_key(
    normalized_ir_version: NormalizedIrVersionMetadata | Mapping[str, Any] | None,
) -> str:
    if normalized_ir_version is None:
        return _UNVERSIONED_NORMALIZED_IR
    return coerce_normalized_ir_version_metadata(
        normalized_ir_version
    ).normalized_ir_version


def _sorted_bars(document: MotifMlIrDocument) -> tuple[Bar, ...]:
    return tuple(sorted(document.bars, key=lambda bar: (bar.bar_index, bar.bar_id)))


def _onset_tables_by_bar_id(
    tables: Sequence[VoiceLaneOnsetTable],
    *,
    voice_lane_chain_id: str,
) -> dict[str, VoiceLaneOnsetTable]:
    return {
        table.bar_id: table
        for table in tables
        if table.voice_lane_chain_id == voice_lane_chain_id
    }


def _onset_note_tables_by_onset_id(
    tables: Sequence[OnsetNoteTable],
    *,
    voice_lane_chain_id: str,
) -> dict[str, OnsetNoteTable]:
    return {
        table.onset_id: table
        for table in tables
        if table.voice_lane_chain_id == voice_lane_chain_id
    }


def _note_rows_for_onset(
    onset_note_tables_by_onset_id: Mapping[str, OnsetNoteTable],
    *,
    onset_id: str,
) -> tuple[Any, ...]:
    onset_table = onset_note_tables_by_onset_id.get(onset_id)
    if onset_table is None:
        return ()
    return onset_table.rows


def _effective_bar_offset(*, preferred: ScoreTime, fallback: ScoreTime) -> ScoreTime:
    if preferred.numerator >= 0:
        return preferred
    return fallback


def _artifact_relative_path(
    *,
    split: str,
    source_relative_path: str,
    voice_lane_chain_id: str,
    target_bar_index: int,
) -> str:
    return (
        f"{split.strip()}/{source_relative_path}.bar_{target_bar_index:04d}."
        f"{_path_safe_identifier(voice_lane_chain_id)}"
    )


def _path_safe_identifier(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )


__all__ = ["extract_continuation_examples"]
