"""Document assembly and manifest nodes for IR build."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from motifml import __version__ as MOTIFML_VERSION
from motifml.datasets.motif_ir_corpus_dataset import (
    MotifIrDocumentRecord,
    ir_artifact_path_for_source,
)
from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.models import (
    IrDocumentMetadata,
    IrManifestDiagnosticCategory,
    IrManifestDiagnosticSummary,
    IrManifestEntry,
    MotifMlIrDocument,
    OptionalOverlays,
    OptionalViews,
    TimeUnit,
)
from motifml.pipelines.ir_build.common import IR_DOCUMENT_OUTPUT_ROOT
from motifml.pipelines.ir_build.models import (
    BarEmissionResult,
    CanonicalScoreValidationResult,
    DiagnosticSeverity,
    IntrinsicEdgeEmissionResult,
    IrBuildDiagnostic,
    NoteEventEmissionResult,
    OnsetGroupEmissionResult,
    PartStaffEmissionResult,
    PointControlEmissionResult,
    SpanControlEmissionResult,
    VoiceLaneEmissionResult,
    WrittenTimeMapResult,
)


def assemble_ir_document(  # noqa: PLR0913
    documents: list[MotifJsonDocument],
    part_staff_emissions: list[PartStaffEmissionResult],
    bar_emissions: list[BarEmissionResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
    onset_group_emissions: list[OnsetGroupEmissionResult],
    note_event_emissions: list[NoteEventEmissionResult],
    point_control_emissions: list[PointControlEmissionResult],
    span_control_emissions: list[SpanControlEmissionResult],
    intrinsic_edge_emissions: list[IntrinsicEdgeEmissionResult],
    ir_build_metadata: Mapping[str, Any],
) -> list[MotifIrDocumentRecord]:
    """Assemble canonical IR documents from emitted entity families."""
    part_staff_by_path = {
        result.relative_path: result for result in part_staff_emissions
    }
    bar_by_path = {result.relative_path: result for result in bar_emissions}
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    onset_group_by_path = {
        result.relative_path: result for result in onset_group_emissions
    }
    note_event_by_path = {
        result.relative_path: result for result in note_event_emissions
    }
    point_control_by_path = {
        result.relative_path: result for result in point_control_emissions
    }
    span_control_by_path = {
        result.relative_path: result for result in span_control_emissions
    }
    intrinsic_edge_by_path = {
        result.relative_path: result for result in intrinsic_edge_emissions
    }

    records: list[MotifIrDocumentRecord] = []
    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        relative_path = document.relative_path
        part_staff_emission = _require_assembly_input(
            part_staff_by_path.get(relative_path),
            emission_name="part/staff emission",
            relative_path=relative_path,
        )
        bar_emission = _require_assembly_input(
            bar_by_path.get(relative_path),
            emission_name="bar emission",
            relative_path=relative_path,
        )
        voice_lane_emission = _require_assembly_input(
            voice_lane_by_path.get(relative_path),
            emission_name="voice lane emission",
            relative_path=relative_path,
        )
        onset_group_emission = _require_assembly_input(
            onset_group_by_path.get(relative_path),
            emission_name="onset group emission",
            relative_path=relative_path,
        )
        note_event_emission = _require_assembly_input(
            note_event_by_path.get(relative_path),
            emission_name="note event emission",
            relative_path=relative_path,
        )
        point_control_emission = _require_assembly_input(
            point_control_by_path.get(relative_path),
            emission_name="point control emission",
            relative_path=relative_path,
        )
        span_control_emission = _require_assembly_input(
            span_control_by_path.get(relative_path),
            emission_name="span control emission",
            relative_path=relative_path,
        )
        intrinsic_edge_emission = _require_assembly_input(
            intrinsic_edge_by_path.get(relative_path),
            emission_name="intrinsic edge emission",
            relative_path=relative_path,
        )

        records.append(
            MotifIrDocumentRecord(
                relative_path=relative_path,
                document=_assemble_document_model(
                    source_hash=document.sha256,
                    part_staff_emission=part_staff_emission,
                    bar_emission=bar_emission,
                    voice_lane_emission=voice_lane_emission,
                    onset_group_emission=onset_group_emission,
                    note_event_emission=note_event_emission,
                    point_control_emission=point_control_emission,
                    span_control_emission=span_control_emission,
                    intrinsic_edge_emission=intrinsic_edge_emission,
                    ir_build_metadata=ir_build_metadata,
                ),
            )
        )

    return records


def build_ir_manifest(  # noqa: PLR0913
    ir_documents: list[MotifIrDocumentRecord],
    canonical_score_validation_results: list[CanonicalScoreValidationResult],
    written_time_maps: list[WrittenTimeMapResult],
    part_staff_emissions: list[PartStaffEmissionResult],
    bar_emissions: list[BarEmissionResult],
    voice_lane_emissions: list[VoiceLaneEmissionResult],
    onset_group_emissions: list[OnsetGroupEmissionResult],
    note_event_emissions: list[NoteEventEmissionResult],
    point_control_emissions: list[PointControlEmissionResult],
    span_control_emissions: list[SpanControlEmissionResult],
    intrinsic_edge_emissions: list[IntrinsicEdgeEmissionResult],
    ir_build_metadata: Mapping[str, Any],
) -> list[IrManifestEntry]:
    """Build a deterministic manifest describing emitted IR documents."""
    validation_by_path = {
        result.relative_path: result for result in canonical_score_validation_results
    }
    written_time_by_path = {
        result.relative_path: result for result in written_time_maps
    }
    part_staff_by_path = {
        result.relative_path: result for result in part_staff_emissions
    }
    bar_by_path = {result.relative_path: result for result in bar_emissions}
    voice_lane_by_path = {
        result.relative_path: result for result in voice_lane_emissions
    }
    onset_group_by_path = {
        result.relative_path: result for result in onset_group_emissions
    }
    note_event_by_path = {
        result.relative_path: result for result in note_event_emissions
    }
    point_control_by_path = {
        result.relative_path: result for result in point_control_emissions
    }
    span_control_by_path = {
        result.relative_path: result for result in span_control_emissions
    }
    intrinsic_edge_by_path = {
        result.relative_path: result for result in intrinsic_edge_emissions
    }
    build_timestamp = _require_manifest_build_timestamp(ir_build_metadata)

    manifest_entries: list[IrManifestEntry] = []
    for record in sorted(ir_documents, key=lambda item: item.relative_path.casefold()):
        relative_path = record.relative_path
        validation_result = _require_assembly_input(
            validation_by_path.get(relative_path),
            emission_name="canonical score validation result",
            relative_path=relative_path,
        )
        written_time_map = _require_assembly_input(
            written_time_by_path.get(relative_path),
            emission_name="written time map",
            relative_path=relative_path,
        )
        part_staff_emission = _require_assembly_input(
            part_staff_by_path.get(relative_path),
            emission_name="part/staff emission",
            relative_path=relative_path,
        )
        bar_emission = _require_assembly_input(
            bar_by_path.get(relative_path),
            emission_name="bar emission",
            relative_path=relative_path,
        )
        voice_lane_emission = _require_assembly_input(
            voice_lane_by_path.get(relative_path),
            emission_name="voice lane emission",
            relative_path=relative_path,
        )
        onset_group_emission = _require_assembly_input(
            onset_group_by_path.get(relative_path),
            emission_name="onset group emission",
            relative_path=relative_path,
        )
        note_event_emission = _require_assembly_input(
            note_event_by_path.get(relative_path),
            emission_name="note event emission",
            relative_path=relative_path,
        )
        point_control_emission = _require_assembly_input(
            point_control_by_path.get(relative_path),
            emission_name="point control emission",
            relative_path=relative_path,
        )
        span_control_emission = _require_assembly_input(
            span_control_by_path.get(relative_path),
            emission_name="span control emission",
            relative_path=relative_path,
        )
        intrinsic_edge_emission = _require_assembly_input(
            intrinsic_edge_by_path.get(relative_path),
            emission_name="intrinsic edge emission",
            relative_path=relative_path,
        )

        manifest_diagnostics = _summarize_manifest_diagnostics(
            diagnostics=(
                *validation_result.diagnostics,
                *written_time_map.diagnostics,
                *part_staff_emission.diagnostics,
                *bar_emission.diagnostics,
                *voice_lane_emission.diagnostics,
                *onset_group_emission.diagnostics,
                *note_event_emission.diagnostics,
                *point_control_emission.diagnostics,
                *span_control_emission.diagnostics,
                *intrinsic_edge_emission.diagnostics,
            )
        )
        manifest_entries.append(
            IrManifestEntry(
                source_path=relative_path,
                source_hash=record.document.metadata.source_document_hash,
                ir_document_path=(
                    IR_DOCUMENT_OUTPUT_ROOT / ir_artifact_path_for_source(relative_path)
                ).as_posix(),
                build_timestamp=build_timestamp,
                node_counts=_count_document_node_families(record.document),
                edge_counts=_count_document_edge_families(record.document),
                unsupported_features_dropped=tuple(
                    summary.code
                    for summary in manifest_diagnostics
                    if summary.category
                    in {
                        IrManifestDiagnosticCategory.UNSUPPORTED,
                        IrManifestDiagnosticCategory.EXCLUDED,
                    }
                ),
                conversion_diagnostics=manifest_diagnostics,
            )
        )

    return manifest_entries


def merge_ir_manifest_fragments(
    shard_manifest_fragments: list[list[Mapping[str, Any]] | list[IrManifestEntry]],
) -> list[Mapping[str, Any] | IrManifestEntry]:
    """Merge shard-local IR manifest fragments into one global manifest."""
    merged_entries: list[Mapping[str, Any] | IrManifestEntry] = []
    for fragment in shard_manifest_fragments:
        merged_entries.extend(fragment)

    return sorted(
        merged_entries,
        key=lambda entry: (
            entry.source_path.casefold()
            if isinstance(entry, IrManifestEntry)
            else str(entry["source_path"]).casefold()
        ),
    )


def _require_assembly_input(
    emission: object,
    *,
    emission_name: str,
    relative_path: str,
) -> object:
    if emission is None:
        raise ValueError(
            f"Cannot assemble IR document for '{relative_path}': "
            f"{emission_name} is missing."
        )

    passed = getattr(emission, "passed", None)
    if passed is False:
        raise ValueError(
            f"Cannot assemble IR document for '{relative_path}': "
            f"{emission_name} contains fatal diagnostics."
        )

    return emission


def _require_manifest_build_timestamp(ir_build_metadata: Mapping[str, Any]) -> str:
    if not isinstance(ir_build_metadata, Mapping):
        raise ValueError("ir_build_metadata parameters must be a mapping.")

    build_timestamp = ir_build_metadata.get("build_timestamp")
    if not isinstance(build_timestamp, str) or not build_timestamp.strip():
        raise ValueError(
            "ir_build_metadata.build_timestamp must be a non-empty string."
        )

    return build_timestamp.strip()


def _assemble_document_model(  # noqa: PLR0913
    *,
    source_hash: str,
    part_staff_emission: PartStaffEmissionResult,
    bar_emission: BarEmissionResult,
    voice_lane_emission: VoiceLaneEmissionResult,
    onset_group_emission: OnsetGroupEmissionResult,
    note_event_emission: NoteEventEmissionResult,
    point_control_emission: PointControlEmissionResult,
    span_control_emission: SpanControlEmissionResult,
    intrinsic_edge_emission: IntrinsicEdgeEmissionResult,
    ir_build_metadata: Mapping[str, Any],
) -> MotifMlIrDocument:
    return MotifMlIrDocument(
        metadata=_build_ir_document_metadata(
            source_hash=source_hash,
            ir_build_metadata=ir_build_metadata,
        ),
        # Upstream emission result models already enforce canonical collection order.
        parts=tuple(part_staff_emission.parts),
        staves=tuple(part_staff_emission.staves),
        bars=tuple(bar_emission.bars),
        voice_lanes=tuple(voice_lane_emission.voice_lanes),
        point_control_events=tuple(point_control_emission.point_control_events),
        span_control_events=tuple(span_control_emission.span_control_events),
        onset_groups=tuple(onset_group_emission.onset_groups),
        note_events=tuple(note_event_emission.note_events),
        edges=tuple(intrinsic_edge_emission.edges),
        optional_overlays=OptionalOverlays(),
        optional_views=OptionalViews(),
    )


def _build_ir_document_metadata(
    *,
    source_hash: str,
    ir_build_metadata: Mapping[str, Any],
) -> IrDocumentMetadata:
    if not isinstance(ir_build_metadata, Mapping):
        raise ValueError("ir_build_metadata parameters must be a mapping.")

    ir_schema_version = ir_build_metadata.get("ir_schema_version")
    corpus_build_version = ir_build_metadata.get("corpus_build_version")
    compiled_resolution_hint = _coerce_optional_positive_int_param(
        ir_build_metadata.get("compiled_resolution_hint"),
        field_name="compiled_resolution_hint",
    )

    if not isinstance(ir_schema_version, str) or not ir_schema_version.strip():
        raise ValueError(
            "ir_build_metadata.ir_schema_version must be a non-empty string."
        )
    if not isinstance(corpus_build_version, str) or not corpus_build_version.strip():
        raise ValueError(
            "ir_build_metadata.corpus_build_version must be a non-empty string."
        )

    return IrDocumentMetadata(
        ir_schema_version=ir_schema_version.strip(),
        corpus_build_version=corpus_build_version.strip(),
        generator_version=MOTIFML_VERSION,
        source_document_hash=source_hash,
        time_unit=TimeUnit.WHOLE_NOTE_FRACTION,
        compiled_resolution_hint=compiled_resolution_hint,
    )


def _coerce_optional_positive_int_param(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None

    if not isinstance(value, int):
        raise ValueError(f"ir_build_metadata.{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"ir_build_metadata.{field_name} must be positive.")

    return value


def _count_document_node_families(document: MotifMlIrDocument) -> dict[str, int]:
    counts = {
        "Bar": len(document.bars),
        "NoteEvent": len(document.note_events),
        "OnsetGroup": len(document.onset_groups),
        "Part": len(document.parts),
        "PointControlEvent": len(document.point_control_events),
        "SpanControlEvent": len(document.span_control_events),
        "Staff": len(document.staves),
        "VoiceLane": len(document.voice_lanes),
    }
    if document.optional_overlays.phrase_spans:
        counts["PhraseSpan"] = len(document.optional_overlays.phrase_spans)

    return {name: count for name, count in counts.items() if count > 0}


def _count_document_edge_families(document: MotifMlIrDocument) -> dict[str, int]:
    counts = Counter(edge.edge_type.value for edge in document.edges)
    return {edge_type: counts[edge_type] for edge_type in sorted(counts)}


def _summarize_manifest_diagnostics(
    diagnostics: tuple[IrBuildDiagnostic, ...],
) -> tuple[IrManifestDiagnosticSummary, ...]:
    if not diagnostics:
        return ()

    grouped: dict[tuple[str, str, str], list[IrBuildDiagnostic]] = {}
    for diagnostic in diagnostics:
        category = _manifest_diagnostic_category(diagnostic)
        key = (category.value, diagnostic.severity.value, diagnostic.code)
        grouped.setdefault(key, []).append(diagnostic)

    summaries = [
        IrManifestDiagnosticSummary(
            category=category,
            severity=severity,
            code=code,
            count=len(group),
            paths=tuple(
                sorted({diagnostic.path for diagnostic in group}, key=str.casefold)
            ),
            messages=tuple(
                sorted({diagnostic.message for diagnostic in group}, key=str.casefold)
            ),
        )
        for (category, severity, code), group in grouped.items()
    ]
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                item.category.value,
                item.severity,
                item.code,
            ),
        )
    )


def _manifest_diagnostic_category(
    diagnostic: IrBuildDiagnostic,
) -> IrManifestDiagnosticCategory:
    if diagnostic.code == "unsupported_span_control_kind":
        if (
            diagnostic.severity is DiagnosticSeverity.WARNING
            and "'Legato'" in diagnostic.message
        ):
            return IrManifestDiagnosticCategory.EXCLUDED
        return IrManifestDiagnosticCategory.UNSUPPORTED

    if diagnostic.code in {
        "unsupported_note_relation_kind",
        "unsupported_onset_technique",
        "unsupported_point_control_kind",
    }:
        return IrManifestDiagnosticCategory.UNSUPPORTED

    if diagnostic.code in {
        "non_positive_fermata_length_scale",
        "open_ended_span_control",
        "out_of_range_control_position",
    }:
        return IrManifestDiagnosticCategory.EXCLUDED

    if diagnostic.code in {
        "ambiguous_note_reference",
        "bar_emission_failed",
        "canonical_surface_validation_failed",
        "duplicate_bar_index",
        "duplicate_part_id",
        "duplicate_raw_note_id",
        "duplicate_staff_index",
        "duplicate_voice_index",
        "invalid_canonical_field",
        "invalid_edge",
        "missing_bar_emission",
        "missing_bar_reference",
        "missing_canonical_field",
        "missing_edge_endpoint_reference",
        "missing_note_event_emission",
        "missing_note_event_reference",
        "missing_onset_group_emission",
        "missing_onset_group_reference",
        "missing_part_reference",
        "missing_part_staff_emission",
        "missing_staff_reference",
        "missing_validation_result",
        "missing_voice_lane_emission",
        "missing_voice_lane_reference",
        "missing_written_time_entry",
        "missing_written_time_map",
        "non_contiguous_bar_geometry",
        "non_positive_fermata_length_scale",
        "note_event_alignment_failed",
        "note_event_emission_failed",
        "onset_group_emission_failed",
        "overlapping_bar_geometry",
        "part_staff_emission_failed",
        "voice_lane_emission_failed",
        "written_time_map_failed",
    }:
        return IrManifestDiagnosticCategory.MALFORMED

    return IrManifestDiagnosticCategory.OTHER
