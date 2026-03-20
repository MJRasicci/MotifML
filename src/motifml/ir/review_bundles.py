"""Review-bundle generation for human IR approval workflows."""

from __future__ import annotations

import csv
import importlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from motifml.datasets.json_dataset import JsonDataset
from motifml.datasets.motif_ir_corpus_dataset import (
    MotifIrCorpusDataset,
    MotifIrDocumentRecord,
)
from motifml.datasets.motif_json_corpus_dataset import (
    MotifJsonCorpusDataset,
    MotifJsonDocument,
)
from motifml.ir.review_models import IrReviewBundleManifest
from motifml.ir.review_tables import (
    build_control_event_rows,
    build_onset_note_tables,
    build_structure_summary,
    build_voice_lane_onset_tables,
    format_score_time,
)
from motifml.ir.review_visualizations import (
    render_control_timeline_svg,
    render_note_relations_svg,
    render_timeline_plot_svg,
    render_voice_lane_ladder_svg,
)
from motifml.ir.serialization import serialize_document
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.pipeline import (
    create_pipeline as create_ir_build_pipeline,
)
from motifml.pipelines.ir_validation.pipeline import (
    create_pipeline as create_ir_validation_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
RAW_FIXTURE_ROOT = FIXTURE_ROOT / "motif_json"
FIXTURE_CATALOG_PATH = FIXTURE_ROOT / "ir_fixture_catalog.json"
IR_SCHEMA_PATH = (
    REPO_ROOT / "src" / "motifml" / "ir" / "schema" / "motifml-ir-document.schema.json"
)
PARAMETERS_PATH = REPO_ROOT / "conf" / "base" / "parameters.yml"
DEFAULT_OUTPUT_ROOT = FIXTURE_ROOT / "ir" / "review_bundles"
DEFAULT_REVIEW_BUNDLE_FIXTURE_IDS = (
    "ensemble_polyphony_controls",
    "guitar_techniques_tuplets",
)
REVIEW_BUNDLE_VERSION = "1.0.0"
BUNDLE_ARTIFACT_NAMES = (
    "README.md",
    "bundle_manifest.json",
    "source_identity.json",
    "ir_document.ir.json",
    "schema_validation.json",
    "ir_validation_report.json",
    "structural_summary.json",
    "voice_lane_onsets.csv",
    "onset_notes.csv",
    "control_events.csv",
    "timeline_plot.svg",
    "voice_lane_ladder.svg",
    "note_relations.svg",
    "control_timeline.svg",
)


def load_fixture_catalog(
    fixture_root: Path = FIXTURE_ROOT,
) -> dict[str, Any]:
    """Load the tracked IR fixture catalog."""
    return json.loads(
        (fixture_root / FIXTURE_CATALOG_PATH.name).read_text(encoding="utf-8")
    )


def generate_review_bundles(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fixture_root: Path = FIXTURE_ROOT,
    fixture_ids: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Generate deterministic review bundles for selected tracked fixtures."""
    catalog = load_fixture_catalog(fixture_root)
    raw_documents = _load_raw_documents_by_relative_path(fixture_root)
    parameters = _load_parameters()
    entries = _resolve_fixture_entries(catalog, fixture_ids)

    if fixture_ids is None and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for entry in entries:
        fixture_id = str(entry["fixture_id"])
        bundle_root = output_root / fixture_id
        if bundle_root.exists():
            shutil.rmtree(bundle_root)

        raw_relative_path = _raw_relative_path(entry)
        raw_document = raw_documents[raw_relative_path]
        bundle_files = build_review_bundle_files(entry, raw_document, parameters)
        _write_bundle(bundle_root, bundle_files)
        written_paths.append(bundle_root)

    return tuple(written_paths)


def build_review_bundle_files(
    fixture_entry: Mapping[str, Any],
    raw_document: MotifJsonDocument,
    parameters: Mapping[str, Any],
) -> dict[str, str]:
    """Build all persisted files for one review bundle."""
    ir_record, manifest_entry, validation_report = _run_review_pipelines(
        raw_document,
        parameters,
    )
    serialized_document = serialize_document(ir_record.document)
    schema_validation = _validate_serialized_ir(serialized_document)
    structural_summary = build_structure_summary(ir_record.document)
    voice_lane_tables = build_voice_lane_onset_tables(ir_record.document)
    onset_note_tables = build_onset_note_tables(ir_record.document)
    control_rows = build_control_event_rows(ir_record.document)
    bundle_manifest = IrReviewBundleManifest(
        bundle_version=REVIEW_BUNDLE_VERSION,
        fixture_id=str(fixture_entry["fixture_id"]),
        description=str(fixture_entry["description"]),
        source_path=str(fixture_entry["raw_motif_json_path"]),
        source_hash=raw_document.sha256,
        ir_document_path="ir_document.ir.json",
        schema_validation_passed=bool(schema_validation["passed"]),
        validation_error_count=int(validation_report["error_count"]),
        validation_warning_count=int(validation_report["warning_count"]),
        artifacts=BUNDLE_ARTIFACT_NAMES,
    )

    return {
        "README.md": _render_bundle_readme(
            fixture_entry=fixture_entry,
            raw_document=raw_document,
            manifest_entry=manifest_entry,
            validation_report=validation_report,
            schema_validation=schema_validation,
            structural_summary=structural_summary,
            bundle_manifest=bundle_manifest,
        ),
        "bundle_manifest.json": _dump_json(bundle_manifest),
        "source_identity.json": _dump_json(
            {
                "fixture_id": fixture_entry["fixture_id"],
                "description": fixture_entry["description"],
                "covers": list(fixture_entry["covers"]),
                "source_path": fixture_entry["raw_motif_json_path"],
                "source_hash": raw_document.sha256,
                "source_file_size_bytes": raw_document.file_size_bytes,
                "pipeline_ir_artifact_path": manifest_entry["ir_document_path"],
                "golden_ir_path": fixture_entry.get("golden_ir_path"),
                "golden_ir_review_status": fixture_entry.get("golden_ir_review_status"),
            }
        ),
        "ir_document.ir.json": serialized_document,
        "schema_validation.json": _dump_json(schema_validation),
        "ir_validation_report.json": _dump_json(validation_report),
        "structural_summary.json": _dump_json(structural_summary),
        "voice_lane_onsets.csv": _render_voice_lane_onsets_csv(voice_lane_tables),
        "onset_notes.csv": _render_onset_notes_csv(onset_note_tables),
        "control_events.csv": _render_control_events_csv(control_rows),
        "timeline_plot.svg": render_timeline_plot_svg(ir_record.document),
        "voice_lane_ladder.svg": render_voice_lane_ladder_svg(ir_record.document),
        "note_relations.svg": render_note_relations_svg(ir_record.document),
        "control_timeline.svg": render_control_timeline_svg(ir_record.document),
    }


def _load_parameters() -> dict[str, Any]:
    return yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8"))


def _load_raw_documents_by_relative_path(
    fixture_root: Path,
) -> dict[str, MotifJsonDocument]:
    dataset = MotifJsonCorpusDataset(filepath=str(fixture_root / RAW_FIXTURE_ROOT.name))
    return {document.relative_path: document for document in dataset.load()}


def _resolve_fixture_entries(
    catalog: Mapping[str, Any],
    fixture_ids: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    entries_by_id = {
        str(entry["fixture_id"]): entry for entry in catalog.get("fixtures", [])
    }
    selected_ids = (
        tuple(DEFAULT_REVIEW_BUNDLE_FIXTURE_IDS)
        if fixture_ids is None
        else tuple(fixture_ids)
    )
    missing_ids = [
        fixture_id for fixture_id in selected_ids if fixture_id not in entries_by_id
    ]
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise ValueError(f"Unknown fixture ids requested for review bundles: {missing}")

    return [entries_by_id[fixture_id] for fixture_id in selected_ids]


def _raw_relative_path(fixture_entry: Mapping[str, Any]) -> str:
    raw_path = Path(str(fixture_entry["raw_motif_json_path"]))
    if raw_path.parts and raw_path.parts[0] == RAW_FIXTURE_ROOT.name:
        return Path(*raw_path.parts[1:]).as_posix()
    return raw_path.as_posix()


def _run_review_pipelines(
    raw_document: MotifJsonDocument,
    parameters: Mapping[str, Any],
) -> tuple[MotifIrDocumentRecord, Mapping[str, Any], Mapping[str, Any]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        catalog = DataCatalog(
            {
                "raw_motif_json_corpus": MemoryDataset(data=[raw_document]),
                "motif_ir_corpus": MotifIrCorpusDataset(
                    filepath=str(temp_root / "documents")
                ),
                "motif_ir_manifest": JsonDataset(
                    filepath=str(temp_root / "motif_ir_manifest.json")
                ),
                "motif_ir_validation_report": JsonDataset(
                    filepath=str(temp_root / "motif_ir_validation_report.json")
                ),
                "motif_ir_summary": JsonDataset(
                    filepath=str(temp_root / "motif_ir_summary.json")
                ),
                "params:ir_build_metadata": MemoryDataset(
                    data=parameters["ir_build_metadata"]
                ),
                "params:ir_validation": MemoryDataset(data=parameters["ir_validation"]),
            }
        )

        runner = SequentialRunner()
        runner.run(
            create_ir_build_pipeline() + create_ir_validation_pipeline(), catalog
        )

        ir_records = catalog.load("motif_ir_corpus")
        manifest_entries = catalog.load("motif_ir_manifest")
        validation_reports = catalog.load("motif_ir_validation_report")

        if (
            len(ir_records) != 1
            or len(manifest_entries) != 1
            or len(validation_reports) != 1
        ):
            raise ValueError(
                "Review bundle generation expects exactly one emitted IR document, "
                "manifest entry, and validation report per fixture."
            )

        return (ir_records[0], manifest_entries[0], validation_reports[0])


def _validate_serialized_ir(serialized_document: str) -> dict[str, Any]:
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in dev env
        raise RuntimeError(
            "jsonschema is required to generate IR review bundles. "
            "Install the MotifML dev dependencies first."
        ) from exc

    schema = json.loads(IR_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    payload = json.loads(serialized_document)
    errors = sorted(validator.iter_errors(payload), key=lambda error: error.json_path)
    return {
        "schema_path": IR_SCHEMA_PATH.relative_to(REPO_ROOT).as_posix(),
        "passed": not errors,
        "error_count": len(errors),
        "errors": [
            {
                "json_path": error.json_path or "$",
                "message": error.message,
            }
            for error in errors
        ],
    }


def _render_bundle_readme(  # noqa: PLR0913
    *,
    fixture_entry: Mapping[str, Any],
    raw_document: MotifJsonDocument,
    manifest_entry: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    schema_validation: Mapping[str, Any],
    structural_summary: object,
    bundle_manifest: IrReviewBundleManifest,
) -> str:
    lines = [
        f"# IR Review Bundle: {fixture_entry['fixture_id']}",
        "",
        str(fixture_entry["description"]),
        "",
        "## Source",
        f"- Raw fixture: `{fixture_entry['raw_motif_json_path']}`",
        f"- Source hash: `{raw_document.sha256}`",
        f"- Source file size: `{raw_document.file_size_bytes}` bytes",
        f"- Pipeline IR artifact path: `{manifest_entry['ir_document_path']}`",
        "",
        "## Validation",
        (
            f"- JSON schema validation: `{'passed' if schema_validation['passed'] else 'failed'}`"
        ),
        f"- IR validation errors: `{validation_report['error_count']}`",
        f"- IR validation warnings: `{validation_report['warning_count']}`",
        "",
        "## Structure Counts",
        f"- Parts: `{structural_summary.part_count}`",
        f"- Staves: `{structural_summary.staff_count}`",
        f"- Bars: `{structural_summary.bar_count}`",
        f"- Voice lanes: `{structural_summary.voice_lane_count}`",
        f"- Onset groups: `{structural_summary.onset_count}`",
        f"- Note events: `{structural_summary.note_count}`",
        f"- Point controls: `{structural_summary.point_control_count}`",
        f"- Span controls: `{structural_summary.span_control_count}`",
        f"- Edges: `{structural_summary.edge_count}`",
    ]

    unsupported_features = manifest_entry.get("unsupported_features_dropped", [])
    if unsupported_features:
        lines.extend(
            (
                "",
                "## Unsupported or Dropped Features",
                *(f"- `{feature}`" for feature in unsupported_features),
            )
        )

    conversion_diagnostics = manifest_entry.get("conversion_diagnostics", [])
    if conversion_diagnostics:
        lines.extend(("", "## Conversion Diagnostics"))
        lines.extend(
            (
                f"- `{diagnostic['code']}` ({diagnostic['category']}, "
                f"{diagnostic['severity']}, count={diagnostic['count']})"
            )
            for diagnostic in conversion_diagnostics
        )

    lines.extend(("", "## Bundle Artifacts"))
    lines.extend(f"- `{artifact}`" for artifact in bundle_manifest.artifacts)
    return "\n".join(lines) + "\n"


def _render_voice_lane_onsets_csv(tables: Sequence[object]) -> str:
    rows = (
        (
            row.part_id,
            row.staff_id,
            str(row.bar_index),
            row.bar_id,
            row.voice_lane_chain_id,
            row.voice_lane_id,
            str(row.voice_index),
            row.onset_id,
            format_score_time(row.time),
            format_score_time(row.bar_offset),
            format_score_time(row.duration_notated),
            format_score_time(row.duration_sounding_max),
            "true" if row.is_rest else "false",
            str(row.attack_order_in_voice),
            str(row.note_count),
            row.grace_type or "",
            row.dynamic_local or "",
            row.technique_summary or "",
        )
        for table in tables
        for row in table.rows
    )
    return _render_csv(
        (
            "part_id",
            "staff_id",
            "bar_index",
            "bar_id",
            "voice_lane_chain_id",
            "voice_lane_id",
            "voice_index",
            "onset_id",
            "time",
            "bar_offset",
            "duration_notated",
            "duration_sounding_max",
            "is_rest",
            "attack_order_in_voice",
            "note_count",
            "grace_type",
            "dynamic_local",
            "technique_summary",
        ),
        rows,
    )


def _render_onset_notes_csv(tables: Sequence[object]) -> str:
    rows = (
        (
            row.part_id,
            row.staff_id,
            str(row.bar_index),
            row.bar_id,
            row.voice_lane_chain_id,
            row.voice_lane_id,
            str(row.voice_index),
            row.onset_id,
            row.note_id,
            format_score_time(row.time),
            format_score_time(row.bar_offset),
            row.pitch_text,
            format_score_time(row.attack_duration),
            format_score_time(row.sounding_duration),
            "" if row.string_number is None else str(row.string_number),
            "" if row.velocity is None else str(row.velocity),
            row.technique_summary or "",
        )
        for table in tables
        for row in table.rows
    )
    return _render_csv(
        (
            "part_id",
            "staff_id",
            "bar_index",
            "bar_id",
            "voice_lane_chain_id",
            "voice_lane_id",
            "voice_index",
            "onset_id",
            "note_id",
            "time",
            "bar_offset",
            "pitch_text",
            "attack_duration",
            "sounding_duration",
            "string_number",
            "velocity",
            "technique_summary",
        ),
        rows,
    )


def _render_control_events_csv(rows: Sequence[object]) -> str:
    return _render_csv(
        (
            "control_id",
            "family",
            "kind",
            "scope",
            "target_ref",
            "start_time",
            "end_time",
            "start_bar_index",
            "end_bar_index",
            "value_summary",
            "start_anchor_ref",
            "end_anchor_ref",
        ),
        (
            (
                row.control_id,
                row.family,
                row.kind,
                row.scope,
                row.target_ref,
                format_score_time(row.start_time),
                format_score_time(row.end_time),
                "" if row.start_bar_index is None else str(row.start_bar_index),
                "" if row.end_bar_index is None else str(row.end_bar_index),
                row.value_summary,
                row.start_anchor_ref or "",
                row.end_anchor_ref or "",
            )
            for row in rows
        ),
    )


def _render_csv(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]] | Any,
) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _dump_json(value: Any) -> str:
    return json.dumps(_json_value(value), indent=2, ensure_ascii=True) + "\n"


def _json_value(value: Any) -> Any:
    if isinstance(value, ScoreTime):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
        }

    if is_dataclass(value):
        payload: dict[str, Any] = {}
        for field_info in fields(value):
            payload[field_info.name] = _json_value(getattr(value, field_info.name))
        return payload

    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}

    if hasattr(value, "value"):
        return value.value

    return value


def _write_bundle(bundle_root: Path, bundle_files: Mapping[str, str]) -> None:
    bundle_root.mkdir(parents=True, exist_ok=True)
    for relative_path, content in sorted(bundle_files.items()):
        target_path = bundle_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_content = content if content.endswith("\n") else f"{content}\n"
        target_path.write_text(normalized_content, encoding="utf-8")


__all__ = [
    "BUNDLE_ARTIFACT_NAMES",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_REVIEW_BUNDLE_FIXTURE_IDS",
    "REVIEW_BUNDLE_VERSION",
    "build_review_bundle_files",
    "generate_review_bundles",
    "load_fixture_catalog",
]
