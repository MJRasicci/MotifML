"""End-to-end Kedro pipeline integration tests for IR build and validation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "motif_json"
EXPECTED_FIXTURE_COUNT = 4
EXPECTED_DEFAULT_STAGE_ORDER = [
    "build_raw_corpus_manifest",
    "summarize_raw_corpus",
    "stage_raw_corpus_for_ir_build",
    "validate_canonical_score_surface",
    "build_written_time_map",
    "emit_parts_and_staves",
    "emit_bars",
    "emit_voice_lanes",
    "emit_onset_groups",
    "emit_point_control_events",
    "emit_span_control_events",
    "emit_note_events",
    "emit_intrinsic_edges",
    "assemble_ir_document",
    "build_ir_manifest",
    "validate_ir_documents",
    "publish_ir_validation_report",
    "summarize_ir_corpus",
    "report_ir_scale_metrics",
]


def test_kedro_session_runs_ir_build_and_ir_validation_in_isolation(tmp_path: Path):
    bootstrap_project(REPO_ROOT)
    conf_source, output_root = _write_test_conf(tmp_path)

    with KedroSession.create(
        project_path=REPO_ROOT,
        conf_source=str(conf_source),
    ) as session:
        session.run(pipeline_names=["ir_build"], runner=SequentialRunner())

    ir_documents = sorted((output_root / "documents").rglob("*.ir.json"))
    manifest = _load_json(output_root / "motif_ir_manifest.json")

    assert len(ir_documents) == EXPECTED_FIXTURE_COUNT
    assert len(manifest) == EXPECTED_FIXTURE_COUNT
    assert not (output_root / "motif_ir_validation_report.json").exists()
    assert not (output_root / "motif_ir_summary.json").exists()

    with KedroSession.create(
        project_path=REPO_ROOT,
        conf_source=str(conf_source),
    ) as session:
        session.run(pipeline_names=["ir_validation"], runner=SequentialRunner())

    validation_report = _load_json(output_root / "motif_ir_validation_report.json")
    summary = _load_json(output_root / "motif_ir_summary.json")

    assert len(validation_report) == EXPECTED_FIXTURE_COUNT
    assert summary["document_count"] == EXPECTED_FIXTURE_COUNT


def test_kedro_session_runs_default_pipeline_in_stage_sequence(tmp_path: Path):
    bootstrap_project(REPO_ROOT)
    conf_source, output_root = _write_test_conf(tmp_path)

    with KedroSession.create(
        project_path=REPO_ROOT,
        conf_source=str(conf_source),
    ) as session:
        session.run(runner=SequentialRunner())

    assert _load_json(output_root / "raw_motif_json_manifest.json")
    assert _load_json(output_root / "raw_motif_json_summary.json")
    assert _load_json(output_root / "motif_ir_manifest.json")
    assert _load_json(output_root / "motif_ir_validation_report.json")
    assert _load_json(output_root / "motif_ir_summary.json")
    assert _default_stage_order() == EXPECTED_DEFAULT_STAGE_ORDER


def _write_test_conf(tmp_path: Path) -> tuple[Path, Path]:
    conf_source = tmp_path / "conf"
    conf_base = conf_source / "base"
    conf_local = conf_source / "local"
    conf_base.mkdir(parents=True, exist_ok=True)
    conf_local.mkdir(parents=True, exist_ok=True)

    output_root = tmp_path / "artifacts"
    catalog = {
        "raw_motif_json_corpus": {
            "type": "motifml.datasets.motif_json_corpus_dataset.MotifJsonCorpusDataset",
            "filepath": str(FIXTURE_ROOT),
            "glob_pattern": "**/*.json",
        },
        "raw_motif_json_manifest": {
            "type": "motifml.datasets.json_dataset.JsonDataset",
            "filepath": str(output_root / "raw_motif_json_manifest.json"),
        },
        "raw_motif_json_summary": {
            "type": "motifml.datasets.json_dataset.JsonDataset",
            "filepath": str(output_root / "raw_motif_json_summary.json"),
        },
        "motif_ir_manifest": {
            "type": "motifml.datasets.json_dataset.JsonDataset",
            "filepath": str(output_root / "motif_ir_manifest.json"),
        },
        "motif_ir_validation_report": {
            "type": "motifml.datasets.json_dataset.JsonDataset",
            "filepath": str(output_root / "motif_ir_validation_report.json"),
        },
        "motif_ir_summary": {
            "type": "motifml.datasets.json_dataset.JsonDataset",
            "filepath": str(output_root / "motif_ir_summary.json"),
        },
        "motif_ir_corpus": {
            "type": "motifml.datasets.motif_ir_corpus_dataset.MotifIrCorpusDataset",
            "filepath": str(output_root / "documents"),
        },
    }
    parameters = {
        "ir_build_metadata": {
            "ir_schema_version": "1.0.0",
            "corpus_build_version": "ir-build-v1",
            "build_timestamp": "2026-03-19T00:00:00-04:00",
        },
        "ir_validation": {
            "rule_severities": {
                "onset_ownership": "error",
                "note_ownership": "error",
                "note_time_alignment": "error",
                "voice_lane_onset_timing": "error",
                "attack_order_contiguity": "error",
                "sounding_duration_positive": "error",
                "tie_chain_linear": "error",
                "voice_lane_chain_stability": "error",
                "note_order_canonical": "error",
                "edge_endpoint_reference_integrity": "error",
                "forbidden_metadata_absent": "error",
                "phrase_span_validity": "error",
                "fretted_string_collision": "error",
            }
        },
    }

    (conf_base / "catalog.yml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    (conf_base / "parameters.yml").write_text(
        yaml.safe_dump(parameters, sort_keys=False),
        encoding="utf-8",
    )

    return conf_source, output_root


def _default_stage_order() -> list[str]:
    from motifml.pipeline_registry import register_pipelines

    return [node.name for node in register_pipelines()["__default__"].nodes]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
