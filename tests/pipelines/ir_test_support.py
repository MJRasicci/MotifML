"""Shared helpers for IR pipeline integration and fixture determinism tests."""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.runner import SequentialRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
MOTIF_JSON_FIXTURE_ROOT = FIXTURE_ROOT / "motif_json"
IR_FIXTURE_CATALOG_PATH = FIXTURE_ROOT / "ir_fixture_catalog.json"
PARAMETERS_PATH = REPO_ROOT / "conf" / "base" / "parameters.yml"


@lru_cache(maxsize=1)
def default_parameters() -> dict[str, Any]:
    """Load the project's canonical Kedro parameters for test runs."""
    return yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_fixture_catalog() -> dict[str, Any]:
    """Load the tracked representative fixture catalog."""
    return json.loads(IR_FIXTURE_CATALOG_PATH.read_text(encoding="utf-8"))


def fixture_entries() -> list[dict[str, Any]]:
    """Return all tracked representative fixture entries."""
    return list(load_fixture_catalog()["fixtures"])


def approved_fixture_entries() -> list[dict[str, Any]]:
    """Return the fixtures whose golden IR artifacts are human-approved."""
    return [
        entry
        for entry in fixture_entries()
        if entry["golden_ir_review_status"] == "approved_by_human"
    ]


def materialize_raw_fixture_subset(
    destination: Path,
    raw_motif_json_paths: list[str],
) -> Path:
    """Copy the requested raw fixtures into a temporary corpus root."""
    raw_root = destination / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    for raw_relative_path in raw_motif_json_paths:
        source_path = FIXTURE_ROOT / raw_relative_path
        relative_parts = Path(raw_relative_path).parts
        target_relative_path = (
            Path(*relative_parts[1:])
            if relative_parts and relative_parts[0] == "motif_json"
            else Path(raw_relative_path).name
        )
        target_path = raw_root / target_relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    return raw_root


def write_test_conf(tmp_path: Path, raw_corpus_path: Path) -> tuple[Path, Path]:
    """Write a temporary Kedro config tree that points at fixture-backed data."""
    conf_source = tmp_path / "conf"
    conf_base = conf_source / "base"
    conf_local = conf_source / "local"
    conf_base.mkdir(parents=True, exist_ok=True)
    conf_local.mkdir(parents=True, exist_ok=True)

    output_root = tmp_path / "artifacts"
    catalog = {
        "raw_motif_json_corpus": {
            "type": "motifml.datasets.motif_json_corpus_dataset.MotifJsonCorpusDataset",
            "filepath": str(raw_corpus_path),
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
        "normalized_ir_corpus": {
            "type": "motifml.datasets.motif_ir_corpus_dataset.MotifIrCorpusDataset",
            "filepath": str(output_root / "normalized_documents"),
        },
        "ir_features": {
            "type": "motifml.datasets.json_dataset.JsonDataset",
            "filepath": str(output_root / "ir_features.json"),
        },
    }

    (conf_base / "catalog.yml").write_text(
        yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    (conf_base / "parameters.yml").write_text(
        yaml.safe_dump(default_parameters(), sort_keys=False),
        encoding="utf-8",
    )

    return conf_source, output_root


def run_session(conf_source: Path, pipeline_names: list[str] | None = None) -> None:
    """Execute one Kedro session run against the temporary test config."""
    bootstrap_project(REPO_ROOT)
    with KedroSession.create(
        project_path=REPO_ROOT,
        conf_source=str(conf_source),
    ) as session:
        session.run(pipeline_names=pipeline_names, runner=SequentialRunner())


def load_json(path: Path) -> Any:
    """Load JSON content from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_ir_document_bytes(output_root: Path) -> dict[str, bytes]:
    """Load emitted IR document artifacts in deterministic relative-path order."""
    documents_root = output_root / "documents"
    return {
        path.relative_to(documents_root).as_posix(): path.read_bytes()
        for path in sorted(documents_root.rglob("*.ir.json"))
    }
