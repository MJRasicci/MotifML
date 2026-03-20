"""Fixture-backed determinism tests for the full IR build pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from motifml.datasets.json_dataset import JsonDataset
from motifml.datasets.motif_ir_corpus_dataset import MotifIrCorpusDataset
from motifml.datasets.motif_json_corpus_dataset import MotifJsonCorpusDataset
from motifml.pipelines.ir_build.pipeline import create_pipeline as create_ir_build
from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    default_parameters,
    fixture_entries,
    golden_fixture_entries,
    load_ir_document_bytes,
    load_json,
    materialize_raw_fixture_subset,
    run_session,
    write_test_conf,
)


def _fixture_id(fixture_entry: dict[str, Any]) -> str:
    return str(fixture_entry["fixture_id"])


EXPECTED_MANIFEST_COUNTS: dict[str, dict[str, dict[str, int]]] = {
    "ensemble_polyphony_controls": {
        "node_counts": {
            "Bar": 2,
            "NoteEvent": 7,
            "OnsetGroup": 6,
            "Part": 2,
            "PointControlEvent": 4,
            "SpanControlEvent": 2,
            "Staff": 3,
            "VoiceLane": 6,
        },
        "edge_counts": {
            "contains": 22,
            "next_in_voice": 3,
        },
    },
    "guitar_techniques_tuplets": {
        "node_counts": {
            "Bar": 1,
            "NoteEvent": 4,
            "OnsetGroup": 4,
            "Part": 1,
            "Staff": 1,
            "VoiceLane": 1,
        },
        "edge_counts": {
            "contains": 10,
            "next_in_voice": 3,
            "technique_to": 3,
        },
    },
    "single_track_monophonic_pickup": {
        "node_counts": {
            "Bar": 2,
            "NoteEvent": 3,
            "OnsetGroup": 4,
            "Part": 1,
            "PointControlEvent": 2,
            "Staff": 1,
            "VoiceLane": 2,
        },
        "edge_counts": {
            "contains": 10,
            "next_in_voice": 3,
            "tie_to": 1,
        },
    },
    "voice_reentry": {
        "node_counts": {
            "Bar": 3,
            "NoteEvent": 5,
            "OnsetGroup": 5,
            "Part": 1,
            "Staff": 1,
            "VoiceLane": 5,
        },
        "edge_counts": {
            "contains": 16,
            "next_in_voice": 3,
        },
    },
}


@pytest.mark.parametrize("fixture_entry", fixture_entries(), ids=_fixture_id)
def test_ir_build_fixture_runs_are_byte_stable_and_manifested(
    tmp_path: Path,
    fixture_entry: dict[str, Any],
):
    first_output_root = _run_fixture_pipeline(
        tmp_path / "first",
        fixture_entry["raw_motif_json_path"],
    )
    second_output_root = _run_fixture_pipeline(
        tmp_path / "second",
        fixture_entry["raw_motif_json_path"],
    )

    assert load_ir_document_bytes(first_output_root) == load_ir_document_bytes(
        second_output_root
    )
    assert (first_output_root / "motif_ir_manifest.json").read_bytes() == (
        second_output_root / "motif_ir_manifest.json"
    ).read_bytes()

    manifest_entry = load_json(first_output_root / "motif_ir_manifest.json")[0]
    expected_counts = EXPECTED_MANIFEST_COUNTS[fixture_entry["fixture_id"]]

    assert manifest_entry["node_counts"] == expected_counts["node_counts"]
    assert manifest_entry["edge_counts"] == expected_counts["edge_counts"]


@pytest.mark.parametrize("fixture_entry", golden_fixture_entries(), ids=_fixture_id)
def test_ir_validation_reports_zero_errors_for_golden_fixtures(
    tmp_path: Path,
    fixture_entry: dict[str, Any],
):
    output_root = _run_fixture_pipeline(
        tmp_path,
        fixture_entry["raw_motif_json_path"],
        include_validation=True,
    )

    validation_report = load_json(output_root / "motif_ir_validation_report.json")[0]

    assert validation_report["passed"] is True
    assert validation_report["error_count"] == 0
    assert validation_report["warning_count"] == 0


def test_ir_build_output_is_stable_when_raw_document_order_is_reversed(tmp_path: Path):
    raw_documents = MotifJsonCorpusDataset(filepath=str(MOTIF_JSON_FIXTURE_ROOT)).load()

    forward_output_root = _run_ir_build_with_documents(
        tmp_path / "forward",
        raw_documents,
    )
    reverse_output_root = _run_ir_build_with_documents(
        tmp_path / "reverse",
        list(reversed(raw_documents)),
    )

    assert load_ir_document_bytes(forward_output_root) == load_ir_document_bytes(
        reverse_output_root
    )
    assert (forward_output_root / "motif_ir_manifest.json").read_bytes() == (
        reverse_output_root / "motif_ir_manifest.json"
    ).read_bytes()


def _run_fixture_pipeline(
    tmp_path: Path,
    raw_motif_json_path: str,
    *,
    include_validation: bool = False,
) -> Path:
    raw_root = materialize_raw_fixture_subset(tmp_path, [raw_motif_json_path])
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(conf_source, ["ir_build"])
    if include_validation:
        run_session(conf_source, ["ir_validation"])

    return output_root


def _run_ir_build_with_documents(
    tmp_path: Path,
    raw_documents: list[object],
) -> Path:
    output_root = tmp_path / "artifacts"
    catalog = DataCatalog(
        {
            "raw_motif_json_corpus": MemoryDataset(data=raw_documents),
            "motif_ir_corpus": MotifIrCorpusDataset(
                filepath=str(output_root / "documents")
            ),
            "motif_ir_manifest": JsonDataset(
                filepath=str(output_root / "motif_ir_manifest.json")
            ),
            "params:ir_build_metadata": MemoryDataset(
                data=default_parameters()["ir_build_metadata"]
            ),
        }
    )

    SequentialRunner().run(create_ir_build(), catalog)

    return output_root


def _fixture_id(fixture_entry: dict[str, Any]) -> str:
    return str(fixture_entry["fixture_id"])
