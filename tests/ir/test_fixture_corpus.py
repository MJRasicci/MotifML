"""Contract tests for the tracked IR fixture corpus and golden artifacts."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from motifml.ir.serialization import deserialize_document, serialize_document

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
CATALOG_PATH = FIXTURE_ROOT / "ir_fixture_catalog.json"
README_PATH = FIXTURE_ROOT / "motif_json" / "README.md"
REGENERATOR_PATH = REPO_ROOT / "tools" / "regenerate_ir_fixture_corpus.py"
PENDING_REVIEW_STATUS = "provisional_pending_human_review"


def test_fixture_catalog_covers_the_full_section_4_surface():
    catalog = _load_json(CATALOG_PATH)
    readme = README_PATH.read_text(encoding="utf-8")
    required_coverage = set(catalog["required_coverage"])
    allowed_review_statuses = set(catalog["allowed_golden_ir_review_statuses"])
    covered_surface: set[str] = set()
    golden_fixture_count = 0

    assert catalog["regeneration_command"] in readme

    for entry in catalog["fixtures"]:
        fixture_id = entry["fixture_id"]
        raw_path = FIXTURE_ROOT / entry["raw_motif_json_path"]

        assert fixture_id in readme
        assert raw_path.exists()

        covered_surface.update(entry["covers"])

        golden_relative_path = entry["golden_ir_path"]
        review_status = entry["golden_ir_review_status"]
        if golden_relative_path is None:
            assert review_status is None
            continue

        golden_fixture_count += 1
        assert (FIXTURE_ROOT / golden_relative_path).exists()
        assert review_status in allowed_review_statuses

    assert covered_surface == required_coverage
    assert 0 < golden_fixture_count < len(catalog["fixtures"])


def test_raw_fixture_corpus_validates_against_the_checked_in_motif_score_schema():
    catalog = _load_json(CATALOG_PATH)
    validator = Draft202012Validator(
        _load_json((FIXTURE_ROOT / catalog["raw_schema_path"]).resolve())
    )

    for entry in catalog["fixtures"]:
        payload = _load_json(FIXTURE_ROOT / entry["raw_motif_json_path"])
        errors = sorted(
            validator.iter_errors(payload), key=lambda error: error.json_path
        )
        assert not errors, _format_validation_errors(entry["fixture_id"], errors)


def test_golden_ir_artifacts_validate_and_round_trip_canonically():
    catalog = _load_json(CATALOG_PATH)
    validator = Draft202012Validator(
        _load_json((FIXTURE_ROOT / catalog["ir_schema_path"]).resolve())
    )

    for entry in catalog["fixtures"]:
        golden_relative_path = entry["golden_ir_path"]
        if golden_relative_path is None:
            continue

        golden_path = FIXTURE_ROOT / golden_relative_path
        serialized = golden_path.read_text(encoding="utf-8")
        payload = json.loads(serialized)
        errors = sorted(
            validator.iter_errors(payload), key=lambda error: error.json_path
        )

        assert not errors, _format_validation_errors(entry["fixture_id"], errors)
        assert serialize_document(deserialize_document(serialized)) == serialized


def test_regenerator_reproduces_the_tracked_fixture_corpus(tmp_path: Path):
    fixture_root = tmp_path / "fixtures"
    raw_root = fixture_root / "motif_json"
    raw_root.mkdir(parents=True, exist_ok=True)
    stale_raw_fixture = raw_root / "obsolete_fixture.json"
    stale_raw_fixture.write_text("{}\n", encoding="utf-8")
    (fixture_root / CATALOG_PATH.name).write_text(
        CATALOG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    generate_fixture_corpus = runpy.run_path(str(REGENERATOR_PATH))[
        "generate_fixture_corpus"
    ]
    generate_fixture_corpus(fixture_root)

    assert not stale_raw_fixture.exists()

    expected_paths = _generated_artifact_paths(_load_json(CATALOG_PATH))
    actual_paths = sorted(
        path.relative_to(fixture_root).as_posix()
        for path in fixture_root.rglob("*.json")
    )

    assert actual_paths == expected_paths
    for relative_path in expected_paths:
        assert (fixture_root / relative_path).read_text(encoding="utf-8") == (
            FIXTURE_ROOT / relative_path
        ).read_text(encoding="utf-8")


def test_new_golden_artifacts_default_to_pending_human_review(tmp_path: Path):
    fixture_root = tmp_path / "fixtures"
    generate_fixture_corpus = runpy.run_path(str(REGENERATOR_PATH))[
        "generate_fixture_corpus"
    ]
    generate_fixture_corpus(fixture_root)

    catalog = _load_json(fixture_root / CATALOG_PATH.name)
    golden_entries = [
        entry for entry in catalog["fixtures"] if entry["golden_ir_path"] is not None
    ]

    assert golden_entries
    assert all(
        entry["golden_ir_review_status"] == PENDING_REVIEW_STATUS
        for entry in golden_entries
    )


def _generated_artifact_paths(catalog: dict[str, Any]) -> list[str]:
    artifact_paths = [CATALOG_PATH.name]
    artifact_paths.extend(entry["raw_motif_json_path"] for entry in catalog["fixtures"])
    artifact_paths.extend(
        entry["golden_ir_path"]
        for entry in catalog["fixtures"]
        if entry["golden_ir_path"] is not None
    )
    return sorted(artifact_paths)


def _format_validation_errors(fixture_id: str, errors: list[Any]) -> str:
    lines = [f"{fixture_id} failed schema validation:"]
    for error in errors[:5]:
        lines.append(f"- {error.json_path or '$'}: {error.message}")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
