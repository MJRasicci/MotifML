"""Contract tests for the approved training fixture slice."""

from __future__ import annotations

from pathlib import Path

from tests.pipelines.ir_test_support import FIXTURE_ROOT
from tests.pipelines.training_test_support import (
    TRAINING_FIXTURE_ROOT,
    load_training_fixture_catalog,
    materialize_training_fixture_corpus,
)

MAX_APPROVED_FIXTURE_COUNT = 4
MAX_APPROVED_TOTAL_BYTES = 50_000


def test_training_fixture_catalog_covers_objectives_and_readme() -> None:
    catalog = load_training_fixture_catalog()
    readme = (TRAINING_FIXTURE_ROOT / "README.md").read_text(encoding="utf-8")

    fixtures = catalog["fixtures"]
    assert fixtures
    assert len(fixtures) <= MAX_APPROVED_FIXTURE_COUNT

    covered_objectives: set[str] = set()
    total_bytes = 0
    for entry in fixtures:
        fixture_id = entry["fixture_id"]
        raw_relative_path = entry["raw_motif_json_path"]
        source_path = FIXTURE_ROOT / raw_relative_path

        assert source_path.exists()
        assert fixture_id in readme
        assert raw_relative_path in readme

        total_bytes += source_path.stat().st_size
        for objective in entry["training_proves"]:
            covered_objectives.add(objective)
            assert objective in readme

    assert set(catalog["training_objectives"]) == covered_objectives
    assert total_bytes <= MAX_APPROVED_TOTAL_BYTES


def test_materialize_training_fixture_corpus_copies_only_approved_raw_documents(
    tmp_path: Path,
) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path)

    copied_paths = sorted(
        path.relative_to(raw_root).as_posix() for path in raw_root.rglob("*.json")
    )
    expected_paths = sorted(
        Path(entry["raw_motif_json_path"]).name
        for entry in load_training_fixture_catalog()["fixtures"]
    )

    assert copied_paths == expected_paths
