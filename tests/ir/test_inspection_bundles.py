"""Regression coverage for deterministic IR inspection bundle generation."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "generate_ir_inspection_bundles.py"
CHECKED_IN_BUNDLE_ROOT = REPO_ROOT / "tests" / "fixtures" / "ir" / "inspection_bundles"


def test_generate_inspection_bundles_reproduces_checked_in_artifacts(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "inspection_bundles"
    generate_inspection_bundles = runpy.run_path(str(TOOL_PATH))[
        "generate_inspection_bundles"
    ]

    generate_inspection_bundles(bundle_root)

    expected_files = _relative_files(CHECKED_IN_BUNDLE_ROOT)
    actual_files = _relative_files(bundle_root)

    assert actual_files == expected_files
    for relative_path in expected_files:
        assert (bundle_root / relative_path).read_text(encoding="utf-8") == (
            CHECKED_IN_BUNDLE_ROOT / relative_path
        ).read_text(encoding="utf-8")


def test_checked_in_inspection_bundles_cover_required_fixture_types() -> None:
    manifests = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CHECKED_IN_BUNDLE_ROOT.rglob("bundle_manifest.json"))
    }

    assert set(manifests) == {
        "ensemble_polyphony_controls",
        "guitar_techniques_tuplets",
    }
    assert all(manifest["schema_validation_passed"] for manifest in manifests.values())
    assert manifests["ensemble_polyphony_controls"]["validation_error_count"] == 0
    assert manifests["guitar_techniques_tuplets"]["validation_error_count"] == 1

    guitar_validation_report = json.loads(
        (
            CHECKED_IN_BUNDLE_ROOT
            / "guitar_techniques_tuplets"
            / "ir_validation_report.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        guitar_validation_report["rule_reports"][0]["rule"] == "voice_lane_onset_timing"
    )


def _relative_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
