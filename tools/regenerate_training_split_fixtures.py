"""Regenerate tracked split-planning regression fixtures."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from tests.pipelines.ir_test_support import (  # noqa: E402
    MOTIF_JSON_FIXTURE_ROOT,
    run_session,
    write_test_conf,
)

TRAINING_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "training"
OUTPUT_FILE_NAMES = ("split_manifest.json", "split_stats.json")


def main() -> None:
    """Regenerate the tracked split manifest and split stats fixtures."""
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)
        run_session(conf_source, ["ir_build"])
        run_session(conf_source, ["normalization"])
        run_session(conf_source, ["dataset_splitting"])

        TRAINING_FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        for file_name in OUTPUT_FILE_NAMES:
            shutil.copy2(output_root / file_name, TRAINING_FIXTURE_ROOT / file_name)

    sys.stdout.write(
        "Regenerated training split fixtures: "
        + ", ".join(
            (TRAINING_FIXTURE_ROOT / file_name).relative_to(REPO_ROOT).as_posix()
            for file_name in OUTPUT_FILE_NAMES
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
