"""Regenerate tracked training-preparation regression fixtures."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from motifml.datasets.tokenized_model_input_dataset import (  # noqa: E402
    TokenizedModelInputDataset,
)
from tests.pipelines.ir_test_support import run_session, write_test_conf  # noqa: E402
from tests.pipelines.training_test_support import (  # noqa: E402
    TRAINING_FIXTURE_ROOT,
    baseline_training_runtime_overrides,
    materialize_training_fixture_corpus,
)

TRACKED_OUTPUTS = (
    "split_manifest.json",
    "split_stats.json",
    "vocabulary.json",
    "vocabulary_version.json",
    "vocab_stats.json",
    "model_input_stats.json",
    "model_input/model_input_version.json",
    "model_input/parameters.json",
    "model_input/storage_schema.json",
)
REPRESENTATIVE_ROW_ROOT = TRAINING_FIXTURE_ROOT / "representative_rows"


def main() -> None:
    """Regenerate the tracked split, vocabulary, and model-input fixture outputs."""
    regenerated_paths: list[Path] = []
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
        conf_source, output_root = write_test_conf(tmp_path, raw_root)

        run_session(
            conf_source,
            ["__default__"],
            runtime_params=baseline_training_runtime_overrides(),
        )

        TRAINING_FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        for relative_path in TRACKED_OUTPUTS:
            source_path = output_root / relative_path
            target_path = TRAINING_FIXTURE_ROOT / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            regenerated_paths.append(target_path)

        rows = TokenizedModelInputDataset(
            filepath=str(output_root / "model_input")
        ).load()["records"]
        if REPRESENTATIVE_ROW_ROOT.exists():
            shutil.rmtree(REPRESENTATIVE_ROW_ROOT)
        for row in rows:
            target_path = _representative_row_path(row)
            _write_json(target_path, row)
            regenerated_paths.append(target_path)

    sys.stdout.write(
        "Regenerated training fixtures: "
        + ", ".join(
            path.relative_to(REPO_ROOT).as_posix() for path in sorted(regenerated_paths)
        )
        + "\n"
    )


def _representative_row_path(row: dict[str, Any]) -> Path:
    split = str(row["split"])
    relative_path = PurePosixPath(str(row["relative_path"]))
    row_path = PurePosixPath(
        "representative_rows",
        split,
        f"{relative_path.as_posix()}.row.json",
    )
    return TRAINING_FIXTURE_ROOT / Path(row_path.as_posix())


def _write_json(path: Path, payload: Any) -> None:
    serialized_text = json.dumps(
        payload,
        indent=2,
        ensure_ascii=True,
        check_circular=False,
    )
    serialized_bytes = f"{serialized_text}\n".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == serialized_bytes:
        return
    path.write_bytes(serialized_bytes)


if __name__ == "__main__":
    main()
