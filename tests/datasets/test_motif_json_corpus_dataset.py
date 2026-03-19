"""Tests for the raw Motif JSON Kedro dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

import pytest
from kedro.io import DatasetError

from motifml.datasets.motif_json_corpus_dataset import MotifJsonCorpusDataset

EXPECTED_INITIAL_AUTOBUILD_COUNT = 1
EXPECTED_REBUILD_AUTOBUILD_COUNT = 2


def test_motif_json_corpus_dataset_loads_documents_in_path_order(tmp_path):
    raw_root = tmp_path / "01_raw" / "motif_json"
    alpha_path = raw_root / "Artist A" / "Alpha.json"
    beta_path = raw_root / "Artist B" / "Beta.json"
    alpha_path.parent.mkdir(parents=True)
    beta_path.parent.mkdir(parents=True)

    alpha_path.write_text(
        json.dumps(
            {
                "Title": "Alpha",
                "Artist": "Artist A",
                "Album": "Album A",
                "Tracks": [{"Name": "Lead"}],
                "PlaybackMasterBarSequence": [{}, {}],
            }
        ),
        encoding="utf-8",
    )
    beta_path.write_text(
        json.dumps(
            {
                "Title": "Beta",
                "Artist": "Artist B",
                "Album": "Album B",
                "Tracks": [{"Name": "Rhythm"}, {"Name": "Bass"}],
                "PlaybackMasterBarSequence": [{}],
            }
        ),
        encoding="utf-8",
    )

    dataset = MotifJsonCorpusDataset(filepath=str(raw_root))

    documents = dataset.load()

    assert [document.relative_path for document in documents] == [
        "Artist A/Alpha.json",
        "Artist B/Beta.json",
    ]
    assert documents[0].score["Title"] == "Alpha"
    assert documents[1].score["Tracks"][1]["Name"] == "Bass"
    assert documents[0].sha256 == hashlib.sha256(alpha_path.read_bytes()).hexdigest()


def test_motif_json_corpus_dataset_requires_existing_root(tmp_path):
    dataset = MotifJsonCorpusDataset(filepath=str(tmp_path / "missing"))

    with pytest.raises(DatasetError, match="does not exist"):
        dataset.load()


def test_motif_json_corpus_dataset_is_read_only(tmp_path):
    raw_root = tmp_path / "01_raw" / "motif_json"
    raw_root.mkdir(parents=True)
    dataset = MotifJsonCorpusDataset(filepath=str(raw_root))

    with pytest.raises(DatasetError, match="read-only"):
        dataset.save(object())


def test_motif_json_corpus_dataset_auto_builds_raw_corpus(tmp_path):
    source_root = tmp_path / "00_corpus"
    build_state_path = (
        tmp_path / "02_intermediate" / "ingestion" / "raw_motif_json_build_state.json"
    )
    cli_path = _write_fake_motif_cli(tmp_path / "tools" / "motif-cli")
    _write_source_file(source_root / "Artist A" / "Alpha.gp5")

    dataset = MotifJsonCorpusDataset(
        filepath=str(tmp_path / "01_raw" / "motif_json"),
        autobuild={
            "source_filepath": str(source_root),
            "cli_filepath": str(cli_path),
            "build_state_filepath": str(build_state_path),
        },
    )

    documents = dataset.load()

    assert [document.relative_path for document in documents] == ["Artist A/Alpha.json"]
    assert documents[0].score["Title"] == "Alpha"
    assert build_state_path.exists()
    assert (
        _read_fake_motif_cli_invocations(cli_path) == EXPECTED_INITIAL_AUTOBUILD_COUNT
    )


def test_motif_json_corpus_dataset_skips_autobuild_when_inputs_are_unchanged(tmp_path):
    source_root = tmp_path / "00_corpus"
    build_state_path = (
        tmp_path / "02_intermediate" / "ingestion" / "raw_motif_json_build_state.json"
    )
    cli_path = _write_fake_motif_cli(tmp_path / "tools" / "motif-cli")
    _write_source_file(source_root / "Artist A" / "Alpha.gp5")

    dataset = MotifJsonCorpusDataset(
        filepath=str(tmp_path / "01_raw" / "motif_json"),
        autobuild={
            "source_filepath": str(source_root),
            "cli_filepath": str(cli_path),
            "build_state_filepath": str(build_state_path),
        },
    )

    first_documents = dataset.load()
    second_documents = dataset.load()

    assert [document.relative_path for document in first_documents] == [
        "Artist A/Alpha.json"
    ]
    assert [document.relative_path for document in second_documents] == [
        "Artist A/Alpha.json"
    ]
    assert (
        _read_fake_motif_cli_invocations(cli_path) == EXPECTED_INITIAL_AUTOBUILD_COUNT
    )


def test_motif_json_corpus_dataset_skips_autobuild_when_build_state_paths_differ(
    tmp_path,
):
    source_root = tmp_path / "00_corpus"
    raw_root = tmp_path / "01_raw" / "motif_json"
    build_state_path = (
        tmp_path / "02_intermediate" / "ingestion" / "raw_motif_json_build_state.json"
    )
    cli_path = _write_fake_motif_cli(tmp_path / "tools" / "motif-cli")
    _write_source_file(source_root / "Artist A" / "Alpha.gp5")

    dataset = MotifJsonCorpusDataset(
        filepath=str(raw_root),
        autobuild={
            "source_filepath": str(source_root),
            "cli_filepath": str(cli_path),
            "build_state_filepath": str(build_state_path),
        },
    )

    first_documents = dataset.load()
    stored_state = json.loads(build_state_path.read_text(encoding="utf-8"))
    stored_state["source_filepath"] = "data/00_corpus"
    stored_state["cli_filepath"] = "tools/motif-cli"
    stored_state["output_filepath"] = "data/01_raw/motif_json"
    build_state_path.write_text(json.dumps(stored_state), encoding="utf-8")

    second_documents = dataset.load()

    assert [document.relative_path for document in first_documents] == [
        "Artist A/Alpha.json"
    ]
    assert [document.relative_path for document in second_documents] == [
        "Artist A/Alpha.json"
    ]
    assert (
        _read_fake_motif_cli_invocations(cli_path) == EXPECTED_INITIAL_AUTOBUILD_COUNT
    )


def test_motif_json_corpus_dataset_rebuilds_when_cli_binary_changes(tmp_path):
    source_root = tmp_path / "00_corpus"
    raw_root = tmp_path / "01_raw" / "motif_json"
    build_state_path = (
        tmp_path / "02_intermediate" / "ingestion" / "raw_motif_json_build_state.json"
    )
    cli_path = _write_fake_motif_cli(tmp_path / "tools" / "motif-cli")
    _write_source_file(source_root / "Artist A" / "Alpha.gp5")

    dataset = MotifJsonCorpusDataset(
        filepath=str(raw_root),
        autobuild={
            "source_filepath": str(source_root),
            "cli_filepath": str(cli_path),
            "build_state_filepath": str(build_state_path),
        },
    )

    dataset.load()
    cli_path.write_text(
        f"{cli_path.read_text(encoding='utf-8')}\n# hash changes should rebuild\n",
        encoding="utf-8",
    )
    cli_path.chmod(0o755)

    documents = dataset.load()

    assert [document.relative_path for document in documents] == ["Artist A/Alpha.json"]
    assert (
        _read_fake_motif_cli_invocations(cli_path) == EXPECTED_REBUILD_AUTOBUILD_COUNT
    )


def test_motif_json_corpus_dataset_rebuilds_and_cleans_stale_outputs_when_source_changes(
    tmp_path,
):
    source_root = tmp_path / "00_corpus"
    raw_root = tmp_path / "01_raw" / "motif_json"
    build_state_path = (
        tmp_path / "02_intermediate" / "ingestion" / "raw_motif_json_build_state.json"
    )
    cli_path = _write_fake_motif_cli(tmp_path / "tools" / "motif-cli")
    alpha_source_path = source_root / "Artist A" / "Alpha.gp5"
    _write_source_file(alpha_source_path)

    dataset = MotifJsonCorpusDataset(
        filepath=str(raw_root),
        autobuild={
            "source_filepath": str(source_root),
            "cli_filepath": str(cli_path),
            "build_state_filepath": str(build_state_path),
        },
    )

    dataset.load()
    alpha_output_path = raw_root / "Artist A" / "Alpha.json"
    assert alpha_output_path.exists()

    alpha_source_path.unlink()
    _write_source_file(source_root / "Artist B" / "Beta.gp5")

    documents = dataset.load()

    assert [document.relative_path for document in documents] == ["Artist B/Beta.json"]
    assert not alpha_output_path.exists()
    assert (
        _read_fake_motif_cli_invocations(cli_path) == EXPECTED_REBUILD_AUTOBUILD_COUNT
    )


def test_motif_json_corpus_dataset_auto_build_requires_motif_cli(tmp_path):
    source_root = tmp_path / "00_corpus"
    _write_source_file(source_root / "Artist A" / "Alpha.gp5")

    dataset = MotifJsonCorpusDataset(
        filepath=str(tmp_path / "01_raw" / "motif_json"),
        autobuild={
            "source_filepath": str(source_root),
            "cli_filepath": str(tmp_path / "tools" / "motif-cli"),
            "build_state_filepath": str(
                tmp_path
                / "02_intermediate"
                / "ingestion"
                / "raw_motif_json_build_state.json"
            ),
        },
    )

    with pytest.raises(DatasetError, match="Motif CLI binary was not found"):
        dataset.load()


def _write_source_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("source", encoding="utf-8")


def _write_fake_motif_cli(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys
            from pathlib import Path

            counter_path = Path(__file__).with_name("motif_cli_invocations.txt")
            count = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
            counter_path.write_text(str(count + 1), encoding="utf-8")

            arguments = sys.argv[1:]
            input_dir = Path(arguments[arguments.index("--batch-input-dir") + 1])
            output_dir = Path(arguments[arguments.index("--batch-output-dir") + 1])

            for source_path in sorted(path for path in input_dir.rglob("*") if path.is_file()):
                relative_path = source_path.relative_to(input_dir)
                target_path = output_dir / relative_path.with_suffix(".json")
                artist = None if relative_path.parent == Path(".") else relative_path.parent.as_posix()
                payload = {
                    "Title": source_path.stem,
                    "Artist": artist,
                    "Album": None,
                    "Tracks": [{"Name": "Lead"}],
                    "PlaybackMasterBarSequence": [{}],
                }
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(json.dumps(payload), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _read_fake_motif_cli_invocations(cli_path: Path) -> int:
    counter_path = cli_path.with_name("motif_cli_invocations.txt")
    return int(counter_path.read_text(encoding="utf-8"))
