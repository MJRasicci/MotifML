"""Kedro dataset for loading a Motif JSON corpus from the raw data stage."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

from kedro.io import AbstractDataset, DatasetError

LOGGER = logging.getLogger(__name__)


class RawMotifTrack(TypedDict, total=False):
    """Subset of Motif's raw mapped JSON track structure needed for ingestion."""

    Name: str


class RawMotifScore(TypedDict, total=False):
    """Subset of Motif's raw mapped JSON score structure needed for ingestion."""

    Title: str
    Artist: str
    Album: str
    Tracks: list[RawMotifTrack]
    PlaybackMasterBarSequence: list[object]


@dataclass(frozen=True, slots=True)
class MotifJsonDocument:
    """A single raw Motif JSON document loaded from the corpus."""

    relative_path: str
    sha256: str
    file_size_bytes: int
    score: RawMotifScore


@dataclass(frozen=True, slots=True)
class SourceCorpusFile:
    """A deterministic fingerprint entry for one source file in `data/00_corpus`."""

    relative_path: str
    file_size_bytes: int
    sha256: str
    mtime_ns: int | None = None


@dataclass(frozen=True, slots=True)
class RawCorpusBuildState:
    """Stable metadata used to decide whether the raw corpus needs rebuilding."""

    cli_filepath: str
    cli_sha256: str
    source_filepath: str
    source_glob_pattern: str
    output_filepath: str
    source_files: tuple[SourceCorpusFile, ...]


class MotifJsonCorpusDataset(AbstractDataset[None, list[MotifJsonDocument]]):
    """Load a directory tree of raw Motif JSON files as immutable documents."""

    def __init__(
        self,
        filepath: str,
        glob_pattern: str = "**/*.json",
        autobuild: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = Path(filepath)
        self._glob_pattern = glob_pattern
        self._auto_build = autobuild is not None

        if autobuild is None:
            self._source_filepath = None
            self._source_glob_pattern = "**/*"
            self._cli_filepath = None
            self._build_state_filepath = None
            self._clean_output_on_rebuild = True
            return

        try:
            self._source_filepath = Path(str(autobuild["source_filepath"]))
            self._source_glob_pattern = str(
                autobuild.get("source_glob_pattern", "**/*")
            )
            self._cli_filepath = Path(str(autobuild["cli_filepath"]))
            self._build_state_filepath = Path(str(autobuild["build_state_filepath"]))
        except KeyError as exc:
            raise ValueError(
                "MotifJsonCorpusDataset autobuild requires source_filepath, "
                "cli_filepath, and build_state_filepath."
            ) from exc

        self._clean_output_on_rebuild = bool(
            autobuild.get("clean_output_on_rebuild", True)
        )

    def load(self) -> list[MotifJsonDocument]:
        """Load the raw Motif JSON corpus from disk."""
        if self._auto_build:
            self._ensure_raw_corpus_built()

        if not self._filepath.exists():
            raise DatasetError(
                f"Raw Motif JSON corpus directory does not exist: {self._filepath.as_posix()}"
            )

        documents: list[MotifJsonDocument] = []
        for path in sorted(self._filepath.glob(self._glob_pattern)):
            if not path.is_file():
                continue

            payload = path.read_bytes()
            loaded = json.loads(payload)
            if not isinstance(loaded, dict):
                raise DatasetError(
                    "Motif JSON documents must deserialize to objects, "
                    f"but '{path.as_posix()}' did not."
                )

            documents.append(
                MotifJsonDocument(
                    relative_path=path.relative_to(self._filepath).as_posix(),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    file_size_bytes=len(payload),
                    score=cast(RawMotifScore, loaded),
                )
            )

        return documents

    def save(self, data: None) -> None:
        """Reject writes because the raw corpus is source data."""
        raise DatasetError(
            "MotifJsonCorpusDataset is read-only. Populate `data/00_corpus` and run "
            "the pipeline to build raw data with the Motif CLI."
        )

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "glob_pattern": self._glob_pattern,
            "auto_build": self._auto_build,
            "source_filepath": (
                self._source_filepath.as_posix()
                if self._source_filepath is not None
                else None
            ),
            "source_glob_pattern": self._source_glob_pattern,
            "cli_filepath": (
                self._cli_filepath.as_posix()
                if self._cli_filepath is not None
                else None
            ),
            "build_state_filepath": (
                self._build_state_filepath.as_posix()
                if self._build_state_filepath is not None
                else None
            ),
            "clean_output_on_rebuild": self._clean_output_on_rebuild,
        }

    def _ensure_raw_corpus_built(self) -> None:
        previous_state = self._load_build_state()
        current_state = self._build_current_state(previous_state)

        if (
            previous_state is not None
            and self._raw_corpus_inputs_match(previous_state, current_state)
            and self._has_raw_output()
        ):
            return

        LOGGER.info(
            "Building raw Motif JSON corpus from '%s' with '%s'.",
            self._source_filepath,
            self._cli_filepath,
        )
        self._run_motif_cli()
        if not self._has_raw_output():
            raise DatasetError(
                "Motif CLI completed without producing any raw Motif JSON files in "
                f"'{self._filepath.as_posix()}'."
            )

        self._write_build_state(current_state)

    @staticmethod
    def _raw_corpus_inputs_match(
        previous_state: RawCorpusBuildState,
        current_state: RawCorpusBuildState,
    ) -> bool:
        """Compare only the inputs that affect raw Motif JSON output."""

        return (
            previous_state.cli_sha256 == current_state.cli_sha256
            and _stable_source_file_fingerprints(previous_state.source_files)
            == _stable_source_file_fingerprints(current_state.source_files)
        )

    def _build_current_state(
        self,
        previous_state: RawCorpusBuildState | None,
    ) -> RawCorpusBuildState:
        if self._source_filepath is None or self._cli_filepath is None:
            raise DatasetError("Auto-build paths must be configured before loading.")

        source_root = self._source_filepath
        if not source_root.exists():
            raise DatasetError(
                f"Source corpus directory does not exist: {source_root.as_posix()}"
            )

        cli_path = self._resolve_cli_path(self._cli_filepath)
        if not cli_path.is_file():
            raise DatasetError(
                "Motif CLI binary was not found. Place `motif-cli` in `tools/` "
                f"or update the catalog config. Tried: {cli_path.as_posix()}"
            )

        previous_source_files_by_path = (
            {entry.relative_path: entry for entry in previous_state.source_files}
            if previous_state is not None
            else {}
        )
        source_files = tuple(
            _build_source_corpus_file(
                path=path,
                source_root=source_root,
                previous_entry=previous_source_files_by_path.get(
                    path.relative_to(source_root).as_posix()
                ),
            )
            for path in sorted(source_root.glob(self._source_glob_pattern))
            if path.is_file()
        )
        if not source_files:
            raise DatasetError(
                "Source corpus directory does not contain any files. Place your source "
                f"music files in '{source_root.as_posix()}' before running Kedro."
            )

        return RawCorpusBuildState(
            cli_filepath=cli_path.as_posix(),
            cli_sha256=_hash_file(cli_path),
            source_filepath=source_root.as_posix(),
            source_glob_pattern=self._source_glob_pattern,
            output_filepath=self._filepath.as_posix(),
            source_files=source_files,
        )

    def _resolve_cli_path(self, cli_path: Path) -> Path:
        if cli_path.exists():
            return cli_path

        windows_variant = cli_path.with_suffix(".exe")
        if windows_variant.exists():
            return windows_variant

        return cli_path

    def _load_build_state(self) -> RawCorpusBuildState | None:
        if (
            self._build_state_filepath is None
            or not self._build_state_filepath.exists()
        ):
            return None

        try:
            with self._build_state_filepath.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)

            source_files = tuple(
                SourceCorpusFile(
                    relative_path=str(entry["relative_path"]),
                    file_size_bytes=int(entry["file_size_bytes"]),
                    sha256=str(entry["sha256"]),
                    mtime_ns=(
                        int(entry["mtime_ns"])
                        if entry.get("mtime_ns") is not None
                        else None
                    ),
                )
                for entry in loaded["source_files"]
            )

            return RawCorpusBuildState(
                cli_filepath=str(loaded["cli_filepath"]),
                cli_sha256=str(loaded["cli_sha256"]),
                source_filepath=str(loaded["source_filepath"]),
                source_glob_pattern=str(loaded["source_glob_pattern"]),
                output_filepath=str(loaded["output_filepath"]),
                source_files=source_files,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.warning(
                "Ignoring invalid raw corpus build state at '%s'.",
                self._build_state_filepath.as_posix(),
            )
            return None

    def _write_build_state(self, state: RawCorpusBuildState) -> None:
        if self._build_state_filepath is None:
            raise DatasetError("Build state path must be configured before loading.")

        serializable = {
            "cli_filepath": state.cli_filepath,
            "cli_sha256": state.cli_sha256,
            "source_filepath": state.source_filepath,
            "source_glob_pattern": state.source_glob_pattern,
            "output_filepath": state.output_filepath,
            "source_files": [
                {
                    "relative_path": entry.relative_path,
                    "file_size_bytes": entry.file_size_bytes,
                    "sha256": entry.sha256,
                    "mtime_ns": entry.mtime_ns,
                }
                for entry in state.source_files
            ],
        }

        self._build_state_filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._build_state_filepath.open("w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=2, ensure_ascii=True)
            stream.write("\n")

    def _has_raw_output(self) -> bool:
        if not self._filepath.exists():
            return False

        return any(path.is_file() for path in self._filepath.glob(self._glob_pattern))

    def _run_motif_cli(self) -> None:
        if self._source_filepath is None or self._cli_filepath is None:
            raise DatasetError("Auto-build paths must be configured before loading.")

        cli_path = self._resolve_cli_path(self._cli_filepath)
        if self._clean_output_on_rebuild and self._filepath.exists():
            if self._filepath.is_dir():
                shutil.rmtree(self._filepath)
            else:
                self._filepath.unlink()

        self._filepath.mkdir(parents=True, exist_ok=True)

        completed = subprocess.run(
            [
                str(cli_path),
                "--batch-input-dir",
                str(self._source_filepath),
                "--batch-output-dir",
                str(self._filepath),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            message_parts = [
                "Motif CLI failed while building the raw Motif JSON corpus.",
                f"Command: {completed.args}",
            ]
            if completed.stdout:
                message_parts.append(f"stdout:\n{completed.stdout.strip()}")
            if completed.stderr:
                message_parts.append(f"stderr:\n{completed.stderr.strip()}")

            raise DatasetError("\n".join(message_parts))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _build_source_corpus_file(
    *,
    path: Path,
    source_root: Path,
    previous_entry: SourceCorpusFile | None,
) -> SourceCorpusFile:
    relative_path = path.relative_to(source_root).as_posix()
    stat_result = path.stat()
    file_size_bytes = stat_result.st_size
    mtime_ns = stat_result.st_mtime_ns
    if (
        previous_entry is not None
        and previous_entry.file_size_bytes == file_size_bytes
        and previous_entry.mtime_ns == mtime_ns
    ):
        sha256 = previous_entry.sha256
    else:
        sha256 = _hash_file(path)

    return SourceCorpusFile(
        relative_path=relative_path,
        file_size_bytes=file_size_bytes,
        sha256=sha256,
        mtime_ns=mtime_ns,
    )


def _stable_source_file_fingerprints(
    source_files: tuple[SourceCorpusFile, ...],
) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (entry.relative_path, entry.file_size_bytes, entry.sha256)
        for entry in source_files
    )
