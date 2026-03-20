"""Kedro dataset for persisting a corpus of MotifML IR documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError

from motifml.ir.models import MotifMlIrDocument
from motifml.ir.serialization import deserialize_document, serialize_document


@dataclass(frozen=True, slots=True)
class MotifIrDocumentRecord:
    """One IR document paired with its stable source-relative identity."""

    relative_path: str
    document: MotifMlIrDocument


class MotifIrCorpusDataset(
    AbstractDataset[list[MotifIrDocumentRecord], list[MotifIrDocumentRecord]]
):
    """Persist a directory tree of canonical IR JSON documents."""

    def __init__(self, filepath: str, glob_pattern: str = "**/*.ir.json") -> None:
        self._filepath = Path(filepath)
        self._glob_pattern = glob_pattern

    def load(self) -> list[MotifIrDocumentRecord]:
        """Load the IR corpus from disk in deterministic path order."""
        if not self._filepath.exists():
            raise DatasetError(
                f"Motif IR corpus directory does not exist: {self._filepath.as_posix()}"
            )

        records: list[MotifIrDocumentRecord] = []
        for path in sorted(self._filepath.glob(self._glob_pattern)):
            if not path.is_file():
                continue

            relative_artifact_path = path.relative_to(self._filepath).as_posix()
            records.append(
                MotifIrDocumentRecord(
                    relative_path=source_relative_path_from_ir_path(
                        relative_artifact_path
                    ),
                    document=deserialize_document(path.read_bytes()),
                )
            )

        return records

    def save(self, data: list[MotifIrDocumentRecord]) -> None:
        """Persist the IR corpus to disk without rewriting unchanged files."""
        self._filepath.mkdir(parents=True, exist_ok=True)

        for record in sorted(data, key=lambda item: item.relative_path.casefold()):
            target_path = self._filepath / ir_artifact_path_for_source(
                record.relative_path
            )
            serialized_bytes = serialize_document(record.document).encode("utf-8")
            if target_path.exists():
                if target_path.stat().st_size == len(serialized_bytes) and (
                    target_path.read_bytes() == serialized_bytes
                ):
                    continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(serialized_bytes)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "glob_pattern": self._glob_pattern,
        }


def ir_artifact_path_for_source(relative_path: str) -> str:
    """Map a raw-corpus relative path to the persisted IR artifact path."""
    normalized = _normalize_relative_path(relative_path)
    return Path(f"{normalized.as_posix()}.ir.json").as_posix()


def source_relative_path_from_ir_path(relative_ir_path: str) -> str:
    """Recover the source-relative identity from a persisted IR artifact path."""
    normalized = _normalize_relative_path(relative_ir_path)
    normalized_text = normalized.as_posix()
    if normalized_text.endswith(".ir.json"):
        return normalized_text[: -len(".ir.json")]

    raise DatasetError(
        "Motif IR artifact paths must end with '.ir.json', "
        f"but received '{relative_ir_path}'."
    )


def _normalize_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise DatasetError("Relative corpus paths must not be absolute.")

    if any(part == ".." for part in path.parts):
        raise DatasetError("Relative corpus paths must not escape the dataset root.")

    if str(path) in {"", "."}:
        raise DatasetError("Relative corpus paths must point to a file-like location.")

    return path
