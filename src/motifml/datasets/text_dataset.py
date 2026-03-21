"""Kedro dataset for small UTF-8 text artifacts such as Markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError


class TextDataset(AbstractDataset[str, str]):
    """Persist one UTF-8 text artifact to a configured file path."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> str:
        """Load one text artifact from disk."""
        if not self._filepath.exists():
            raise DatasetError(
                f"Text artifact does not exist: {self._filepath.as_posix()}."
            )
        return self._filepath.read_text(encoding="utf-8")

    def save(self, data: str) -> None:
        """Persist one text artifact to disk."""
        if not isinstance(data, str):
            raise DatasetError("TextDataset.save expects a text string.")
        serialized = data.encode("utf-8")
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        if self._filepath.exists() and self._filepath.read_bytes() == serialized:
            return
        self._filepath.write_bytes(serialized)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath.as_posix()}


__all__ = ["TextDataset"]
