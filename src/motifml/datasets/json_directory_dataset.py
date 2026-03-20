"""Dataset helpers for reading a directory of JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError


class JsonDirectoryDataset(AbstractDataset[None, list[Any]]):
    """Load all JSON files under a directory tree in deterministic path order."""

    def __init__(self, filepath: str, glob_pattern: str = "**/*.json") -> None:
        self._filepath = Path(filepath)
        self._glob_pattern = glob_pattern

    def load(self) -> list[Any]:
        """Load every matching JSON file from disk."""
        if not self._filepath.exists():
            return []

        loaded: list[Any] = []
        for path in sorted(self._filepath.glob(self._glob_pattern)):
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as stream:
                loaded.append(json.load(stream))

        return loaded

    def save(self, data: None) -> None:
        """Reject writes because reducer inputs are produced elsewhere."""
        del data
        raise DatasetError("JsonDirectoryDataset is read-only.")

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "glob_pattern": self._glob_pattern,
        }
