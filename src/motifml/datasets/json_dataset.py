"""Small JSON dataset helpers for Kedro."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset


class JsonDataset(AbstractDataset[Any, Any]):
    """Persist JSON-serializable Python objects to a local file."""

    def __init__(self, filepath: str, indent: int = 2) -> None:
        self._filepath = Path(filepath)
        self._indent = indent

    def load(self) -> Any:
        """Load JSON content from disk."""
        with self._filepath.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def save(self, data: Any) -> None:
        """Save JSON content to disk."""
        serializable = _to_json_compatible(data)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._filepath.open("w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=self._indent, ensure_ascii=True)
            stream.write("\n")

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": self._filepath.as_posix(),
            "indent": self._indent,
        }


def _to_json_compatible(data: Any) -> Any:
    """Convert common Python containers and dataclasses into JSON-safe values."""
    if is_dataclass(data):
        return _to_json_compatible(asdict(data))

    if isinstance(data, dict):
        return {str(key): _to_json_compatible(value) for key, value in data.items()}

    if isinstance(data, list | tuple | set):
        return [_to_json_compatible(value) for value in data]

    if isinstance(data, Path):
        return data.as_posix()

    return data
