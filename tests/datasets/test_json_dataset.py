from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from motifml.datasets.json_dataset import JsonDataset


@dataclass(frozen=True)
class _SampleRecord:
    name: str
    values: tuple[int, ...]


def test_json_dataset_saves_dataclasses_without_preconverting(tmp_path: Path) -> None:
    dataset = JsonDataset(filepath=str(tmp_path / "sample.json"))

    dataset.save(
        {
            "record": _SampleRecord(name="example", values=(1, 2, 3)),
            "paths": [Path("alpha/beta")],
            "tags": {"x", "y"},
        }
    )

    payload = dataset.load()
    assert payload["record"] == {"name": "example", "values": [1, 2, 3]}
    assert payload["paths"] == ["alpha/beta"]
    assert sorted(payload["tags"]) == ["x", "y"]


def test_json_dataset_falls_back_for_non_string_mapping_keys(tmp_path: Path) -> None:
    dataset = JsonDataset(filepath=str(tmp_path / "fallback.json"))

    dataset.save({Path("alpha"): "value"})

    assert dataset.load() == {"alpha": "value"}
