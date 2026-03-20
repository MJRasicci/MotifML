"""Tests for the read-only JSON directory dataset."""

from __future__ import annotations

import json

from motifml.datasets.json_directory_dataset import JsonDirectoryDataset


def test_json_directory_dataset_loads_payloads_in_path_order(tmp_path):
    root = tmp_path / "payloads"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir(parents=True)
    (root / "b" / "two.json").write_text(json.dumps({"name": "two"}), encoding="utf-8")
    (root / "a" / "one.json").write_text(json.dumps({"name": "one"}), encoding="utf-8")

    dataset = JsonDirectoryDataset(filepath=str(root))

    assert dataset.load() == [{"name": "one"}, {"name": "two"}]
