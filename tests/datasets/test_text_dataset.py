"""Tests for the small UTF-8 text Kedro dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
from kedro.io import DatasetError

from motifml.datasets.text_dataset import TextDataset


def test_text_dataset_round_trips_markdown_text(tmp_path: Path) -> None:
    dataset = TextDataset(filepath=str(tmp_path / "report.md"))

    dataset.save("# Report\n")

    assert dataset.load() == "# Report\n"


def test_text_dataset_rejects_non_text_payloads(tmp_path: Path) -> None:
    dataset = TextDataset(filepath=str(tmp_path / "report.md"))

    with pytest.raises(DatasetError, match="expects a text string"):
        dataset.save({"not": "text"})  # type: ignore[arg-type]
