"""Tests for the IR corpus Kedro dataset."""

from __future__ import annotations

import time
from pathlib import Path

from motifml.datasets.motif_ir_corpus_dataset import (
    MotifIrCorpusDataset,
    MotifIrDocumentRecord,
    ir_artifact_path_for_source,
)
from motifml.ir.serialization import deserialize_document

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "ir"
    / "representative_document.ir.json"
)
EXPECTED_RECORD_COUNT = 2


def test_motif_ir_corpus_dataset_saves_and_loads_documents_in_path_order(tmp_path):
    dataset = MotifIrCorpusDataset(filepath=str(tmp_path / "02_intermediate" / "ir"))
    document = deserialize_document(FIXTURE_PATH.read_text(encoding="utf-8"))

    dataset.save(
        [
            MotifIrDocumentRecord(relative_path="b/Beta.json", document=document),
            MotifIrDocumentRecord(relative_path="a/Alpha.json", document=document),
        ]
    )

    loaded = dataset.load()

    assert len(loaded) == EXPECTED_RECORD_COUNT
    assert [record.relative_path for record in loaded] == [
        "a/Alpha.json",
        "b/Beta.json",
    ]
    assert loaded[0].document == document


def test_motif_ir_corpus_dataset_uses_stable_ir_artifact_paths():
    assert (
        ir_artifact_path_for_source("Artist A/Alpha.json")
        == "Artist A/Alpha.json.ir.json"
    )
    assert ir_artifact_path_for_source("Artist B/Beta") == "Artist B/Beta.ir.json"


def test_motif_ir_corpus_dataset_skips_rewriting_unchanged_documents(tmp_path):
    dataset = MotifIrCorpusDataset(filepath=str(tmp_path / "02_intermediate" / "ir"))
    document = deserialize_document(FIXTURE_PATH.read_text(encoding="utf-8"))
    record = MotifIrDocumentRecord(
        relative_path="Artist A/Alpha.json", document=document
    )

    dataset.save([record])
    artifact_path = (
        tmp_path / "02_intermediate" / "ir" / "Artist A" / "Alpha.json.ir.json"
    )
    first_mtime = artifact_path.stat().st_mtime_ns

    time.sleep(0.01)
    dataset.save([record])
    second_mtime = artifact_path.stat().st_mtime_ns

    assert first_mtime == second_mtime
