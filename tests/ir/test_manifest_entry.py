"""Tests for IR manifest entries and catalog wiring."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from motifml.ir.models import IrManifestEntry

CATALOG_PATH = Path(__file__).resolve().parents[2] / "conf" / "base" / "catalog.yml"


def test_ir_manifest_entry_normalizes_count_maps_and_is_json_serializable():
    entry = IrManifestEntry(
        source_path=" data/00_corpus/example.json ",
        source_hash=" abc123 ",
        ir_document_path=" data/02_intermediate/ir/documents/example.json.ir.json ",
        build_timestamp=" 2026-03-18T15:00:00-04:00 ",
        node_counts={"NoteEvent": 3, "Bar": 1},
        edge_counts={"next_in_voice": 2, "contains": 8},
        unsupported_features_dropped=("tempo changes", "slide links"),
    )

    assert entry.source_path == "data/00_corpus/example.json"
    assert entry.source_hash == "abc123"
    assert (
        entry.ir_document_path
        == "data/02_intermediate/ir/documents/example.json.ir.json"
    )
    assert entry.build_timestamp == "2026-03-18T15:00:00-04:00"
    assert entry.node_counts == {"Bar": 1, "NoteEvent": 3}
    assert entry.edge_counts == {"contains": 8, "next_in_voice": 2}
    assert entry.unsupported_features_dropped == (
        "slide links",
        "tempo changes",
    )

    serialized = json.loads(json.dumps(asdict(entry)))

    assert serialized == {
        "source_path": "data/00_corpus/example.json",
        "source_hash": "abc123",
        "ir_document_path": "data/02_intermediate/ir/documents/example.json.ir.json",
        "build_timestamp": "2026-03-18T15:00:00-04:00",
        "node_counts": {"Bar": 1, "NoteEvent": 3},
        "edge_counts": {"contains": 8, "next_in_voice": 2},
        "unsupported_features_dropped": ["slide links", "tempo changes"],
    }


def test_ir_catalog_registers_the_manifest_and_reporting_datasets():
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    assert catalog["motif_ir_manifest"] == {
        "type": "motifml.datasets.json_dataset.JsonDataset",
        "filepath": "data/02_intermediate/ir/motif_ir_manifest.json",
    }
    assert catalog["motif_ir_validation_report"] == {
        "type": "motifml.datasets.json_dataset.JsonDataset",
        "filepath": "data/08_reporting/ir/motif_ir_validation_report.json",
    }
    assert catalog["motif_ir_summary"] == {
        "type": "motifml.datasets.json_dataset.JsonDataset",
        "filepath": "data/08_reporting/ir/motif_ir_summary.json",
    }
    assert catalog["motif_ir_corpus"] == {
        "type": "motifml.datasets.motif_ir_corpus_dataset.MotifIrCorpusDataset",
        "filepath": "data/02_intermediate/ir/documents",
    }
