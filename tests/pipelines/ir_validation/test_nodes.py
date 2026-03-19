"""Tests for IR validation pipeline nodes."""

from __future__ import annotations

from pathlib import Path

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.serialization import deserialize_document
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_validation.nodes import validate_ir_documents

GOLDEN_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "ir" / "golden"


def test_validate_ir_documents_returns_reports_in_relative_path_order():
    document = deserialize_document(
        (GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json").read_text(
            encoding="utf-8"
        )
    )

    reports = validate_ir_documents(
        [
            MotifIrDocumentRecord(
                relative_path="zeta/document.json", document=document
            ),
            MotifIrDocumentRecord(
                relative_path="alpha/document.json", document=document
            ),
        ]
    )

    assert [report.relative_path for report in reports] == [
        "alpha/document.json",
        "zeta/document.json",
    ]
    assert all(report.passed for report in reports)


def test_validate_ir_documents_applies_rule_severity_overrides():
    document = deserialize_document(
        (GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json").read_text(
            encoding="utf-8"
        )
    )
    object.__setattr__(document.note_events[0], "time", ScoreTime(1, 8))

    reports = validate_ir_documents(
        [
            MotifIrDocumentRecord(
                relative_path="alpha/document.json",
                document=document,
            )
        ],
        ir_validation={
            "rule_severities": {
                "note_time_alignment": "warn",
            }
        },
    )

    assert len(reports) == 1
    assert reports[0].passed is True
    assert reports[0].error_count == 0
    assert reports[0].warning_count == 1
