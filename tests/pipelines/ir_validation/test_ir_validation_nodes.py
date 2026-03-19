"""Tests for IR validation pipeline nodes."""

from __future__ import annotations

from pathlib import Path

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.serialization import deserialize_document
from motifml.ir.time import ScoreTime
from motifml.ir.validation import build_document_validation_report
from motifml.pipelines.ir_validation.nodes import (
    report_ir_scale_metrics,
    summarize_ir_corpus,
    validate_ir_documents,
)

GOLDEN_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "ir" / "golden"
EXPECTED_DOCUMENT_COUNT = 2


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


def test_summarize_ir_corpus_rolls_up_scale_and_feature_metrics():
    alpha_document = deserialize_document(
        (GOLDEN_FIXTURE_ROOT / "single_track_monophonic_pickup.ir.json").read_text(
            encoding="utf-8"
        )
    )
    zeta_document = deserialize_document(
        (GOLDEN_FIXTURE_ROOT / "ensemble_polyphony_controls.ir.json").read_text(
            encoding="utf-8"
        )
    )
    object.__setattr__(zeta_document.note_events[0], "time", ScoreTime(1, 8))

    records = [
        MotifIrDocumentRecord(
            relative_path="zeta/document.json", document=zeta_document
        ),
        MotifIrDocumentRecord(
            relative_path="alpha/document.json", document=alpha_document
        ),
    ]
    validation_reports = [
        build_document_validation_report(
            relative_path="alpha/document.json",
            source_hash=alpha_document.metadata.source_document_hash,
            document=alpha_document,
        ),
        build_document_validation_report(
            relative_path="zeta/document.json",
            source_hash=zeta_document.metadata.source_document_hash,
            document=zeta_document,
            rule_severities={"note_time_alignment": "warn"},
        ),
    ]
    manifest_entries = [
        {
            "source_path": "alpha/document.json",
            "source_hash": "alpha",
            "ir_document_path": "data/02_intermediate/ir/documents/alpha/document.json.ir.json",
            "build_timestamp": "2026-03-19T00:00:00-04:00",
            "node_counts": {"NoteEvent": len(alpha_document.note_events)},
            "edge_counts": {"contains": len(alpha_document.edges)},
            "unsupported_features_dropped": ["unsupported_onset_technique"],
            "conversion_diagnostics": [
                {
                    "category": "unsupported",
                    "severity": "warning",
                    "code": "unsupported_onset_technique",
                    "count": 2,
                    "paths": ["tracks[0].beats[0].brush"],
                    "messages": ["unsupported brush technique"],
                }
            ],
        },
        {
            "source_path": "zeta/document.json",
            "source_hash": "zeta",
            "ir_document_path": "data/02_intermediate/ir/documents/zeta/document.json.ir.json",
            "build_timestamp": "2026-03-19T00:00:00-04:00",
            "node_counts": {"NoteEvent": len(zeta_document.note_events)},
            "edge_counts": {"contains": len(zeta_document.edges)},
            "unsupported_features_dropped": ["open_ended_span_control"],
            "conversion_diagnostics": [
                {
                    "category": "excluded",
                    "severity": "warning",
                    "code": "open_ended_span_control",
                    "count": 1,
                    "paths": ["spanControls[0].end"],
                    "messages": ["open-ended span skipped"],
                }
            ],
        },
    ]

    summary = summarize_ir_corpus(records, validation_reports, manifest_entries)
    reported_summary = report_ir_scale_metrics(summary)

    assert summary.document_count == EXPECTED_DOCUMENT_COUNT
    assert [item.relative_path for item in summary.per_document] == [
        "alpha/document.json",
        "zeta/document.json",
    ]
    assert any(item.family == "NoteEvent" for item in summary.node_count_distributions)
    assert summary.optional_family_presence.total_phrase_spans == 0
    assert summary.optional_family_presence.total_derived_views == 0
    assert summary.validation_issue_counts_by_rule[0].rule == "note_time_alignment"
    assert summary.validation_issue_counts_by_rule[0].warning_count == 1
    assert [item.name for item in summary.unsupported_feature_counts] == [
        "open_ended_span_control",
        "unsupported_onset_technique",
    ]
    assert reported_summary.scale_report is not None
    assert f"Documents: {EXPECTED_DOCUMENT_COUNT}" in reported_summary.scale_report
    assert "unsupported_onset_technique: 2" in reported_summary.scale_report
