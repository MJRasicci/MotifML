"""Tests for IR validation pipeline nodes."""

from __future__ import annotations

from pathlib import Path

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.serialization import deserialize_document
from motifml.ir.summary import (
    IrCorpusDocumentSummary,
    IrCorpusSummary,
    IrNamedCount,
    IrOptionalFamilyPresence,
    IrRuleIssueCount,
)
from motifml.ir.time import ScoreTime
from motifml.ir.validation import build_document_validation_report
from motifml.pipelines.ir_validation.nodes import (
    merge_ir_shard_summaries,
    merge_ir_validation_report_fragments,
    report_ir_scale_metrics,
    summarize_ir_corpus,
    validate_ir_documents,
)

GOLDEN_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "ir" / "golden"
EXPECTED_DOCUMENT_COUNT = 2
EXPECTED_MERGED_DOCUMENT_COUNT = 2
EXPECTED_TOTAL_DERIVED_VIEWS = 2


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


def test_merge_ir_validation_report_fragments_orders_reports_by_relative_path():
    merged = merge_ir_validation_report_fragments(
        [
            [{"relative_path": "zeta/document.json", "rule_reports": []}],
            [{"relative_path": "alpha/document.json", "rule_reports": []}],
        ]
    )

    assert [report["relative_path"] for report in merged] == [
        "alpha/document.json",
        "zeta/document.json",
    ]


def test_merge_ir_shard_summaries_combines_counts_without_reloading_documents():
    merged = merge_ir_shard_summaries(
        [
            IrCorpusSummary(
                document_count=1,
                per_document=(
                    IrCorpusDocumentSummary(
                        relative_path="alpha/document.json",
                        node_counts={"NoteEvent": 2},
                        edge_counts={"contains": 1},
                    ),
                ),
                validation_issue_counts_by_rule=(
                    IrRuleIssueCount(
                        rule="note_time_alignment",
                        error_count=0,
                        warning_count=1,
                    ),
                ),
                optional_family_presence=IrOptionalFamilyPresence(
                    total_phrase_spans=1,
                    documents_with_phrase_spans=1,
                ),
                unsupported_feature_counts=(
                    IrNamedCount(name="unsupported_a", count=2),
                ),
            ),
            IrCorpusSummary(
                document_count=1,
                per_document=(
                    IrCorpusDocumentSummary(
                        relative_path="beta/document.json",
                        node_counts={"NoteEvent": 4},
                        edge_counts={"contains": 3},
                    ),
                ),
                validation_issue_counts_by_rule=(
                    IrRuleIssueCount(
                        rule="note_time_alignment",
                        error_count=1,
                        warning_count=0,
                    ),
                ),
                optional_family_presence=IrOptionalFamilyPresence(
                    total_derived_views=2,
                    documents_with_derived_views=1,
                ),
                unsupported_feature_counts=(
                    IrNamedCount(name="unsupported_b", count=1),
                ),
            ),
        ]
    )

    assert merged.document_count == EXPECTED_MERGED_DOCUMENT_COUNT
    assert [item.relative_path for item in merged.per_document] == [
        "alpha/document.json",
        "beta/document.json",
    ]
    assert merged.validation_issue_counts_by_rule[0].error_count == 1
    assert merged.validation_issue_counts_by_rule[0].warning_count == 1
    assert merged.optional_family_presence.total_phrase_spans == 1
    assert (
        merged.optional_family_presence.total_derived_views
        == EXPECTED_TOTAL_DERIVED_VIEWS
    )
    assert [item.name for item in merged.unsupported_feature_counts] == [
        "unsupported_a",
        "unsupported_b",
    ]
