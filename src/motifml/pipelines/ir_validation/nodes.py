"""Nodes for validating canonical IR documents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from math import ceil, floor
from statistics import fmean
from typing import Any

from motifml.datasets.motif_ir_corpus_dataset import MotifIrDocumentRecord
from motifml.ir.summary import (
    IrCorpusDocumentSummary,
    IrCorpusSummary,
    IrCountDistribution,
    IrNamedCount,
    IrOptionalFamilyPresence,
    IrRuleIssueCount,
)
from motifml.ir.validation import (
    IrDocumentValidationReport,
    IrValidationSeverity,
    build_document_validation_report,
)


def validate_ir_documents(
    documents: list[MotifIrDocumentRecord],
    ir_validation: Mapping[str, Any] | None = None,
) -> list[IrDocumentValidationReport]:
    """Validate IR documents in deterministic path order."""
    rule_severities = _resolve_rule_severities(ir_validation)
    return [
        build_document_validation_report(
            relative_path=record.relative_path,
            source_hash=record.document.metadata.source_document_hash,
            document=record.document,
            rule_severities=rule_severities,
        )
        for record in sorted(documents, key=lambda item: item.relative_path.casefold())
    ]


def summarize_ir_corpus(
    documents: list[MotifIrDocumentRecord],
    validation_reports: list[IrDocumentValidationReport],
    manifest_entries: list[Mapping[str, Any]] | list[object],
) -> IrCorpusSummary:
    """Produce machine-readable corpus summary metrics."""
    sorted_documents = sorted(documents, key=lambda item: item.relative_path.casefold())
    per_document = tuple(
        IrCorpusDocumentSummary(
            relative_path=record.relative_path,
            node_counts=_count_document_node_families(record.document),
            edge_counts=_count_document_edge_families(record.document),
        )
        for record in sorted_documents
    )
    return IrCorpusSummary(
        document_count=len(sorted_documents),
        per_document=per_document,
        node_count_distributions=_build_node_count_distributions(per_document),
        validation_issue_counts_by_rule=_build_rule_issue_counts(validation_reports),
        optional_family_presence=_build_optional_family_presence(sorted_documents),
        unsupported_feature_counts=_build_unsupported_feature_counts(manifest_entries),
    )


def report_ir_scale_metrics(summary: IrCorpusSummary) -> IrCorpusSummary:
    """Attach a human-readable scale report to the machine-readable summary."""
    return replace(summary, scale_report=_render_scale_report(summary))


def _resolve_rule_severities(
    ir_validation: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if ir_validation is None:
        return None
    if not isinstance(ir_validation, Mapping):
        raise ValueError("ir_validation parameters must be a mapping.")

    rule_severities = ir_validation.get("rule_severities")
    if rule_severities is None:
        return None
    if not isinstance(rule_severities, Mapping):
        raise ValueError("ir_validation.rule_severities must be a mapping.")

    return rule_severities


def _count_document_node_families(document: object) -> dict[str, int]:
    if not hasattr(document, "parts"):
        raise ValueError("IR documents must expose canonical entity collections.")

    counts = {
        "Bar": len(document.bars),
        "NoteEvent": len(document.note_events),
        "OnsetGroup": len(document.onset_groups),
        "Part": len(document.parts),
        "PointControlEvent": len(document.point_control_events),
        "SpanControlEvent": len(document.span_control_events),
        "Staff": len(document.staves),
        "VoiceLane": len(document.voice_lanes),
    }
    if document.optional_overlays.phrase_spans:
        counts["PhraseSpan"] = len(document.optional_overlays.phrase_spans)

    return {name: count for name, count in counts.items() if count > 0}


def _count_document_edge_families(document: object) -> dict[str, int]:
    counts = Counter(edge.edge_type.value for edge in document.edges)
    return {edge_type: counts[edge_type] for edge_type in sorted(counts)}


def _build_node_count_distributions(
    per_document: tuple[IrCorpusDocumentSummary, ...],
) -> tuple[IrCountDistribution, ...]:
    families = sorted(
        {family for summary in per_document for family in summary.node_counts}
    )
    distributions = [
        IrCountDistribution(
            family=family,
            min=min(
                values := [
                    summary.node_counts.get(family, 0) for summary in per_document
                ]
            ),
            max=max(values),
            mean=fmean(values),
            p25=_percentile(values, 0.25),
            p50=_percentile(values, 0.50),
            p75=_percentile(values, 0.75),
            p95=_percentile(values, 0.95),
        )
        for family in families
    ]
    return tuple(distributions)


def _build_rule_issue_counts(
    validation_reports: list[IrDocumentValidationReport],
) -> tuple[IrRuleIssueCount, ...]:
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    for report in validation_reports:
        for rule_report in report.rule_reports:
            if rule_report.severity is IrValidationSeverity.ERROR:
                error_counts[rule_report.rule.value] += rule_report.issue_count
            elif rule_report.severity is IrValidationSeverity.WARN:
                warning_counts[rule_report.rule.value] += rule_report.issue_count

    rules = sorted(set(error_counts) | set(warning_counts))
    return tuple(
        IrRuleIssueCount(
            rule=rule,
            error_count=error_counts.get(rule, 0),
            warning_count=warning_counts.get(rule, 0),
        )
        for rule in rules
    )


def _build_optional_family_presence(
    documents: list[MotifIrDocumentRecord],
) -> IrOptionalFamilyPresence:
    total_phrase_spans = sum(
        len(record.document.optional_overlays.phrase_spans) for record in documents
    )
    documents_with_phrase_spans = sum(
        1 for record in documents if record.document.optional_overlays.phrase_spans
    )
    total_derived_views = sum(
        len(record.document.optional_views.playback_instances)
        + len(record.document.optional_views.derived_edge_sets)
        for record in documents
    )
    documents_with_derived_views = sum(
        1
        for record in documents
        if record.document.optional_views.playback_instances
        or record.document.optional_views.derived_edge_sets
    )
    return IrOptionalFamilyPresence(
        total_phrase_spans=total_phrase_spans,
        documents_with_phrase_spans=documents_with_phrase_spans,
        total_derived_views=total_derived_views,
        documents_with_derived_views=documents_with_derived_views,
    )


def _build_unsupported_feature_counts(
    manifest_entries: list[Mapping[str, Any]] | list[object],
) -> tuple[IrNamedCount, ...]:
    counts: Counter[str] = Counter()
    for entry in manifest_entries:
        if not isinstance(entry, Mapping):
            continue

        diagnostics = entry.get("conversion_diagnostics")
        if isinstance(diagnostics, list):
            for diagnostic in diagnostics:
                if not isinstance(diagnostic, Mapping):
                    continue
                category = str(diagnostic.get("category", "")).casefold()
                if category not in {"unsupported", "excluded"}:
                    continue

                code = str(diagnostic.get("code", "")).strip()
                if not code:
                    continue

                counts[code] += int(diagnostic.get("count", 0))
            continue

        fallback = entry.get("unsupported_features_dropped")
        if isinstance(fallback, list):
            for feature in fallback:
                feature_name = str(feature).strip()
                if feature_name:
                    counts[feature_name] += 1

    return tuple(IrNamedCount(name=name, count=counts[name]) for name in sorted(counts))


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * fraction
    lower_index = floor(position)
    upper_index = ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    blend = position - lower_index
    return float(lower_value + ((upper_value - lower_value) * blend))


def _render_scale_report(summary: IrCorpusSummary) -> str:
    lines = [
        "MotifML IR Corpus Summary",
        f"Documents: {summary.document_count}",
        (
            "Optional families: "
            f"phrase_spans={summary.optional_family_presence.total_phrase_spans} "
            f"across {summary.optional_family_presence.documents_with_phrase_spans} docs; "
            f"derived_views={summary.optional_family_presence.total_derived_views} "
            f"across {summary.optional_family_presence.documents_with_derived_views} docs"
        ),
    ]

    if summary.node_count_distributions:
        lines.append("Node family distributions:")
        lines.extend(
            (
                f"- {distribution.family}: min={distribution.min}, "
                f"mean={distribution.mean:.2f}, p95={distribution.p95:.2f}, "
                f"max={distribution.max}"
            )
            for distribution in summary.node_count_distributions
        )

    if summary.validation_issue_counts_by_rule:
        lines.append("Validation issues by rule:")
        lines.extend(
            (
                f"- {rule_count.rule}: errors={rule_count.error_count}, "
                f"warnings={rule_count.warning_count}"
            )
            for rule_count in summary.validation_issue_counts_by_rule
        )

    if summary.unsupported_feature_counts:
        lines.append("Unsupported or dropped source features:")
        lines.extend(
            f"- {feature.name}: {feature.count}"
            for feature in summary.unsupported_feature_counts
        )

    return "\n".join(lines)
