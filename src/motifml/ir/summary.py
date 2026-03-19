"""Corpus-level summary models for MotifML IR validation reporting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IrCorpusDocumentSummary:
    """Per-document node and edge counts used in corpus rollups."""

    relative_path: str
    node_counts: dict[str, int]
    edge_counts: dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "node_counts",
            _normalize_count_mapping(self.node_counts, "node_counts"),
        )
        object.__setattr__(
            self,
            "edge_counts",
            _normalize_count_mapping(self.edge_counts, "edge_counts"),
        )


@dataclass(frozen=True)
class IrCountDistribution:
    """Distribution statistics for one node family across the corpus."""

    family: str
    min: int
    max: int
    mean: float
    p25: float
    p50: float
    p75: float
    p95: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _normalize_text(self.family, "family"))
        if self.min < 0 or self.max < 0:
            raise ValueError("distribution bounds must be non-negative.")
        if self.max < self.min:
            raise ValueError("distribution max must be greater than or equal to min.")


@dataclass(frozen=True)
class IrRuleIssueCount:
    """Corpus-wide issue totals for one validation rule."""

    rule: str
    error_count: int = 0
    warning_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule", _normalize_text(self.rule, "rule"))
        if self.error_count < 0 or self.warning_count < 0:
            raise ValueError("validation issue counts must be non-negative.")


@dataclass(frozen=True)
class IrNamedCount:
    """A stable name/count pair used in summary rollups."""

    name: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_text(self.name, "name"))
        if self.count < 0:
            raise ValueError("count must be non-negative.")


@dataclass(frozen=True)
class IrOptionalFamilyPresence:
    """Presence totals for optional overlay and derived-view families."""

    total_phrase_spans: int = 0
    documents_with_phrase_spans: int = 0
    total_derived_views: int = 0
    documents_with_derived_views: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "total_phrase_spans",
            "documents_with_phrase_spans",
            "total_derived_views",
            "documents_with_derived_views",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True)
class IrCorpusSummary:
    """Machine-readable corpus summary plus a human-readable scale report."""

    document_count: int
    per_document: tuple[IrCorpusDocumentSummary, ...] = ()
    node_count_distributions: tuple[IrCountDistribution, ...] = ()
    validation_issue_counts_by_rule: tuple[IrRuleIssueCount, ...] = ()
    optional_family_presence: IrOptionalFamilyPresence = field(
        default_factory=IrOptionalFamilyPresence
    )
    unsupported_feature_counts: tuple[IrNamedCount, ...] = ()
    scale_report: str | None = None

    def __post_init__(self) -> None:
        if self.document_count < 0:
            raise ValueError("document_count must be non-negative.")

        object.__setattr__(
            self,
            "per_document",
            tuple(
                sorted(
                    self.per_document,
                    key=lambda item: item.relative_path.casefold(),
                )
            ),
        )
        object.__setattr__(
            self,
            "node_count_distributions",
            tuple(
                sorted(
                    self.node_count_distributions,
                    key=lambda item: item.family.casefold(),
                )
            ),
        )
        object.__setattr__(
            self,
            "validation_issue_counts_by_rule",
            tuple(
                sorted(
                    self.validation_issue_counts_by_rule,
                    key=lambda item: item.rule.casefold(),
                )
            ),
        )
        object.__setattr__(
            self,
            "unsupported_feature_counts",
            tuple(
                sorted(
                    self.unsupported_feature_counts,
                    key=lambda item: item.name.casefold(),
                )
            ),
        )
        if self.scale_report is not None:
            object.__setattr__(
                self,
                "scale_report",
                _normalize_text(self.scale_report, "scale_report"),
            )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


def _normalize_count_mapping(value: dict[str, int], field_name: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for raw_key, raw_count in sorted(
        value.items(), key=lambda item: str(item[0]).casefold()
    ):
        key = _normalize_text(str(raw_key), f"{field_name} key")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"{field_name} values must be non-negative.")
        normalized[key] = count

    return normalized


__all__ = [
    "IrCorpusDocumentSummary",
    "IrCorpusSummary",
    "IrCountDistribution",
    "IrNamedCount",
    "IrOptionalFamilyPresence",
    "IrRuleIssueCount",
]
