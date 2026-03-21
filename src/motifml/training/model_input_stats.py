"""Reporting helpers for tokenized ``05_model_input`` artifacts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motifml.datasets.json_dataset import to_json_compatible
from motifml.training.contracts import DatasetSplit
from motifml.training.model_input import (
    TokenizedDocumentRow,
    coerce_tokenized_document_rows,
)

_SPLIT_ORDER = {
    DatasetSplit.TRAIN: 0,
    DatasetSplit.VALIDATION: 1,
    DatasetSplit.TEST: 2,
}


@dataclass(frozen=True, slots=True)
class ModelInputReportingParameters:
    """Configurable thresholds for model-input reporting surfaces."""

    worst_document_limit: int = 10
    oversized_token_count_threshold: int = 8192

    def __post_init__(self) -> None:
        _require_positive_int(self.worst_document_limit, "worst_document_limit")
        _require_positive_int(
            self.oversized_token_count_threshold,
            "oversized_token_count_threshold",
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the reporting parameters for persistence."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ModelInputOversizedDocumentEntry:
    """One worst-offending tokenized document for reporting and inspection."""

    relative_path: str
    document_id: str
    split: DatasetSplit
    shard_id: str
    token_count: int
    exceeds_oversized_threshold: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relative_path", _normalize_text(self.relative_path, "relative_path")
        )
        object.__setattr__(
            self, "document_id", _normalize_text(self.document_id, "document_id")
        )
        object.__setattr__(self, "split", DatasetSplit(self.split))
        object.__setattr__(self, "shard_id", _normalize_text(self.shard_id, "shard_id"))
        _require_non_negative_int(self.token_count, "token_count")
        object.__setattr__(
            self,
            "exceeds_oversized_threshold",
            bool(self.exceeds_oversized_threshold),
        )


@dataclass(frozen=True, slots=True)
class ModelInputSplitStatsEntry:
    """Aggregate token statistics for one experiment split."""

    split: DatasetSplit
    document_count: int
    total_token_count: int
    max_token_count: int
    p95_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", DatasetSplit(self.split))
        _require_non_negative_int(self.document_count, "document_count")
        _require_non_negative_int(self.total_token_count, "total_token_count")
        _require_non_negative_int(self.max_token_count, "max_token_count")
        _require_non_negative_int(self.p95_token_count, "p95_token_count")


@dataclass(frozen=True, slots=True)
class ModelInputShardSummaryEntry:
    """Aggregate token statistics for one execution shard."""

    shard_id: str
    document_count: int
    total_token_count: int
    max_token_count: int
    p95_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard_id", _normalize_text(self.shard_id, "shard_id"))
        _require_non_negative_int(self.document_count, "document_count")
        _require_non_negative_int(self.total_token_count, "total_token_count")
        _require_non_negative_int(self.max_token_count, "max_token_count")
        _require_non_negative_int(self.p95_token_count, "p95_token_count")


@dataclass(frozen=True, slots=True)
class ModelInputShardStatsArtifact:
    """Reducer-friendly shard-local model-input statistics."""

    model_input_version: str
    storage_schema_version: str
    shard_id: str
    split_token_counts: dict[str, tuple[int, ...]]
    top_documents: tuple[ModelInputOversizedDocumentEntry, ...]
    reporting_parameters: ModelInputReportingParameters

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_input_version",
            _normalize_text(self.model_input_version, "model_input_version"),
        )
        object.__setattr__(
            self,
            "storage_schema_version",
            _normalize_text(self.storage_schema_version, "storage_schema_version"),
        )
        object.__setattr__(self, "shard_id", _normalize_text(self.shard_id, "shard_id"))
        object.__setattr__(
            self,
            "split_token_counts",
            {
                DatasetSplit(str(split)).value: tuple(
                    sorted(
                        _normalize_non_negative_int(
                            token_count,
                            field_name=f"split_token_counts[{split}]",
                        )
                        for token_count in token_counts
                    )
                )
                for split, token_counts in sorted(
                    self.split_token_counts.items(),
                    key=lambda item: _SPLIT_ORDER[DatasetSplit(str(item[0]))],
                )
            },
        )
        object.__setattr__(
            self,
            "top_documents",
            tuple(
                sorted(
                    (
                        entry
                        if isinstance(entry, ModelInputOversizedDocumentEntry)
                        else ModelInputOversizedDocumentEntry(
                            relative_path=str(entry["relative_path"]),
                            document_id=str(entry["document_id"]),
                            split=DatasetSplit(str(entry["split"])),
                            shard_id=str(entry["shard_id"]),
                            token_count=int(entry["token_count"]),
                            exceeds_oversized_threshold=bool(
                                entry["exceeds_oversized_threshold"]
                            ),
                        )
                        for entry in self.top_documents
                    ),
                    key=_document_sort_key,
                )
            ),
        )
        object.__setattr__(
            self,
            "reporting_parameters",
            coerce_model_input_reporting_parameters(self.reporting_parameters),
        )

    @property
    def document_count(self) -> int:
        """Return the total number of shard-local tokenized documents."""
        return sum(
            len(token_counts) for token_counts in self.split_token_counts.values()
        )

    @property
    def total_token_count(self) -> int:
        """Return the summed token count for the shard."""
        return sum(
            sum(token_counts) for token_counts in self.split_token_counts.values()
        )

    @property
    def max_token_count(self) -> int:
        """Return the maximum token count observed in the shard."""
        if not self.split_token_counts:
            return 0
        return max(
            (
                max(token_counts, default=0)
                for token_counts in self.split_token_counts.values()
            ),
            default=0,
        )

    @property
    def p95_token_count(self) -> int:
        """Return the p95 token count observed in the shard."""
        all_counts: list[int] = []
        for token_counts in self.split_token_counts.values():
            all_counts.extend(token_counts)
        return _p95_token_count(all_counts)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the shard stats for JSON persistence."""
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ModelInputStatsReport:
    """Corpus-level reporting surface for persisted tokenized model input."""

    model_input_version: str
    storage_schema_version: str
    total_document_count: int
    split_summaries: tuple[ModelInputSplitStatsEntry, ...]
    shard_summaries: tuple[ModelInputShardSummaryEntry, ...]
    worst_offending_documents: tuple[ModelInputOversizedDocumentEntry, ...]
    reporting_parameters: ModelInputReportingParameters

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_input_version",
            _normalize_text(self.model_input_version, "model_input_version"),
        )
        object.__setattr__(
            self,
            "storage_schema_version",
            _normalize_text(self.storage_schema_version, "storage_schema_version"),
        )
        _require_non_negative_int(self.total_document_count, "total_document_count")
        object.__setattr__(
            self,
            "split_summaries",
            tuple(
                sorted(
                    (
                        entry
                        if isinstance(entry, ModelInputSplitStatsEntry)
                        else ModelInputSplitStatsEntry(
                            split=DatasetSplit(str(entry["split"])),
                            document_count=int(entry["document_count"]),
                            total_token_count=int(entry["total_token_count"]),
                            max_token_count=int(entry["max_token_count"]),
                            p95_token_count=int(entry["p95_token_count"]),
                        )
                        for entry in self.split_summaries
                    ),
                    key=lambda item: _SPLIT_ORDER[item.split],
                )
            ),
        )
        object.__setattr__(
            self,
            "shard_summaries",
            tuple(
                sorted(
                    (
                        entry
                        if isinstance(entry, ModelInputShardSummaryEntry)
                        else ModelInputShardSummaryEntry(
                            shard_id=str(entry["shard_id"]),
                            document_count=int(entry["document_count"]),
                            total_token_count=int(entry["total_token_count"]),
                            max_token_count=int(entry["max_token_count"]),
                            p95_token_count=int(entry["p95_token_count"]),
                        )
                        for entry in self.shard_summaries
                    ),
                    key=lambda item: item.shard_id,
                )
            ),
        )
        object.__setattr__(
            self,
            "worst_offending_documents",
            tuple(
                sorted(
                    (
                        entry
                        if isinstance(entry, ModelInputOversizedDocumentEntry)
                        else ModelInputOversizedDocumentEntry(
                            relative_path=str(entry["relative_path"]),
                            document_id=str(entry["document_id"]),
                            split=DatasetSplit(str(entry["split"])),
                            shard_id=str(entry["shard_id"]),
                            token_count=int(entry["token_count"]),
                            exceeds_oversized_threshold=bool(
                                entry["exceeds_oversized_threshold"]
                            ),
                        )
                        for entry in self.worst_offending_documents
                    ),
                    key=_document_sort_key,
                )
            ),
        )
        object.__setattr__(
            self,
            "reporting_parameters",
            coerce_model_input_reporting_parameters(self.reporting_parameters),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the model-input stats for JSON persistence."""
        return to_json_compatible(self)


def render_model_input_stats_markdown(
    value: ModelInputStatsReport | Mapping[str, Any],
) -> str:
    """Render one human-reviewable Markdown summary for model-input pathologies."""
    report = coerce_model_input_stats_report(value)
    lines = [
        "# Model Input Pathology Report",
        "",
        f"- Model Input Version: `{report.model_input_version}`",
        f"- Total Documents: {report.total_document_count}",
        "- Oversized Threshold: "
        f"{report.reporting_parameters.oversized_token_count_threshold} tokens",
        "",
        "## Split Summaries",
        "",
    ]

    for split_summary in report.split_summaries:
        lines.append(
            "- "
            f"{split_summary.split.value}: documents={split_summary.document_count}, "
            f"total_tokens={split_summary.total_token_count}, "
            f"max_tokens={split_summary.max_token_count}, "
            f"p95_tokens={split_summary.p95_token_count}"
        )

    lines.extend(["", "## Worst Offending Documents", ""])
    if not report.worst_offending_documents:
        lines.extend(["No pathological documents were reported.", ""])
        return "\n".join(lines).rstrip() + "\n"

    for entry in report.worst_offending_documents:
        status = (
            "EXCEEDS_THRESHOLD"
            if entry.exceeds_oversized_threshold
            else "WITHIN_THRESHOLD"
        )
        lines.append(
            "- "
            f"`{entry.relative_path}` "
            f"(document_id=`{entry.document_id}`, split=`{entry.split.value}`, "
            f"shard=`{entry.shard_id}`): "
            f"{entry.token_count} tokens [{status}]"
        )

    lines.append("")
    return "\n".join(lines)


def build_model_input_shard_stats(
    rows: Sequence[TokenizedDocumentRow | Mapping[str, Any]],
    *,
    shard_id: str,
    reporting_parameters: ModelInputReportingParameters
    | Mapping[str, Any]
    | None = None,
) -> ModelInputShardStatsArtifact:
    """Build a shard-local reporting artifact from tokenized document rows."""
    typed_rows = coerce_tokenized_document_rows(rows)
    typed_reporting_parameters = coerce_model_input_reporting_parameters(
        reporting_parameters
    )
    if not typed_rows:
        raise ValueError("rows must contain at least one tokenized document.")

    model_input_version = typed_rows[0].model_input_version
    storage_schema_version = typed_rows[0].storage_schema_version
    split_token_counts: dict[str, list[int]] = defaultdict(list)
    top_documents = [
        _document_entry(
            row,
            shard_id=shard_id,
            oversized_token_count_threshold=(
                typed_reporting_parameters.oversized_token_count_threshold
            ),
        )
        for row in typed_rows
    ]
    for row in typed_rows[1:]:
        if row.model_input_version != model_input_version:
            raise ValueError("rows must share one model_input_version.")
        if row.storage_schema_version != storage_schema_version:
            raise ValueError("rows must share one storage_schema_version.")
    for row in typed_rows:
        split_token_counts[row.split.value].append(row.token_count)

    ordered_top_documents = tuple(
        sorted(top_documents, key=_document_sort_key)[
            : typed_reporting_parameters.worst_document_limit
        ]
    )
    return ModelInputShardStatsArtifact(
        model_input_version=model_input_version,
        storage_schema_version=storage_schema_version,
        shard_id=shard_id,
        split_token_counts={
            split: tuple(sorted(token_counts))
            for split, token_counts in split_token_counts.items()
        },
        top_documents=ordered_top_documents,
        reporting_parameters=typed_reporting_parameters,
    )


def reduce_model_input_stats_shards(
    shard_stats: Sequence[ModelInputShardStatsArtifact | Mapping[str, Any]],
) -> ModelInputStatsReport:
    """Reduce shard-local reporting artifacts into one corpus-level stats report."""
    typed_shards = tuple(
        coerce_model_input_shard_stats_artifact(entry) for entry in shard_stats
    )
    if not typed_shards:
        raise ValueError("shard_stats must contain at least one artifact.")

    baseline = typed_shards[0]
    split_token_counts: dict[DatasetSplit, list[int]] = defaultdict(list)
    shard_summaries: list[ModelInputShardSummaryEntry] = []
    worst_documents: list[ModelInputOversizedDocumentEntry] = []
    total_document_count = 0
    for shard in typed_shards:
        _validate_model_input_shard_stats(shard, baseline)
        total_document_count += shard.document_count
        shard_summaries.append(
            ModelInputShardSummaryEntry(
                shard_id=shard.shard_id,
                document_count=shard.document_count,
                total_token_count=shard.total_token_count,
                max_token_count=shard.max_token_count,
                p95_token_count=shard.p95_token_count,
            )
        )
        for split, token_counts in shard.split_token_counts.items():
            split_token_counts[DatasetSplit(split)].extend(token_counts)
        worst_documents.extend(shard.top_documents)

    split_summaries = tuple(
        ModelInputSplitStatsEntry(
            split=split,
            document_count=len(sorted_counts),
            total_token_count=sum(sorted_counts),
            max_token_count=max(sorted_counts, default=0),
            p95_token_count=_p95_token_count(sorted_counts),
        )
        for split, sorted_counts in sorted(
            (
                (split, sorted(token_counts))
                for split, token_counts in split_token_counts.items()
            ),
            key=lambda item: _SPLIT_ORDER[item[0]],
        )
    )
    ordered_worst_documents = tuple(
        sorted(worst_documents, key=_document_sort_key)[
            : baseline.reporting_parameters.worst_document_limit
        ]
    )
    return ModelInputStatsReport(
        model_input_version=baseline.model_input_version,
        storage_schema_version=baseline.storage_schema_version,
        total_document_count=total_document_count,
        split_summaries=split_summaries,
        shard_summaries=tuple(shard_summaries),
        worst_offending_documents=ordered_worst_documents,
        reporting_parameters=baseline.reporting_parameters,
    )


def coerce_model_input_reporting_parameters(
    value: ModelInputReportingParameters | Mapping[str, Any] | None,
) -> ModelInputReportingParameters:
    """Coerce Kedro reporting-parameter mappings into the typed config."""
    if value is None:
        return ModelInputReportingParameters()
    if isinstance(value, ModelInputReportingParameters):
        return value
    return ModelInputReportingParameters(
        worst_document_limit=int(value.get("worst_document_limit", 10)),
        oversized_token_count_threshold=int(
            value.get("oversized_token_count_threshold", 8192)
        ),
    )


def coerce_model_input_shard_stats_artifact(
    value: ModelInputShardStatsArtifact | Mapping[str, Any],
) -> ModelInputShardStatsArtifact:
    """Coerce JSON-loaded shard stats payloads into the typed artifact."""
    if isinstance(value, ModelInputShardStatsArtifact):
        return value
    return ModelInputShardStatsArtifact(
        model_input_version=str(value["model_input_version"]),
        storage_schema_version=str(value["storage_schema_version"]),
        shard_id=str(value["shard_id"]),
        split_token_counts={
            str(split): tuple(token_counts)
            for split, token_counts in value.get("split_token_counts", {}).items()
        },
        top_documents=tuple(value.get("top_documents", ())),
        reporting_parameters=coerce_model_input_reporting_parameters(
            value.get("reporting_parameters", {})
        ),
    )


def coerce_model_input_stats_report(
    value: ModelInputStatsReport | Mapping[str, Any],
) -> ModelInputStatsReport:
    """Coerce JSON-loaded model-input stats payloads into the typed report."""
    if isinstance(value, ModelInputStatsReport):
        return value
    return ModelInputStatsReport(
        model_input_version=str(value["model_input_version"]),
        storage_schema_version=str(value["storage_schema_version"]),
        total_document_count=int(value["total_document_count"]),
        split_summaries=tuple(value.get("split_summaries", ())),
        shard_summaries=tuple(value.get("shard_summaries", ())),
        worst_offending_documents=tuple(value.get("worst_offending_documents", ())),
        reporting_parameters=coerce_model_input_reporting_parameters(
            value.get("reporting_parameters", {})
        ),
    )


def _document_entry(
    row: TokenizedDocumentRow,
    *,
    shard_id: str,
    oversized_token_count_threshold: int,
) -> ModelInputOversizedDocumentEntry:
    return ModelInputOversizedDocumentEntry(
        relative_path=row.relative_path,
        document_id=row.document_id,
        split=row.split,
        shard_id=shard_id,
        token_count=row.token_count,
        exceeds_oversized_threshold=(
            row.token_count >= oversized_token_count_threshold
        ),
    )


def _document_sort_key(entry: ModelInputOversizedDocumentEntry) -> tuple[Any, ...]:
    return (
        -entry.token_count,
        _SPLIT_ORDER[entry.split],
        entry.relative_path.casefold(),
        entry.shard_id.casefold(),
    )


def _p95_token_count(token_counts: Sequence[int]) -> int:
    if not token_counts:
        return 0
    ordered = sorted(token_counts)
    index = max(math.ceil(0.95 * len(ordered)) - 1, 0)
    return ordered[index]


def _validate_model_input_shard_stats(
    shard: ModelInputShardStatsArtifact,
    baseline: ModelInputShardStatsArtifact,
) -> None:
    if shard.model_input_version != baseline.model_input_version:
        raise ValueError("All shard stats must share one model_input_version.")
    if shard.storage_schema_version != baseline.storage_schema_version:
        raise ValueError("All shard stats must share one storage_schema_version.")
    if shard.reporting_parameters != baseline.reporting_parameters:
        raise ValueError("All shard stats must share one reporting_parameters surface.")


def _normalize_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> None:
    _normalize_non_negative_int(value, field_name=field_name)


def _require_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")


__all__ = [
    "ModelInputOversizedDocumentEntry",
    "ModelInputReportingParameters",
    "ModelInputShardStatsArtifact",
    "ModelInputShardSummaryEntry",
    "ModelInputSplitStatsEntry",
    "ModelInputStatsReport",
    "build_model_input_shard_stats",
    "coerce_model_input_stats_report",
    "coerce_model_input_reporting_parameters",
    "coerce_model_input_shard_stats_artifact",
    "reduce_model_input_stats_shards",
    "render_model_input_stats_markdown",
]
