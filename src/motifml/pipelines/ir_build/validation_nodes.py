"""Validation and written-time-map nodes for IR build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.common import (
    REQUIRED_TOP_LEVEL_LIST_FIELDS,
    _build_bar_geometry_warnings,
    _coerce_optional_bool,
    _coerce_score_time,
    _error,
    _require_list_field,
    _require_score_time_field,
    _validate_track_surfaces,
)
from motifml.pipelines.ir_build.models import (
    CanonicalScoreValidationResult,
    IrBuildDiagnostic,
    WrittenTimeMapEntry,
    WrittenTimeMapResult,
)


def validate_canonical_score_surface(
    documents: list[MotifJsonDocument],
) -> list[CanonicalScoreValidationResult]:
    """Validate that raw Motif JSON documents expose the canonical IR input surface.

    Args:
        documents: Raw Motif JSON corpus documents loaded from the `01_raw` stage.

    Returns:
        Deterministic validation results with fatal errors and recoverable warnings.
    """
    return [
        CanonicalScoreValidationResult(
            relative_path=document.relative_path,
            source_hash=document.sha256,
            diagnostics=tuple(_validate_document_surface(document.score)),
        )
        for document in sorted(
            documents, key=lambda item: item.relative_path.casefold()
        )
    ]


def build_written_time_map(
    documents: list[MotifJsonDocument],
    validation_results: list[CanonicalScoreValidationResult],
) -> list[WrittenTimeMapResult]:
    """Build deterministic bar-level written time maps from canonical timeline bars.

    Args:
        documents: Raw Motif JSON corpus documents.
        validation_results: Surface-validation results from
            `validate_canonical_score_surface`.

    Returns:
        One typed written-time-map result per input document.
    """
    validation_results_by_path = {
        result.relative_path: result for result in validation_results
    }
    written_time_maps: list[WrittenTimeMapResult] = []

    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        validation_result = validation_results_by_path.get(document.relative_path)
        diagnostics: list[IrBuildDiagnostic] = []
        bars: tuple[WrittenTimeMapEntry, ...] = ()

        if validation_result is None:
            diagnostics.append(
                _error(
                    path="$",
                    code="missing_validation_result",
                    message=(
                        "canonical score surface validation must run before written "
                        "time-map construction."
                    ),
                )
            )
        elif not validation_result.passed:
            diagnostics.append(
                _error(
                    path="$",
                    code="canonical_surface_validation_failed",
                    message=(
                        "written time map cannot be built because the canonical "
                        "score surface validation failed."
                    ),
                )
            )
        else:
            bars, built_diagnostics = _build_written_time_map_entries(document.score)
            diagnostics.extend(built_diagnostics)

        written_time_maps.append(
            WrittenTimeMapResult(
                relative_path=document.relative_path,
                source_hash=document.sha256,
                bars=bars,
                diagnostics=tuple(diagnostics),
            )
        )

    return written_time_maps


def _validate_document_surface(score: dict[str, Any]) -> list[IrBuildDiagnostic]:
    diagnostics: list[IrBuildDiagnostic] = []

    top_level_lists = {
        field_name: _require_list_field(
            score,
            field_name,
            path=field_name,
            diagnostics=diagnostics,
        )
        for field_name in REQUIRED_TOP_LEVEL_LIST_FIELDS
    }

    timeline_bars = top_level_lists["timelineBars"]
    if timeline_bars is not None:
        _validate_timeline_bars(timeline_bars, diagnostics)

    tracks = top_level_lists["tracks"]
    if tracks is not None:
        _validate_track_surfaces(tracks, diagnostics)

    return diagnostics


def _validate_timeline_bars(
    timeline_bars: list[object],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for bar_index, timeline_bar in enumerate(timeline_bars):
        path = f"timelineBars[{bar_index}]"
        if not isinstance(timeline_bar, Mapping):
            diagnostics.append(
                _error(
                    path=path,
                    code="invalid_canonical_field",
                    message="timelineBars entries must be objects.",
                )
            )
            continue

        _require_score_time_field(
            timeline_bar,
            field_name="start",
            path=f"{path}.start",
            diagnostics=diagnostics,
            require_positive=False,
        )
        _require_score_time_field(
            timeline_bar,
            field_name="duration",
            path=f"{path}.duration",
            diagnostics=diagnostics,
            require_positive=True,
        )


def _build_written_time_map_entries(
    score: dict[str, Any],
) -> tuple[tuple[WrittenTimeMapEntry, ...], list[IrBuildDiagnostic]]:
    diagnostics: list[IrBuildDiagnostic] = []
    timeline_bars = score.get("timelineBars")
    if not isinstance(timeline_bars, list):
        diagnostics.append(
            _error(
                path="timelineBars",
                code="missing_canonical_field",
                message="canonical field 'timelineBars' is required.",
            )
        )
        return (), diagnostics

    raw_entries: list[tuple[int, int, ScoreTime, ScoreTime]] = []
    seen_bar_indexes: set[int] = set()
    anacrusis_flag = _coerce_optional_bool(score.get("anacrusis"), diagnostics)

    for ordinal, timeline_bar in enumerate(timeline_bars):
        timeline_bar_path = f"timelineBars[{ordinal}]"
        if not isinstance(timeline_bar, Mapping):
            diagnostics.append(
                _error(
                    path=timeline_bar_path,
                    code="invalid_canonical_field",
                    message="timelineBars entries must be objects.",
                )
            )
            continue

        bar_index = timeline_bar.get("index")
        if not isinstance(bar_index, int):
            diagnostics.append(
                _error(
                    path=f"{timeline_bar_path}.index",
                    code="invalid_canonical_field",
                    message="timelineBars entries must include an integer index.",
                )
            )
            continue

        if bar_index in seen_bar_indexes:
            diagnostics.append(
                _error(
                    path=f"{timeline_bar_path}.index",
                    code="duplicate_bar_index",
                    message=f"bar index {bar_index} appears more than once.",
                )
            )
            continue

        start = _coerce_score_time(
            timeline_bar.get("start"),
            path=f"{timeline_bar_path}.start",
            diagnostics=diagnostics,
            require_positive=False,
        )
        duration = _coerce_score_time(
            timeline_bar.get("duration"),
            path=f"{timeline_bar_path}.duration",
            diagnostics=diagnostics,
            require_positive=True,
        )
        if start is None or duration is None:
            continue

        seen_bar_indexes.add(bar_index)
        raw_entries.append((bar_index, ordinal, start, duration))

    raw_entries.sort(key=lambda item: item[0])

    bars: list[WrittenTimeMapEntry] = []
    previous_entry: WrittenTimeMapEntry | None = None
    for position, (bar_index, ordinal, start, duration) in enumerate(raw_entries):
        bar = WrittenTimeMapEntry(
            bar_index=bar_index,
            start=start,
            duration=duration,
            is_anacrusis=anacrusis_flag and position == 0,
        )
        if previous_entry is not None:
            expected_start = previous_entry.start + previous_entry.duration
            if bar.start < expected_start:
                diagnostics.append(
                    _error(
                        path=f"timelineBars[{ordinal}].start",
                        code="overlapping_bar_geometry",
                        message=(
                            "timeline bars must be contiguous and non-overlapping; "
                            "this bar starts before the previous bar ends."
                        ),
                    )
                )
            elif bar.start > expected_start:
                diagnostics.append(
                    _error(
                        path=f"timelineBars[{ordinal}].start",
                        code="non_contiguous_bar_geometry",
                        message=(
                            "timeline bars must be contiguous and non-overlapping; "
                            "this bar starts after a gap."
                        ),
                    )
                )

        bars.append(bar)
        previous_entry = bar

    diagnostics.extend(_build_bar_geometry_warnings(score, raw_entries, tuple(bars)))
    return tuple(bars), diagnostics
