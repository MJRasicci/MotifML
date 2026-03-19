"""Nodes for early IR build validation and time-foundation construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from motifml.datasets.motif_json_corpus_dataset import MotifJsonDocument
from motifml.ir.time import ScoreTime
from motifml.pipelines.ir_build.models import (
    CanonicalScoreValidationResult,
    DiagnosticSeverity,
    IrBuildDiagnostic,
)

REQUIRED_TOP_LEVEL_LIST_FIELDS = (
    "pointControls",
    "spanControls",
    "timelineBars",
    "tracks",
)
RELATION_HINT_FIELDS = (
    "hopoDestination",
    "hopoOrigin",
    "hopoType",
    "slides",
    "tieDestination",
    "tieOrigin",
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


def _validate_track_surfaces(
    tracks: list[object],
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for track_index, track in enumerate(tracks):
        track_path = f"tracks[{track_index}]"
        if not isinstance(track, Mapping):
            diagnostics.append(
                _error(
                    path=track_path,
                    code="invalid_canonical_field",
                    message="tracks entries must be objects.",
                )
            )
            continue

        staves = _require_list_field(
            track,
            field_name="staves",
            path=f"{track_path}.staves",
            diagnostics=diagnostics,
        )
        if staves is None:
            continue

        for staff_index, staff in enumerate(staves):
            staff_path = f"{track_path}.staves[{staff_index}]"
            if not isinstance(staff, Mapping):
                diagnostics.append(
                    _error(
                        path=staff_path,
                        code="invalid_canonical_field",
                        message="staves entries must be objects.",
                    )
                )
                continue

            measures = _require_list_field(
                staff,
                field_name="measures",
                path=f"{staff_path}.measures",
                diagnostics=diagnostics,
            )
            if measures is None:
                continue

            for measure_index, measure in enumerate(measures):
                measure_path = f"{staff_path}.measures[{measure_index}]"
                if not isinstance(measure, Mapping):
                    diagnostics.append(
                        _error(
                            path=measure_path,
                            code="invalid_canonical_field",
                            message="measures entries must be objects.",
                        )
                    )
                    continue

                voices = _require_list_field(
                    measure,
                    field_name="voices",
                    path=f"{measure_path}.voices",
                    diagnostics=diagnostics,
                )
                if voices is None:
                    continue

                for voice_index, voice in enumerate(voices):
                    voice_path = f"{measure_path}.voices[{voice_index}]"
                    if not isinstance(voice, Mapping):
                        diagnostics.append(
                            _error(
                                path=voice_path,
                                code="invalid_canonical_field",
                                message="voices entries must be objects.",
                            )
                        )
                        continue

                    beats = _require_list_field(
                        voice,
                        field_name="beats",
                        path=f"{voice_path}.beats",
                        diagnostics=diagnostics,
                    )
                    if beats is None:
                        continue

                    _validate_beat_surfaces(beats, voice_path, diagnostics)


def _validate_beat_surfaces(
    beats: list[object],
    voice_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for beat_index, beat in enumerate(beats):
        beat_path = f"{voice_path}.beats[{beat_index}]"
        if not isinstance(beat, Mapping):
            diagnostics.append(
                _error(
                    path=beat_path,
                    code="invalid_canonical_field",
                    message="beats entries must be objects.",
                )
            )
            continue

        _require_score_time_field(
            beat,
            field_name="offset",
            path=f"{beat_path}.offset",
            diagnostics=diagnostics,
            require_positive=False,
        )
        _require_score_time_field(
            beat,
            field_name="duration",
            path=f"{beat_path}.duration",
            diagnostics=diagnostics,
            require_positive=True,
        )

        notes = _require_list_field(
            beat,
            field_name="notes",
            path=f"{beat_path}.notes",
            diagnostics=diagnostics,
        )
        if notes is None:
            continue

        _validate_note_surfaces(notes, beat_path, diagnostics)


def _validate_note_surfaces(
    notes: list[object],
    beat_path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> None:
    for note_index, note in enumerate(notes):
        note_path = f"{beat_path}.notes[{note_index}]"
        if not isinstance(note, Mapping):
            diagnostics.append(
                _error(
                    path=note_path,
                    code="invalid_canonical_field",
                    message="notes entries must be objects.",
                )
            )
            continue

        articulation = note.get("articulation")
        if articulation is None:
            continue

        articulation_path = f"{note_path}.articulation"
        if not isinstance(articulation, Mapping):
            diagnostics.append(
                _error(
                    path=articulation_path,
                    code="invalid_canonical_field",
                    message="articulation must be an object when present.",
                )
            )
            continue

        relations = articulation.get("relations")
        relations_path = f"{articulation_path}.relations"
        if relations is None:
            if _has_relation_hints(articulation):
                diagnostics.append(
                    _warning(
                        path=relations_path,
                        code="missing_canonical_field",
                        message=(
                            "articulation exposes linked-note flags without the "
                            "canonical relations array."
                        ),
                    )
                )
            continue

        if not isinstance(relations, list):
            diagnostics.append(
                _error(
                    path=relations_path,
                    code="invalid_canonical_field",
                    message="relations must be an array when present.",
                )
            )
            continue

        for relation_index, relation in enumerate(relations):
            relation_path = f"{relations_path}[{relation_index}]"
            if not isinstance(relation, Mapping):
                diagnostics.append(
                    _error(
                        path=relation_path,
                        code="invalid_canonical_field",
                        message="relations entries must be objects.",
                    )
                )
                continue

            if (
                not isinstance(relation.get("kind"), str)
                or not relation["kind"].strip()
            ):
                diagnostics.append(
                    _error(
                        path=f"{relation_path}.kind",
                        code="invalid_canonical_field",
                        message="relations entries must include a non-empty kind.",
                    )
                )

            if not isinstance(relation.get("targetNoteId"), int):
                diagnostics.append(
                    _error(
                        path=f"{relation_path}.targetNoteId",
                        code="invalid_canonical_field",
                        message=(
                            "relations entries must include an integer targetNoteId."
                        ),
                    )
                )


def _require_list_field(
    mapping: Mapping[str, Any],
    field_name: str,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
) -> list[object] | None:
    value = mapping.get(field_name)
    if value is None:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message=f"canonical field '{field_name}' is required.",
            )
        )
        return None

    if not isinstance(value, list):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=f"canonical field '{field_name}' must be an array.",
            )
        )
        return None

    return value


def _require_score_time_field(
    mapping: Mapping[str, Any],
    field_name: str,
    path: str,
    diagnostics: list[IrBuildDiagnostic],
    *,
    require_positive: bool,
) -> None:
    value = mapping.get(field_name)
    if value is None:
        diagnostics.append(
            _error(
                path=path,
                code="missing_canonical_field",
                message=f"canonical field '{field_name}' is required.",
            )
        )
        return

    if not isinstance(value, Mapping):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=f"canonical field '{field_name}' must be a ScoreTime object.",
            )
        )
        return

    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="ScoreTime fields must include integer numerator and denominator values.",
            )
        )
        return

    try:
        score_time = ScoreTime(numerator=numerator, denominator=denominator)
    except ValueError as exc:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message=str(exc),
            )
        )
        return

    if require_positive and score_time.numerator <= 0:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="ScoreTime durations must be positive.",
            )
        )
        return

    if not require_positive and score_time.numerator < 0:
        diagnostics.append(
            _error(
                path=path,
                code="invalid_canonical_field",
                message="ScoreTime positions must be non-negative.",
            )
        )


def _has_relation_hints(articulation: Mapping[str, Any]) -> bool:
    return any(field_name in articulation for field_name in RELATION_HINT_FIELDS)


def _error(path: str, code: str, message: str) -> IrBuildDiagnostic:
    return IrBuildDiagnostic(
        severity=DiagnosticSeverity.ERROR,
        code=code,
        path=path,
        message=message,
    )


def _warning(path: str, code: str, message: str) -> IrBuildDiagnostic:
    return IrBuildDiagnostic(
        severity=DiagnosticSeverity.WARNING,
        code=code,
        path=path,
        message=message,
    )
