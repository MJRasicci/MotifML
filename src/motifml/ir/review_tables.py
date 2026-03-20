"""Deterministic review-table helpers for IR inspection."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from motifml.datasets.motif_ir_corpus_dataset import (
    MotifIrCorpusDataset,
    MotifIrDocumentRecord,
)
from motifml.ir.ids import (
    bar_sort_key,
    note_sort_key,
    onset_sort_key,
    point_control_sort_key,
    span_control_sort_key,
    voice_lane_sort_key,
)
from motifml.ir.models import (
    DynamicChangeValue,
    FermataValue,
    HairpinValue,
    MotifMlIrDocument,
    OttavaValue,
    Pitch,
    TechniquePayload,
    TempoChangeValue,
)
from motifml.ir.review_models import (
    BarReviewRollup,
    ControlEventRow,
    IrStructureSummary,
    OnsetNoteRow,
    OnsetNoteTable,
    ReviewNamedCount,
    VoiceLaneOnsetRow,
    VoiceLaneOnsetTable,
    VoiceLaneReviewRollup,
)
from motifml.ir.serialization import deserialize_document
from motifml.ir.time import ScoreTime


def load_ir_document_record(
    input_path: str | Path,
    relative_path: str | None = None,
) -> MotifIrDocumentRecord:
    """Load either one IR file or one selected member from an IR corpus directory."""
    path = Path(input_path)
    if path.is_dir():
        if relative_path is None or not relative_path.strip():
            raise ValueError(
                "relative_path must be provided when loading from a corpus directory."
            )

        normalized_relative_path = relative_path.strip()
        for record in MotifIrCorpusDataset(filepath=str(path)).load():
            if record.relative_path == normalized_relative_path:
                return record

        raise ValueError(
            f"IR corpus member '{normalized_relative_path}' was not found under "
            f"'{path.as_posix()}'."
        )

    if relative_path is not None:
        raise ValueError(
            "relative_path can only be provided when loading from a corpus directory."
        )

    if not path.is_file():
        raise ValueError(f"IR input path does not exist: {path.as_posix()}")

    relative_identity = (
        path.name.removesuffix(".ir.json")
        if path.name.endswith(".ir.json")
        else path.name
    )
    return MotifIrDocumentRecord(
        relative_path=relative_identity,
        document=deserialize_document(path.read_text(encoding="utf-8")),
    )


def build_structure_summary(document: MotifMlIrDocument) -> IrStructureSummary:
    """Build deterministic document-level counts and rollups for inspection."""
    bars_by_id = _bars_by_id(document)
    voice_lanes_by_bar_id = _voice_lanes_by_bar_id(document)
    onsets_by_voice_lane_id = _onsets_by_voice_lane_id(document)
    notes_by_onset_id = _notes_by_onset_id(document)

    edge_counts = Counter(edge.edge_type.value for edge in document.edges)
    edge_counts_by_type = tuple(
        ReviewNamedCount(name=name, count=edge_counts[name])
        for name in sorted(edge_counts)
    )

    bar_rollups = []
    for bar in _sorted_bars(document):
        bar_voice_lanes = voice_lanes_by_bar_id.get(bar.bar_id, ())
        bar_onsets = [
            onset
            for voice_lane in bar_voice_lanes
            for onset in onsets_by_voice_lane_id.get(voice_lane.voice_lane_id, ())
        ]
        bar_notes = [
            note
            for onset in bar_onsets
            for note in notes_by_onset_id.get(onset.onset_id, ())
        ]
        bar_start = bar.start
        bar_end = bar.start + bar.duration

        point_control_count = sum(
            1
            for control in document.point_control_events
            if bar_start <= control.time < bar_end
        )
        span_control_count = sum(
            1
            for control in document.span_control_events
            if control.start_time < bar_end and control.end_time > bar_start
        )
        bar_rollups.append(
            BarReviewRollup(
                bar_id=bar.bar_id,
                bar_index=bar.bar_index,
                start=bar.start,
                duration=bar.duration,
                voice_lane_count=len(bar_voice_lanes),
                onset_count=len(bar_onsets),
                note_count=len(bar_notes),
                point_control_count=point_control_count,
                span_control_count=span_control_count,
            )
        )

    voice_lane_rollups = []
    for voice_lane in _sorted_voice_lanes(document, bars_by_id):
        onsets = onsets_by_voice_lane_id.get(voice_lane.voice_lane_id, ())
        notes = [
            note
            for onset in onsets
            for note in notes_by_onset_id.get(onset.onset_id, ())
        ]
        voice_lane_rollups.append(
            VoiceLaneReviewRollup(
                part_id=voice_lane.part_id,
                staff_id=voice_lane.staff_id,
                bar_id=voice_lane.bar_id,
                bar_index=bars_by_id[voice_lane.bar_id].bar_index,
                voice_lane_id=voice_lane.voice_lane_id,
                voice_lane_chain_id=voice_lane.voice_lane_chain_id,
                voice_index=voice_lane.voice_index,
                onset_count=len(onsets),
                note_count=len(notes),
                rest_onset_count=sum(1 for onset in onsets if onset.is_rest),
            )
        )

    return IrStructureSummary(
        part_count=len(document.parts),
        staff_count=len(document.staves),
        bar_count=len(document.bars),
        voice_lane_count=len(document.voice_lanes),
        onset_count=len(document.onset_groups),
        note_count=len(document.note_events),
        point_control_count=len(document.point_control_events),
        span_control_count=len(document.span_control_events),
        edge_count=len(document.edges),
        edge_counts_by_type=edge_counts_by_type,
        bar_rollups=tuple(bar_rollups),
        voice_lane_rollups=tuple(voice_lane_rollups),
    )


def build_voice_lane_onset_tables(
    document: MotifMlIrDocument,
) -> tuple[VoiceLaneOnsetTable, ...]:
    """Build per-bar / per-voice onset tables with rational timing preserved."""
    bars_by_id = _bars_by_id(document)
    onsets_by_voice_lane_id = _onsets_by_voice_lane_id(document)
    notes_by_onset_id = _notes_by_onset_id(document)

    tables = []
    for voice_lane in _sorted_voice_lanes(document, bars_by_id):
        bar = bars_by_id[voice_lane.bar_id]
        rows = [
            VoiceLaneOnsetRow(
                part_id=voice_lane.part_id,
                staff_id=voice_lane.staff_id,
                bar_id=voice_lane.bar_id,
                bar_index=bar.bar_index,
                voice_lane_id=voice_lane.voice_lane_id,
                voice_lane_chain_id=voice_lane.voice_lane_chain_id,
                voice_index=voice_lane.voice_index,
                onset_id=onset.onset_id,
                time=onset.time,
                bar_offset=onset.time - bar.start,
                duration_notated=onset.duration_notated,
                duration_sounding_max=onset.duration_sounding_max,
                is_rest=onset.is_rest,
                attack_order_in_voice=onset.attack_order_in_voice,
                note_count=len(notes_by_onset_id.get(onset.onset_id, ())),
                grace_type=onset.grace_type,
                dynamic_local=onset.dynamic_local,
                technique_summary=summarize_techniques(onset.techniques),
            )
            for onset in onsets_by_voice_lane_id.get(voice_lane.voice_lane_id, ())
        ]
        if not rows:
            continue

        tables.append(
            VoiceLaneOnsetTable(
                part_id=voice_lane.part_id,
                staff_id=voice_lane.staff_id,
                bar_id=voice_lane.bar_id,
                bar_index=bar.bar_index,
                voice_lane_id=voice_lane.voice_lane_id,
                voice_lane_chain_id=voice_lane.voice_lane_chain_id,
                voice_index=voice_lane.voice_index,
                rows=tuple(rows),
            )
        )

    return tuple(tables)


def build_onset_note_tables(document: MotifMlIrDocument) -> tuple[OnsetNoteTable, ...]:
    """Build note tables grouped by onset."""
    bars_by_id = _bars_by_id(document)
    voice_lanes_by_id = _voice_lanes_by_id(document)
    notes_by_onset_id = _notes_by_onset_id(document)

    tables = []
    for onset in _sorted_onsets(document):
        notes = notes_by_onset_id.get(onset.onset_id, ())
        if not notes:
            continue

        voice_lane = voice_lanes_by_id[onset.voice_lane_id]
        bar = bars_by_id[onset.bar_id]
        rows = [
            OnsetNoteRow(
                part_id=note.part_id,
                staff_id=note.staff_id,
                bar_id=onset.bar_id,
                bar_index=bar.bar_index,
                voice_lane_id=voice_lane.voice_lane_id,
                voice_lane_chain_id=voice_lane.voice_lane_chain_id,
                voice_index=voice_lane.voice_index,
                onset_id=onset.onset_id,
                note_id=note.note_id,
                time=note.time,
                bar_offset=note.time - bar.start,
                pitch_text=format_pitch(note.pitch),
                attack_duration=note.attack_duration,
                sounding_duration=note.sounding_duration,
                string_number=note.string_number,
                velocity=note.velocity,
                technique_summary=summarize_techniques(note.techniques),
            )
            for note in notes
        ]
        tables.append(
            OnsetNoteTable(
                part_id=voice_lane.part_id,
                staff_id=voice_lane.staff_id,
                bar_id=onset.bar_id,
                bar_index=bar.bar_index,
                voice_lane_id=voice_lane.voice_lane_id,
                voice_lane_chain_id=voice_lane.voice_lane_chain_id,
                voice_index=voice_lane.voice_index,
                onset_id=onset.onset_id,
                onset_time=onset.time,
                onset_bar_offset=onset.time - bar.start,
                rows=tuple(rows),
            )
        )

    return tuple(tables)


def build_control_event_rows(
    document: MotifMlIrDocument,
) -> tuple[ControlEventRow, ...]:
    """Normalize point and span controls into one deterministic review table."""
    bars = _sorted_bars(document)

    rows = [
        ControlEventRow(
            control_id=control.control_id,
            family="point",
            kind=control.kind.value,
            scope=control.scope.value,
            target_ref=control.target_ref,
            start_time=control.time,
            end_time=None,
            start_bar_index=_bar_index_for_time(bars, control.time),
            end_bar_index=None,
            value_summary=summarize_control_value(control.value),
        )
        for control in sorted(
            document.point_control_events,
            key=lambda item: point_control_sort_key(
                item.scope.value,
                item.target_ref,
                item.time,
                item.control_id,
            ),
        )
    ]
    rows.extend(
        ControlEventRow(
            control_id=control.control_id,
            family="span",
            kind=control.kind.value,
            scope=control.scope.value,
            target_ref=control.target_ref,
            start_time=control.start_time,
            end_time=control.end_time,
            start_bar_index=_bar_index_for_time(bars, control.start_time),
            end_bar_index=_bar_index_for_time(bars, control.end_time),
            value_summary=summarize_control_value(control.value),
            start_anchor_ref=control.start_anchor_ref,
            end_anchor_ref=control.end_anchor_ref,
        )
        for control in sorted(
            document.span_control_events,
            key=lambda item: span_control_sort_key(
                item.scope.value,
                item.target_ref,
                item.start_time,
                item.end_time,
                item.control_id,
            ),
        )
    )
    return tuple(
        sorted(
            rows,
            key=lambda item: (
                item.start_time,
                item.end_time if item.end_time is not None else item.start_time,
                item.family,
                item.kind,
                item.control_id,
            ),
        )
    )


def render_structure_summary_markdown(summary: IrStructureSummary) -> str:
    """Render a readable Markdown rollup for notebook review."""
    lines = [
        "| Family | Count |",
        "| --- | ---: |",
        f"| Parts | {summary.part_count} |",
        f"| Staves | {summary.staff_count} |",
        f"| Bars | {summary.bar_count} |",
        f"| Voice lanes | {summary.voice_lane_count} |",
        f"| Onset groups | {summary.onset_count} |",
        f"| Note events | {summary.note_count} |",
        f"| Point controls | {summary.point_control_count} |",
        f"| Span controls | {summary.span_control_count} |",
        f"| Edges | {summary.edge_count} |",
    ]

    if summary.edge_counts_by_type:
        lines.extend(
            (
                "",
                "**Edge families**",
                "",
                _render_markdown_table(
                    ("Edge type", "Count"),
                    (
                        (item.name, str(item.count))
                        for item in summary.edge_counts_by_type
                    ),
                ),
            )
        )

    if summary.bar_rollups:
        lines.extend(
            (
                "",
                "**Bar rollups**",
                "",
                _render_markdown_table(
                    (
                        "Bar",
                        "Start",
                        "Duration",
                        "Voice lanes",
                        "Onsets",
                        "Notes",
                        "Point ctrls",
                        "Span ctrls",
                    ),
                    (
                        (
                            f"{item.bar_index} (`{item.bar_id}`)",
                            format_score_time(item.start),
                            format_score_time(item.duration),
                            str(item.voice_lane_count),
                            str(item.onset_count),
                            str(item.note_count),
                            str(item.point_control_count),
                            str(item.span_control_count),
                        )
                        for item in summary.bar_rollups
                    ),
                ),
            )
        )

    if summary.voice_lane_rollups:
        lines.extend(
            (
                "",
                "**Voice-lane rollups**",
                "",
                _render_markdown_table(
                    (
                        "Bar",
                        "Voice lane",
                        "Chain",
                        "Voice",
                        "Onsets",
                        "Notes",
                        "Rest onsets",
                    ),
                    (
                        (
                            f"{item.bar_index} (`{item.bar_id}`)",
                            f"`{item.voice_lane_id}`",
                            f"`{item.voice_lane_chain_id}`",
                            str(item.voice_index),
                            str(item.onset_count),
                            str(item.note_count),
                            str(item.rest_onset_count),
                        )
                        for item in summary.voice_lane_rollups
                    ),
                ),
            )
        )

    return "\n".join(lines)


def render_voice_lane_onset_table_markdown(table: VoiceLaneOnsetTable) -> str:
    """Render one grouped onset table as Markdown."""
    heading = (
        f"### Bar {table.bar_index} Voice {table.voice_index}\n\n"
        f"- Voice lane: `{table.voice_lane_id}`\n"
        f"- Voice chain: `{table.voice_lane_chain_id}`\n"
        f"- Staff: `{table.staff_id}`\n"
        f"- Part: `{table.part_id}`\n"
    )
    table_markdown = _render_markdown_table(
        (
            "Attack #",
            "Onset",
            "Time",
            "Bar offset",
            "Notated",
            "Sounding max",
            "Rest",
            "Notes",
            "Grace",
            "Dynamic",
            "Techniques",
        ),
        (
            (
                str(row.attack_order_in_voice),
                f"`{row.onset_id}`",
                format_score_time(row.time),
                format_score_time(row.bar_offset),
                format_score_time(row.duration_notated),
                (
                    format_score_time(row.duration_sounding_max)
                    if row.duration_sounding_max is not None
                    else "-"
                ),
                "yes" if row.is_rest else "no",
                str(row.note_count),
                row.grace_type or "-",
                row.dynamic_local or "-",
                row.technique_summary or "-",
            )
            for row in table.rows
        ),
    )
    return f"{heading}\n{table_markdown}"


def render_onset_note_table_markdown(table: OnsetNoteTable) -> str:
    """Render one grouped note table as Markdown."""
    heading = (
        f"### Onset `{table.onset_id}`\n\n"
        f"- Bar: {table.bar_index} (`{table.bar_id}`)\n"
        f"- Voice lane: `{table.voice_lane_id}`\n"
        f"- Voice chain: `{table.voice_lane_chain_id}`\n"
        f"- Voice index: {table.voice_index}\n"
        f"- Time: {format_score_time(table.onset_time)}\n"
        f"- Bar offset: {format_score_time(table.onset_bar_offset)}\n"
    )
    table_markdown = _render_markdown_table(
        (
            "Note",
            "Time",
            "Bar offset",
            "Pitch",
            "Attack",
            "Sounding",
            "String",
            "Velocity",
            "Techniques",
        ),
        (
            (
                f"`{row.note_id}`",
                format_score_time(row.time),
                format_score_time(row.bar_offset),
                row.pitch_text,
                format_score_time(row.attack_duration),
                format_score_time(row.sounding_duration),
                str(row.string_number) if row.string_number is not None else "-",
                str(row.velocity) if row.velocity is not None else "-",
                row.technique_summary or "-",
            )
            for row in table.rows
        ),
    )
    return f"{heading}\n{table_markdown}"


def render_control_event_rows_markdown(rows: Sequence[ControlEventRow]) -> str:
    """Render control rows as one Markdown table."""
    return _render_markdown_table(
        (
            "Control",
            "Family",
            "Kind",
            "Scope",
            "Target",
            "Start",
            "End",
            "Start bar",
            "End bar",
            "Value",
            "Anchors",
        ),
        (
            (
                f"`{row.control_id}`",
                row.family,
                row.kind,
                row.scope,
                f"`{row.target_ref}`",
                format_score_time(row.start_time),
                format_score_time(row.end_time) if row.end_time is not None else "-",
                str(row.start_bar_index) if row.start_bar_index is not None else "-",
                str(row.end_bar_index) if row.end_bar_index is not None else "-",
                row.value_summary,
                _render_anchor_summary(row),
            )
            for row in rows
        ),
    )


def format_score_time(value: ScoreTime | None) -> str:
    """Render a `ScoreTime` using its exact rational form."""
    if value is None:
        return "-"
    return f"{value.numerator}/{value.denominator}"


def format_pitch(pitch: Pitch | None) -> str:
    """Render a pitch as a compact readable label."""
    if pitch is None:
        return "unpitched"
    accidental = "" if pitch.accidental is None else pitch.accidental
    return f"{pitch.step.value}{accidental}{pitch.octave}"


def summarize_techniques(techniques: TechniquePayload | None) -> str | None:
    """Render a stable one-line summary of note or onset techniques."""
    if techniques is None:
        return None

    values = []
    values.extend(_summarize_generic_techniques(techniques))
    values.extend(_summarize_general_techniques(techniques))
    values.extend(_summarize_string_fretted_techniques(techniques))

    return ", ".join(values) if values else None


def summarize_control_value(value: object) -> str:
    """Render a stable one-line summary for a point or span control payload."""
    if isinstance(value, TempoChangeValue):
        return f"beats_per_minute={value.beats_per_minute:g}"
    if isinstance(value, DynamicChangeValue):
        return f"marking={value.marking}"
    if isinstance(value, FermataValue):
        parts = []
        if value.fermata_type is not None:
            parts.append(f"type={value.fermata_type}")
        if value.length_scale is not None:
            parts.append(f"length_scale={value.length_scale:g}")
        return ", ".join(parts) if parts else "fermata"
    if isinstance(value, HairpinValue):
        niente = ", niente=true" if value.niente else ""
        return f"direction={value.direction.value}{niente}"
    if isinstance(value, OttavaValue):
        return f"octave_shift={value.octave_shift}"

    return str(value)


def _bars_by_id(document: MotifMlIrDocument) -> dict[str, object]:
    return {bar.bar_id: bar for bar in document.bars}


def _voice_lanes_by_id(document: MotifMlIrDocument) -> dict[str, object]:
    return {voice_lane.voice_lane_id: voice_lane for voice_lane in document.voice_lanes}


def _voice_lanes_by_bar_id(
    document: MotifMlIrDocument,
) -> dict[str, tuple[object, ...]]:
    bars_by_id = _bars_by_id(document)
    grouped: dict[str, list[object]] = defaultdict(list)
    for voice_lane in _sorted_voice_lanes(document, bars_by_id):
        grouped[voice_lane.bar_id].append(voice_lane)
    return {key: tuple(value) for key, value in grouped.items()}


def _onsets_by_voice_lane_id(
    document: MotifMlIrDocument,
) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for onset in _sorted_onsets(document):
        grouped[onset.voice_lane_id].append(onset)
    return {key: tuple(value) for key, value in grouped.items()}


def _notes_by_onset_id(document: MotifMlIrDocument) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for note in _sorted_notes(document):
        grouped[note.onset_id].append(note)
    return {key: tuple(value) for key, value in grouped.items()}


def _sorted_bars(document: MotifMlIrDocument) -> list[object]:
    return sorted(
        document.bars, key=lambda item: bar_sort_key(item.bar_index, item.bar_id)
    )


def _sorted_voice_lanes(
    document: MotifMlIrDocument,
    bars_by_id: dict[str, object],
) -> list[object]:
    return sorted(
        document.voice_lanes,
        key=lambda item: voice_lane_sort_key(
            bars_by_id[item.bar_id].bar_index,
            item.staff_id,
            item.voice_index,
            item.voice_lane_id,
        ),
    )


def _sorted_onsets(document: MotifMlIrDocument) -> list[object]:
    return sorted(
        document.onset_groups,
        key=lambda item: onset_sort_key(
            item.voice_lane_id,
            item.time,
            item.attack_order_in_voice,
            item.onset_id,
        ),
    )


def _sorted_notes(document: MotifMlIrDocument) -> list[object]:
    return sorted(
        document.note_events,
        key=lambda item: (
            item.onset_id.casefold(),
            *note_sort_key(item.string_number, item.pitch, item.note_id),
        ),
    )


def _bar_index_for_time(bars: Sequence[object], time: ScoreTime) -> int | None:
    for bar in bars:
        bar_end = bar.start + bar.duration
        if bar.start <= time < bar_end:
            return bar.bar_index

    if bars:
        last_bar = bars[-1]
        if time == last_bar.start + last_bar.duration:
            return last_bar.bar_index

    return None


def _summarize_generic_techniques(techniques: TechniquePayload) -> list[str]:
    values: list[str] = []
    generic = techniques.generic
    boolean_flags = (
        ("tie_origin", generic.tie_origin),
        ("tie_destination", generic.tie_destination),
        ("legato_origin", generic.legato_origin),
        ("legato_destination", generic.legato_destination),
        ("let_ring", generic.let_ring),
        ("muted", generic.muted),
        ("palm_muted", generic.palm_muted),
    )
    values.extend(name for name, enabled in boolean_flags if enabled)

    optional_values = (
        ("accent", generic.accent),
        ("ornament", generic.ornament),
        ("vibrato", generic.vibrato),
        ("trill", generic.trill),
    )
    values.extend(
        f"{name}={value}" for name, value in optional_values if value is not None
    )
    return values


def _summarize_general_techniques(techniques: TechniquePayload) -> list[str]:
    if techniques.general is None or techniques.general.ornament is None:
        return []
    return [f"general.ornament={techniques.general.ornament}"]


def _summarize_string_fretted_techniques(techniques: TechniquePayload) -> list[str]:
    string_fretted = techniques.string_fretted
    if string_fretted is None:
        return []

    values: list[str] = []
    if string_fretted.slide_types:
        slide_values = ",".join(str(value) for value in string_fretted.slide_types)
        values.append(f"string_fretted.slide_types=[{slide_values}]")

    optional_values = (
        ("string_fretted.hopo_type", string_fretted.hopo_type),
        ("string_fretted.harmonic_type", string_fretted.harmonic_type),
        ("string_fretted.harmonic_kind", string_fretted.harmonic_kind),
    )
    values.extend(
        f"{name}={value}" for name, value in optional_values if value is not None
    )

    if string_fretted.harmonic_fret is not None:
        values.append(f"string_fretted.harmonic_fret={string_fretted.harmonic_fret:g}")

    boolean_flags = (
        ("string_fretted.tapped", string_fretted.tapped),
        ("string_fretted.left_hand_tapped", string_fretted.left_hand_tapped),
        ("string_fretted.bend_enabled", string_fretted.bend_enabled),
        ("string_fretted.whammy_enabled", string_fretted.whammy_enabled),
    )
    values.extend(name for name, enabled in boolean_flags if enabled)
    return values


def _render_anchor_summary(row: ControlEventRow) -> str:
    anchors = [value for value in (row.start_anchor_ref, row.end_anchor_ref) if value]
    if not anchors:
        return "-"
    return " -> ".join(f"`{anchor}`" for anchor in anchors)


def _render_markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
) -> str:
    normalized_headers = [str(header).strip() for header in headers]
    lines = [
        "| "
        + " | ".join(_escape_markdown_cell(header) for header in normalized_headers)
        + " |",
        "| " + " | ".join("---" for _ in normalized_headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |"
        )
    return "\n".join(lines)


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


__all__ = [
    "build_control_event_rows",
    "build_onset_note_tables",
    "build_structure_summary",
    "build_voice_lane_onset_tables",
    "format_pitch",
    "format_score_time",
    "load_ir_document_record",
    "render_control_event_rows_markdown",
    "render_onset_note_table_markdown",
    "render_structure_summary_markdown",
    "render_voice_lane_onset_table_markdown",
    "summarize_control_value",
    "summarize_techniques",
]
