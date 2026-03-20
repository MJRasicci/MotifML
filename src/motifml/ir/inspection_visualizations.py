"""Deterministic static SVG inspection visualizations for MotifML IR documents."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from html import escape

from motifml.ir.ids import bar_sort_key, edge_sort_key, note_sort_key
from motifml.ir.inspection_tables import (
    build_control_event_rows,
    build_structure_summary,
    build_voice_lane_onset_tables,
    format_pitch,
    format_score_time,
)
from motifml.ir.models import EdgeType, MotifMlIrDocument
from motifml.ir.time import ScoreTime

SVG_WIDTH = 960
LEFT_GUTTER = 240
RIGHT_GUTTER = 24
TOP_GUTTER = 52
BOTTOM_GUTTER = 28
ROW_HEIGHT = 30
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 11
SMALL_FONT_SIZE = 10
COMPACT_IDENTIFIER_MIN_PARTS = 4


def render_timeline_plot_svg(
    document: MotifMlIrDocument,
    *,
    width: int = SVG_WIDTH,
) -> str:
    """Render a deterministic overview of the written timeline."""
    bars = _sorted_bars(document)
    summary = build_structure_summary(document)
    time_start, time_end = _timeline_bounds(bars)
    content_width = _content_width(width)
    plot_top = TOP_GUTTER
    plot_bottom = plot_top + 78
    bar_band_top = plot_top + 16
    bar_band_height = 34
    elements = _chart_header(
        "IR Timeline Plot",
        "Written bars with per-bar counts and onset ticks.",
        width,
    )
    elements.extend(
        _render_bar_guides(
            bars=bars,
            time_start=time_start,
            time_end=time_end,
            width=width,
            top=bar_band_top,
            bottom=bar_band_top + bar_band_height,
            show_labels=True,
        )
    )

    onset_tick_top = bar_band_top + bar_band_height + 10
    onset_tick_bottom = onset_tick_top + 12
    for onset in sorted(
        document.onset_groups,
        key=lambda item: (item.time, item.voice_lane_id.casefold(), item.onset_id),
    ):
        x = _time_to_x(onset.time, time_start, time_end, content_width)
        elements.append(
            _line(
                x1=x,
                y1=onset_tick_top,
                x2=x,
                y2=onset_tick_bottom,
                stroke="#1f2937",
                stroke_width=1.5,
            )
        )

    for rollup in summary.bar_rollups:
        bar = next(item for item in bars if item.bar_id == rollup.bar_id)
        x = _time_to_x(bar.start, time_start, time_end, content_width) + 6
        y = bar_band_top + 20
        label = (
            f"bar {rollup.bar_index} | onsets={rollup.onset_count} | "
            f"notes={rollup.note_count}"
        )
        elements.append(_text(x, y, label, fill="#0f172a", font_size=LABEL_FONT_SIZE))

    height = plot_bottom + BOTTOM_GUTTER
    return _svg_document(width, height, elements)


def render_voice_lane_ladder_svg(
    document: MotifMlIrDocument,
    *,
    width: int = SVG_WIDTH,
) -> str:
    """Render onset ladders grouped by voice-lane chain."""
    bars = _sorted_bars(document)
    time_start, time_end = _timeline_bounds(bars)
    content_width = _content_width(width)
    grouped_rows = _group_voice_lane_chain_rows(document)
    chart_height = TOP_GUTTER + (len(grouped_rows) * ROW_HEIGHT) + BOTTOM_GUTTER
    elements = _chart_header(
        "Voice-Lane Onset Ladder",
        "Rows follow voice-lane chains so dropouts and reentries stay visible.",
        width,
    )
    elements.extend(
        _render_bar_guides(
            bars=bars,
            time_start=time_start,
            time_end=time_end,
            width=width,
            top=TOP_GUTTER - 8,
            bottom=chart_height - BOTTOM_GUTTER,
            show_labels=True,
        )
    )

    for row_index, (label, onset_rows) in enumerate(grouped_rows):
        y = TOP_GUTTER + (row_index * ROW_HEIGHT)
        elements.append(_text(12, y + 4, label, fill="#0f172a"))
        elements.append(
            _line(
                x1=LEFT_GUTTER,
                y1=y,
                x2=LEFT_GUTTER + content_width,
                y2=y,
                stroke="#cbd5e1",
                stroke_width=1,
            )
        )
        for onset_row in onset_rows:
            x = _time_to_x(onset_row.time, time_start, time_end, content_width)
            x_end = _time_to_x(
                onset_row.time + onset_row.duration_notated,
                time_start,
                time_end,
                content_width,
            )
            stroke = "#94a3b8" if onset_row.is_rest else "#2563eb"
            dash_array = "4 3" if onset_row.is_rest else None
            elements.append(
                _line(
                    x1=x,
                    y1=y,
                    x2=max(x + 1, x_end),
                    y2=y,
                    stroke=stroke,
                    stroke_width=3,
                    dash_array=dash_array,
                )
            )
            elements.append(
                _circle(
                    cx=x,
                    cy=y,
                    r=4,
                    fill="#ffffff" if onset_row.is_rest else "#2563eb",
                    stroke=stroke,
                    stroke_width=2,
                )
            )
            if onset_row.note_count > 1:
                elements.append(
                    _text(
                        x + 6,
                        y - 7,
                        str(onset_row.note_count),
                        fill="#1d4ed8",
                        font_size=SMALL_FONT_SIZE,
                    )
                )

    return _svg_document(width, chart_height, elements)


def render_note_relations_svg(
    document: MotifMlIrDocument,
    *,
    width: int = SVG_WIDTH,
) -> str:
    """Render note-to-note tie and technique relations over written time."""
    relation_edges = sorted(
        (
            edge
            for edge in document.edges
            if edge.edge_type in {EdgeType.TIE_TO, EdgeType.TECHNIQUE_TO}
        ),
        key=lambda item: edge_sort_key(
            item.source_id,
            item.edge_type.value,
            item.target_id,
        ),
    )
    notes = _sorted_notes(document)
    bars = _sorted_bars(document)
    time_start, time_end = _timeline_bounds(bars)
    content_width = _content_width(width)
    chart_height = TOP_GUTTER + (len(notes) * ROW_HEIGHT) + BOTTOM_GUTTER
    elements = _chart_header(
        "Note Relation Graph",
        "Ties and linked-note techniques over the written timeline.",
        width,
    )
    elements.extend(
        _render_bar_guides(
            bars=bars,
            time_start=time_start,
            time_end=time_end,
            width=width,
            top=TOP_GUTTER - 8,
            bottom=chart_height - BOTTOM_GUTTER,
            show_labels=True,
        )
    )

    note_positions = {}
    for note_index, note in enumerate(notes):
        y = TOP_GUTTER + (note_index * ROW_HEIGHT)
        x = _time_to_x(note.time, time_start, time_end, content_width)
        note_positions[note.note_id] = (x, y)
        label = f"n{note_index} {format_pitch(note.pitch)}"
        elements.append(_text(12, y + 4, label, fill="#0f172a"))
        elements.append(
            _circle(
                cx=x,
                cy=y,
                r=4,
                fill="#111827",
                stroke="#111827",
                stroke_width=1.5,
            )
        )

    if not relation_edges:
        elements.append(
            _text(
                LEFT_GUTTER,
                TOP_GUTTER + 20,
                "No tie_to or technique_to edges in this document.",
                fill="#475569",
            )
        )
        return _svg_document(width, chart_height, elements)

    elements.extend(_render_relation_legend())
    for edge in relation_edges:
        source_x, source_y = note_positions[edge.source_id]
        target_x, target_y = note_positions[edge.target_id]
        stroke = "#16a34a" if edge.edge_type is EdgeType.TIE_TO else "#d97706"
        dash_array = None if edge.edge_type is EdgeType.TIE_TO else "5 4"
        elements.append(
            _line(
                x1=source_x,
                y1=source_y,
                x2=target_x,
                y2=target_y,
                stroke=stroke,
                stroke_width=2,
                dash_array=dash_array,
            )
        )

    return _svg_document(width, chart_height, elements)


def render_control_timeline_svg(
    document: MotifMlIrDocument,
    *,
    width: int = SVG_WIDTH,
) -> str:
    """Render point and span controls as deterministic static SVG."""
    control_rows = build_control_event_rows(document)
    bars = _sorted_bars(document)
    time_start, time_end = _timeline_bounds(bars)
    content_width = _content_width(width)
    chart_height = TOP_GUTTER + (len(control_rows) * ROW_HEIGHT) + BOTTOM_GUTTER
    elements = _chart_header(
        "Control Timeline",
        "Point controls use diamond markers; span controls use horizontal bars.",
        width,
    )
    elements.extend(
        _render_bar_guides(
            bars=bars,
            time_start=time_start,
            time_end=time_end,
            width=width,
            top=TOP_GUTTER - 8,
            bottom=chart_height - BOTTOM_GUTTER,
            show_labels=True,
        )
    )

    for row_index, row in enumerate(control_rows):
        y = TOP_GUTTER + (row_index * ROW_HEIGHT)
        label = f"{row.kind} | {row.scope} | {_compact_identifier(row.target_ref)}"
        elements.append(_text(12, y + 4, label, fill="#0f172a"))
        elements.append(
            _line(
                x1=LEFT_GUTTER,
                y1=y,
                x2=LEFT_GUTTER + content_width,
                y2=y,
                stroke="#e2e8f0",
                stroke_width=1,
            )
        )

        x_start = _time_to_x(row.start_time, time_start, time_end, content_width)
        if row.family == "point":
            elements.append(_diamond(x_start, y, size=5, fill="#7c3aed"))
            elements.append(
                _text(
                    x_start + 8,
                    y - 6,
                    row.value_summary,
                    fill="#581c87",
                    font_size=SMALL_FONT_SIZE,
                )
            )
            continue

        end_time = row.end_time if row.end_time is not None else row.start_time
        x_end = _time_to_x(end_time, time_start, time_end, content_width)
        elements.append(
            _line(
                x1=x_start,
                y1=y,
                x2=max(x_start + 1, x_end),
                y2=y,
                stroke="#dc2626",
                stroke_width=5,
            )
        )
        elements.append(
            _text(
                x_start + 8,
                y - 6,
                row.value_summary,
                fill="#991b1b",
                font_size=SMALL_FONT_SIZE,
            )
        )

    return _svg_document(width, chart_height, elements)


def _group_voice_lane_chain_rows(
    document: MotifMlIrDocument,
) -> list[tuple[str, list[object]]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    labels: dict[str, str] = {}
    for table in build_voice_lane_onset_tables(document):
        labels[table.voice_lane_chain_id] = (
            f"v{table.voice_index} | {_compact_identifier(table.staff_id)}"
        )
        grouped[table.voice_lane_chain_id].extend(table.rows)

    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            grouped[key][0].part_id.casefold(),
            grouped[key][0].staff_id.casefold(),
            grouped[key][0].voice_index,
            key.casefold(),
        ),
    )
    return [
        (
            labels[key],
            sorted(
                grouped[key],
                key=lambda item: (item.time, item.attack_order_in_voice, item.onset_id),
            ),
        )
        for key in ordered_keys
    ]


def _render_bar_guides(  # noqa: PLR0913
    *,
    bars: Sequence[object],
    time_start: ScoreTime,
    time_end: ScoreTime,
    width: int,
    top: int,
    bottom: int,
    show_labels: bool,
) -> list[str]:
    content_width = _content_width(width)
    elements: list[str] = []
    for index, bar in enumerate(bars):
        x_start = _time_to_x(bar.start, time_start, time_end, content_width)
        x_end = _time_to_x(
            bar.start + bar.duration,
            time_start,
            time_end,
            content_width,
        )
        fill = "#f8fafc" if index % 2 == 0 else "#ffffff"
        elements.append(
            _rect(
                x=x_start,
                y=top,
                width=max(1, x_end - x_start),
                height=bottom - top,
                fill=fill,
                stroke="none",
            )
        )
        elements.append(
            _line(
                x1=x_start,
                y1=top,
                x2=x_start,
                y2=bottom,
                stroke="#cbd5e1",
                stroke_width=1,
            )
        )
        if show_labels:
            label = (
                f"bar {bar.bar_index} "
                f"({format_score_time(bar.start)} + {format_score_time(bar.duration)})"
            )
            elements.append(
                _text(
                    x_start + 6,
                    top - 10,
                    label,
                    fill="#334155",
                    font_size=SMALL_FONT_SIZE,
                )
            )

    last_bar = bars[-1]
    x_end = _time_to_x(
        last_bar.start + last_bar.duration,
        time_start,
        time_end,
        content_width,
    )
    elements.append(
        _line(
            x1=x_end,
            y1=top,
            x2=x_end,
            y2=bottom,
            stroke="#cbd5e1",
            stroke_width=1,
        )
    )
    return elements


def _render_relation_legend() -> list[str]:
    return [
        _text(
            12,
            TOP_GUTTER - 16,
            "Legend:",
            fill="#0f172a",
            font_size=SMALL_FONT_SIZE,
        ),
        _line(
            x1=68,
            y1=TOP_GUTTER - 20,
            x2=96,
            y2=TOP_GUTTER - 20,
            stroke="#16a34a",
            stroke_width=2,
        ),
        _text(
            102,
            TOP_GUTTER - 16,
            "tie_to",
            fill="#166534",
            font_size=SMALL_FONT_SIZE,
        ),
        _line(
            x1=154,
            y1=TOP_GUTTER - 20,
            x2=182,
            y2=TOP_GUTTER - 20,
            stroke="#d97706",
            stroke_width=2,
            dash_array="5 4",
        ),
        _text(
            188,
            TOP_GUTTER - 16,
            "technique_to",
            fill="#92400e",
            font_size=SMALL_FONT_SIZE,
        ),
    ]


def _chart_header(title: str, subtitle: str, width: int) -> list[str]:
    return [
        _text(
            12,
            20,
            title,
            fill="#0f172a",
            font_size=TITLE_FONT_SIZE,
            font_weight="600",
        ),
        _text(12, 38, subtitle, fill="#475569", font_size=LABEL_FONT_SIZE),
        _line(
            x1=12,
            y1=44,
            x2=width - RIGHT_GUTTER,
            y2=44,
            stroke="#e2e8f0",
            stroke_width=1,
        ),
    ]


def _timeline_bounds(bars: Sequence[object]) -> tuple[ScoreTime, ScoreTime]:
    first_bar = bars[0]
    last_bar = bars[-1]
    return (first_bar.start, last_bar.start + last_bar.duration)


def _sorted_bars(document: MotifMlIrDocument) -> list[object]:
    return sorted(
        document.bars, key=lambda item: bar_sort_key(item.bar_index, item.bar_id)
    )


def _sorted_notes(document: MotifMlIrDocument) -> list[object]:
    return sorted(
        document.note_events,
        key=lambda item: (
            item.time,
            *note_sort_key(item.string_number, item.pitch, item.note_id),
        ),
    )


def _time_to_x(
    value: ScoreTime,
    time_start: ScoreTime,
    time_end: ScoreTime,
    content_width: int,
) -> int:
    total_duration = time_end - time_start
    if total_duration.numerator == 0:
        return LEFT_GUTTER

    offset = value - time_start
    ratio = offset.to_float() / total_duration.to_float()
    return LEFT_GUTTER + round(ratio * content_width)


def _content_width(width: int) -> int:
    return width - LEFT_GUTTER - RIGHT_GUTTER


def _compact_identifier(identifier: str) -> str:
    parts = identifier.split(":")
    if len(parts) <= COMPACT_IDENTIFIER_MIN_PARTS:
        return identifier
    return ":".join((parts[0], *parts[-3:]))


def _svg_document(width: int, height: int, elements: Sequence[str]) -> str:
    body = "\n".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="MotifML IR inspection visualization">'
        f"<style>{_svg_style()}</style>{body}</svg>"
    )


def _svg_style() -> str:
    return (
        "text { font-family: monospace; dominant-baseline: middle; } "
        "line, rect, circle, polygon { vector-effect: non-scaling-stroke; }"
    )


def _text(  # noqa: PLR0913
    x: int,
    y: int,
    value: str,
    *,
    fill: str,
    font_size: int = LABEL_FONT_SIZE,
    font_weight: str | None = None,
) -> str:
    font_weight_attr = "" if font_weight is None else f' font-weight="{font_weight}"'
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{font_size}"'
        f"{font_weight_attr}>{escape(value)}</text>"
    )


def _line(  # noqa: PLR0913
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    stroke: str,
    stroke_width: float,
    dash_array: str | None = None,
) -> str:
    dash_attr = "" if dash_array is None else f' stroke-dasharray="{dash_array}"'
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr} />'
    )


def _rect(  # noqa: PLR0913
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    fill: str,
    stroke: str,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
        f'stroke="{stroke}" />'
    )


def _circle(  # noqa: PLR0913
    *,
    cx: int,
    cy: int,
    r: int,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" />'
    )


def _diamond(x: int, y: int, *, size: int, fill: str) -> str:
    points = f"{x},{y - size} {x + size},{y} {x},{y + size} {x - size},{y}"
    return f'<polygon points="{points}" fill="{fill}" stroke="{fill}" />'


__all__ = [
    "render_control_timeline_svg",
    "render_note_relations_svg",
    "render_timeline_plot_svg",
    "render_voice_lane_ladder_svg",
]
