"""Rule-based structural validation for MotifML IR documents."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Any

from motifml.ir.ids import note_sort_key
from motifml.ir.ids import voice_lane_chain_id as build_voice_lane_chain_id
from motifml.ir.models import EdgeType, MotifMlIrDocument, PhraseSpan
from motifml.ir.time import ScoreTime

FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "title",
        "artist",
        "album",
        "composer",
        "copyright",
        "comments",
        "comment",
        "track_name",
        "track_names",
        "free_text",
        "freetext",
    }
)


class IrValidationSeverity(StrEnum):
    """Configurable validation severities for IR invariants."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


class IrValidationRule(StrEnum):
    """Supported structural invariant rules for canonical IR documents."""

    ONSET_OWNERSHIP = "onset_ownership"
    NOTE_OWNERSHIP = "note_ownership"
    NOTE_TIME_ALIGNMENT = "note_time_alignment"
    VOICE_LANE_ONSET_TIMING = "voice_lane_onset_timing"
    ATTACK_ORDER_CONTIGUITY = "attack_order_contiguity"
    SOUNDING_DURATION_POSITIVE = "sounding_duration_positive"
    TIE_CHAIN_LINEAR = "tie_chain_linear"
    VOICE_LANE_CHAIN_STABILITY = "voice_lane_chain_stability"
    NOTE_ORDER_CANONICAL = "note_order_canonical"
    EDGE_ENDPOINT_REFERENCE_INTEGRITY = "edge_endpoint_reference_integrity"
    FORBIDDEN_METADATA_ABSENT = "forbidden_metadata_absent"
    PHRASE_SPAN_VALIDITY = "phrase_span_validity"
    FRETTED_STRING_COLLISION = "fretted_string_collision"


DEFAULT_RULE_SEVERITIES: dict[IrValidationRule, IrValidationSeverity] = {
    rule: IrValidationSeverity.ERROR for rule in IrValidationRule
}


@dataclass(frozen=True, slots=True)
class IrValidationIssue:
    """One concrete invariant violation within a single IR document."""

    path: str
    message: str
    entity_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_text(self.path, "path"))
        object.__setattr__(self, "message", _normalize_text(self.message, "message"))
        if self.entity_id is not None:
            object.__setattr__(
                self,
                "entity_id",
                _normalize_text(self.entity_id, "entity_id"),
            )

    def sort_key(self) -> tuple[str, str, str]:
        """Return a stable sort key for grouped issue ordering."""
        return (
            self.path,
            "" if self.entity_id is None else self.entity_id,
            self.message,
        )


@dataclass(frozen=True, slots=True)
class IrValidationRuleReport:
    """Grouped issues for one validation rule within a document."""

    rule: IrValidationRule
    severity: IrValidationSeverity
    issues: tuple[IrValidationIssue, ...] = ()
    issue_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule",
            IrValidationRule(self.rule),
        )
        object.__setattr__(
            self,
            "severity",
            IrValidationSeverity(self.severity),
        )
        normalized_issues = tuple(
            sorted(self.issues, key=lambda issue: issue.sort_key())
        )
        object.__setattr__(self, "issues", normalized_issues)
        object.__setattr__(self, "issue_count", len(normalized_issues))


@dataclass(frozen=True, slots=True)
class IrDocumentValidationReport:
    """Typed validation report for one IR document."""

    relative_path: str
    source_hash: str
    rule_reports: tuple[IrValidationRuleReport, ...] = ()
    passed: bool = field(init=False)
    error_count: int = field(init=False)
    warning_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_text(self.source_hash, "source_hash"),
        )
        sorted_reports = tuple(
            sorted(self.rule_reports, key=lambda report: report.rule.value)
        )
        object.__setattr__(self, "rule_reports", sorted_reports)
        error_count = sum(
            report.issue_count
            for report in sorted_reports
            if report.severity is IrValidationSeverity.ERROR
        )
        warning_count = sum(
            report.issue_count
            for report in sorted_reports
            if report.severity is IrValidationSeverity.WARN
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


@dataclass(frozen=True, slots=True)
class _ValidationContext:
    """Cached lookup state used across validation rules."""

    voice_lane_by_id: dict[str, object]
    bar_by_id: dict[str, object]
    staff_by_id: dict[str, object]
    onset_by_id: dict[str, object]
    note_by_id: dict[str, object]
    onset_id_counts: Counter[str]
    note_id_counts: Counter[str]
    onsets_by_voice_lane: dict[str, tuple[tuple[int, object], ...]]
    notes_by_onset: dict[str, tuple[tuple[int, object], ...]]
    contains_sources_by_target: dict[str, tuple[str, ...]]
    tie_edge_ordinals: dict[tuple[str, str], int]
    tie_outgoing: dict[str, tuple[str, ...]]
    tie_incoming: dict[str, tuple[str, ...]]
    all_entity_ids: frozenset[str]
    voice_lane_chain_ids: frozenset[str]
    fretted_staff_ids: frozenset[str]


def validate_document(
    document: MotifMlIrDocument,
) -> dict[IrValidationRule, tuple[IrValidationIssue, ...]]:
    """Return grouped invariant violations for a single IR document."""
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]] = (
        defaultdict(list)
    )
    context = _build_validation_context(document)

    _validate_onset_ownership(document, context, issues_by_rule)
    _validate_note_ownership(document, context, issues_by_rule)
    _validate_note_time_alignment(document, context, issues_by_rule)
    _validate_voice_lane_onset_timing(document, context, issues_by_rule)
    _validate_attack_order_contiguity(document, context, issues_by_rule)
    _validate_sounding_duration_positive(document, issues_by_rule)
    _validate_tie_chain_linearity(context, issues_by_rule)
    _validate_voice_lane_chain_stability(document, issues_by_rule)
    _validate_note_order_canonical(document, context, issues_by_rule)
    _validate_edge_endpoint_integrity(document, context, issues_by_rule)
    _validate_forbidden_metadata_absence(document, issues_by_rule)
    _validate_phrase_spans(document, context, issues_by_rule)
    _validate_fretted_string_collisions(document, context, issues_by_rule)

    return {
        rule: tuple(sorted(issues, key=lambda issue: issue.sort_key()))
        for rule, issues in issues_by_rule.items()
        if issues
    }


def build_document_validation_report(
    *,
    relative_path: str,
    source_hash: str,
    document: MotifMlIrDocument,
    rule_severities: Mapping[IrValidationRule | str, IrValidationSeverity | str]
    | None = None,
) -> IrDocumentValidationReport:
    """Validate one document and materialize a grouped report."""
    issues_by_rule = validate_document(document)
    severities = coerce_rule_severities(rule_severities)

    rule_reports = [
        IrValidationRuleReport(
            rule=rule,
            severity=severity,
            issues=issues,
        )
        for rule, issues in issues_by_rule.items()
        for severity in (severities[rule],)
        if severity is not IrValidationSeverity.IGNORE
    ]
    return IrDocumentValidationReport(
        relative_path=relative_path,
        source_hash=source_hash,
        rule_reports=tuple(rule_reports),
    )


def coerce_rule_severities(
    rule_severities: Mapping[IrValidationRule | str, IrValidationSeverity | str] | None,
) -> dict[IrValidationRule, IrValidationSeverity]:
    """Normalize per-rule severities, defaulting every rule to `error`."""
    normalized = dict(DEFAULT_RULE_SEVERITIES)
    if rule_severities is None:
        return normalized

    for raw_rule, raw_severity in rule_severities.items():
        rule = IrValidationRule(raw_rule)
        normalized[rule] = _coerce_validation_severity(raw_severity)

    return normalized


def _build_validation_context(document: MotifMlIrDocument) -> _ValidationContext:
    voice_lane_by_id = {
        voice_lane.voice_lane_id: voice_lane for voice_lane in document.voice_lanes
    }
    bar_by_id = {bar.bar_id: bar for bar in document.bars}
    staff_by_id = {staff.staff_id: staff for staff in document.staves}
    onset_by_id = {onset.onset_id: onset for onset in document.onset_groups}
    note_by_id = {note.note_id: note for note in document.note_events}
    onset_id_counts = Counter(onset.onset_id for onset in document.onset_groups)
    note_id_counts = Counter(note.note_id for note in document.note_events)

    onsets_by_voice_lane: defaultdict[str, list[tuple[int, object]]] = defaultdict(list)
    for onset_index, onset in enumerate(document.onset_groups):
        onsets_by_voice_lane[onset.voice_lane_id].append((onset_index, onset))

    notes_by_onset: defaultdict[str, list[tuple[int, object]]] = defaultdict(list)
    for note_index, note in enumerate(document.note_events):
        notes_by_onset[note.onset_id].append((note_index, note))

    contains_sources_by_target: defaultdict[str, list[str]] = defaultdict(list)
    tie_edge_ordinals: dict[tuple[str, str], int] = {}
    tie_outgoing: defaultdict[str, list[str]] = defaultdict(list)
    tie_incoming: defaultdict[str, list[str]] = defaultdict(list)
    for edge_index, edge in enumerate(document.edges):
        if edge.edge_type is EdgeType.CONTAINS:
            contains_sources_by_target[edge.target_id].append(edge.source_id)
        elif edge.edge_type is EdgeType.TIE_TO:
            tie_edge_ordinals[(edge.source_id, edge.target_id)] = edge_index
            tie_outgoing[edge.source_id].append(edge.target_id)
            tie_incoming[edge.target_id].append(edge.source_id)

    all_entity_ids = (
        {part.part_id for part in document.parts}
        | {staff.staff_id for staff in document.staves}
        | {bar.bar_id for bar in document.bars}
        | {voice_lane.voice_lane_id for voice_lane in document.voice_lanes}
        | {onset.onset_id for onset in document.onset_groups}
        | {note.note_id for note in document.note_events}
        | {control.control_id for control in document.point_control_events}
        | {control.control_id for control in document.span_control_events}
    )
    for phrase_span in document.optional_overlays.phrase_spans:
        if isinstance(phrase_span, Mapping) and "phrase_id" in phrase_span:
            all_entity_ids.add(str(phrase_span["phrase_id"]))

    string_assigned_staff_ids = {
        note.staff_id for note in document.note_events if note.string_number is not None
    }
    fretted_staff_ids = {
        staff.staff_id
        for staff in document.staves
        if staff.tuning_pitches is not None or staff.capo_fret is not None
    } | string_assigned_staff_ids

    return _ValidationContext(
        voice_lane_by_id=voice_lane_by_id,
        bar_by_id=bar_by_id,
        staff_by_id=staff_by_id,
        onset_by_id=onset_by_id,
        note_by_id=note_by_id,
        onset_id_counts=onset_id_counts,
        note_id_counts=note_id_counts,
        onsets_by_voice_lane={
            voice_lane_id: tuple(entries)
            for voice_lane_id, entries in onsets_by_voice_lane.items()
        },
        notes_by_onset={
            onset_id: tuple(entries) for onset_id, entries in notes_by_onset.items()
        },
        contains_sources_by_target={
            target_id: tuple(sorted(source_ids))
            for target_id, source_ids in contains_sources_by_target.items()
        },
        tie_edge_ordinals=tie_edge_ordinals,
        tie_outgoing={
            source_id: tuple(sorted(target_ids))
            for source_id, target_ids in tie_outgoing.items()
        },
        tie_incoming={
            target_id: tuple(sorted(source_ids))
            for target_id, source_ids in tie_incoming.items()
        },
        all_entity_ids=frozenset(all_entity_ids),
        voice_lane_chain_ids=frozenset(
            voice_lane.voice_lane_chain_id for voice_lane in document.voice_lanes
        ),
        fretted_staff_ids=frozenset(fretted_staff_ids),
    )


def _validate_onset_ownership(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for onset_index, onset in enumerate(document.onset_groups):
        path_prefix = f"onset_groups[{onset_index}]"
        if context.onset_id_counts[onset.onset_id] != 1:
            _append_issue(
                issues_by_rule,
                IrValidationRule.ONSET_OWNERSHIP,
                path=f"{path_prefix}.onset_id",
                message=(
                    f"onset id '{onset.onset_id}' must appear exactly once in the "
                    "document."
                ),
                entity_id=onset.onset_id,
            )

        voice_lane = context.voice_lane_by_id.get(onset.voice_lane_id)
        if voice_lane is None:
            _append_issue(
                issues_by_rule,
                IrValidationRule.ONSET_OWNERSHIP,
                path=f"{path_prefix}.voice_lane_id",
                message=(
                    f"onset '{onset.onset_id}' references missing voice lane "
                    f"'{onset.voice_lane_id}'."
                ),
                entity_id=onset.onset_id,
            )
        elif onset.bar_id != voice_lane.bar_id:
            _append_issue(
                issues_by_rule,
                IrValidationRule.ONSET_OWNERSHIP,
                path=f"{path_prefix}.bar_id",
                message=(
                    f"onset '{onset.onset_id}' bar_id '{onset.bar_id}' does not "
                    f"match parent voice lane bar '{voice_lane.bar_id}'."
                ),
                entity_id=onset.onset_id,
            )

        if onset.bar_id not in context.bar_by_id:
            _append_issue(
                issues_by_rule,
                IrValidationRule.ONSET_OWNERSHIP,
                path=f"{path_prefix}.bar_id",
                message=(
                    f"onset '{onset.onset_id}' references missing bar '{onset.bar_id}'."
                ),
                entity_id=onset.onset_id,
            )

        contains_sources = context.contains_sources_by_target.get(onset.onset_id, ())
        if len(contains_sources) != 1 or contains_sources[0] != onset.voice_lane_id:
            _append_issue(
                issues_by_rule,
                IrValidationRule.ONSET_OWNERSHIP,
                path=path_prefix,
                message=(
                    f"onset '{onset.onset_id}' must be contained by exactly one "
                    "owning voice lane edge."
                ),
                entity_id=onset.onset_id,
            )


def _validate_note_ownership(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for note_index, note in enumerate(document.note_events):
        path_prefix = f"note_events[{note_index}]"
        if context.note_id_counts[note.note_id] != 1:
            _append_issue(
                issues_by_rule,
                IrValidationRule.NOTE_OWNERSHIP,
                path=f"{path_prefix}.note_id",
                message=(
                    f"note id '{note.note_id}' must appear exactly once in the document."
                ),
                entity_id=note.note_id,
            )

        if note.onset_id not in context.onset_by_id:
            _append_issue(
                issues_by_rule,
                IrValidationRule.NOTE_OWNERSHIP,
                path=f"{path_prefix}.onset_id",
                message=(
                    f"note '{note.note_id}' references missing onset '{note.onset_id}'."
                ),
                entity_id=note.note_id,
            )

        contains_sources = context.contains_sources_by_target.get(note.note_id, ())
        if len(contains_sources) != 1 or contains_sources[0] != note.onset_id:
            _append_issue(
                issues_by_rule,
                IrValidationRule.NOTE_OWNERSHIP,
                path=path_prefix,
                message=(
                    f"note '{note.note_id}' must be contained by exactly one owning "
                    "onset-group edge."
                ),
                entity_id=note.note_id,
            )


def _validate_note_time_alignment(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for note_index, note in enumerate(document.note_events):
        onset = context.onset_by_id.get(note.onset_id)
        if onset is None or note.time == onset.time:
            continue

        _append_issue(
            issues_by_rule,
            IrValidationRule.NOTE_TIME_ALIGNMENT,
            path=f"note_events[{note_index}].time",
            message=(
                f"note '{note.note_id}' time {note.time} does not match parent onset "
                f"time {onset.time}."
            ),
            entity_id=note.note_id,
        )


def _validate_voice_lane_onset_timing(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    del document

    for voice_lane_id, onset_entries in context.onsets_by_voice_lane.items():
        ordered = sorted(
            onset_entries, key=lambda entry: entry[1].attack_order_in_voice
        )
        previous_onset = None
        for onset_index, onset in ordered:
            if previous_onset is not None and onset.time <= previous_onset.time:
                _append_issue(
                    issues_by_rule,
                    IrValidationRule.VOICE_LANE_ONSET_TIMING,
                    path=f"onset_groups[{onset_index}].time",
                    message=(
                        f"voice lane '{voice_lane_id}' onset times must be unique and "
                        "strictly increasing by attack order."
                    ),
                    entity_id=onset.onset_id,
                )
            previous_onset = onset


def _validate_attack_order_contiguity(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    del document

    for voice_lane_id, onset_entries in context.onsets_by_voice_lane.items():
        ordered = sorted(
            onset_entries, key=lambda entry: entry[1].attack_order_in_voice
        )
        attack_orders = [onset.attack_order_in_voice for _, onset in ordered]
        expected_attack_orders = list(range(len(ordered)))
        if attack_orders != expected_attack_orders:
            path = f"voice_lanes[{_voice_lane_index(context, voice_lane_id)}]"
            _append_issue(
                issues_by_rule,
                IrValidationRule.ATTACK_ORDER_CONTIGUITY,
                path=path,
                message=(
                    f"voice lane '{voice_lane_id}' attack_order_in_voice values must be "
                    "contiguous starting at zero."
                ),
                entity_id=voice_lane_id,
            )


def _validate_sounding_duration_positive(
    document: MotifMlIrDocument,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for note_index, note in enumerate(document.note_events):
        if note.sounding_duration.numerator > 0:
            continue

        _append_issue(
            issues_by_rule,
            IrValidationRule.SOUNDING_DURATION_POSITIVE,
            path=f"note_events[{note_index}].sounding_duration",
            message=f"note '{note.note_id}' sounding_duration must be positive.",
            entity_id=note.note_id,
        )


def _validate_tie_chain_linearity(
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for source_id, target_ids in context.tie_outgoing.items():
        if len(target_ids) <= 1:
            continue

        edge_index = context.tie_edge_ordinals[(source_id, target_ids[0])]
        _append_issue(
            issues_by_rule,
            IrValidationRule.TIE_CHAIN_LINEAR,
            path=f"edges[{edge_index}]",
            message=f"tie source '{source_id}' must not branch to multiple targets.",
            entity_id=source_id,
        )

    for target_id, source_ids in context.tie_incoming.items():
        if len(source_ids) <= 1:
            continue

        edge_index = context.tie_edge_ordinals[(source_ids[0], target_id)]
        _append_issue(
            issues_by_rule,
            IrValidationRule.TIE_CHAIN_LINEAR,
            path=f"edges[{edge_index}]",
            message=f"tie target '{target_id}' must not have multiple incoming ties.",
            entity_id=target_id,
        )

    for source_id, target_ids in context.tie_outgoing.items():
        if len(target_ids) != 1:
            continue

        seen: set[str] = set()
        current = source_id
        while True:
            next_targets = context.tie_outgoing.get(current, ())
            if len(next_targets) != 1:
                break

            next_target = next_targets[0]
            edge_index = context.tie_edge_ordinals[(current, next_target)]
            if next_target == source_id or next_target in seen:
                _append_issue(
                    issues_by_rule,
                    IrValidationRule.TIE_CHAIN_LINEAR,
                    path=f"edges[{edge_index}]",
                    message="tie chains must remain linear and acyclic.",
                    entity_id=source_id,
                )
                break

            seen.add(current)
            current = next_target


def _validate_voice_lane_chain_stability(
    document: MotifMlIrDocument,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for voice_lane_index, voice_lane in enumerate(document.voice_lanes):
        expected_chain_id = build_voice_lane_chain_id(
            voice_lane.part_id,
            voice_lane.staff_id,
            voice_lane.voice_index,
        )
        if voice_lane.voice_lane_chain_id == expected_chain_id:
            continue

        _append_issue(
            issues_by_rule,
            IrValidationRule.VOICE_LANE_CHAIN_STABILITY,
            path=f"voice_lanes[{voice_lane_index}].voice_lane_chain_id",
            message=(
                f"voice lane '{voice_lane.voice_lane_id}' has unstable chain id "
                f"'{voice_lane.voice_lane_chain_id}'; expected '{expected_chain_id}'."
            ),
            entity_id=voice_lane.voice_lane_id,
        )


def _validate_note_order_canonical(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    del document

    for onset_id, note_entries in context.notes_by_onset.items():
        if len(note_entries) <= 1:
            continue

        note_ids_in_document_order = [note.note_id for _, note in note_entries]
        canonical_note_ids = [
            note.note_id
            for _, note in sorted(
                note_entries,
                key=lambda entry: note_sort_key(
                    entry[1].string_number,
                    entry[1].pitch,
                    entry[1].note_id,
                ),
            )
        ]
        if note_ids_in_document_order == canonical_note_ids:
            continue

        note_index = note_entries[0][0]
        _append_issue(
            issues_by_rule,
            IrValidationRule.NOTE_ORDER_CANONICAL,
            path=f"note_events[{note_index}]",
            message=(
                f"notes within onset '{onset_id}' must follow canonical sort order."
            ),
            entity_id=onset_id,
        )


def _validate_edge_endpoint_integrity(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for edge_index, edge in enumerate(document.edges):
        if edge.source_id not in context.all_entity_ids:
            _append_issue(
                issues_by_rule,
                IrValidationRule.EDGE_ENDPOINT_REFERENCE_INTEGRITY,
                path=f"edges[{edge_index}].source_id",
                message=(
                    f"edge source '{edge.source_id}' does not reference a known entity."
                ),
                entity_id=edge.source_id,
            )
        if edge.target_id not in context.all_entity_ids:
            _append_issue(
                issues_by_rule,
                IrValidationRule.EDGE_ENDPOINT_REFERENCE_INTEGRITY,
                path=f"edges[{edge_index}].target_id",
                message=(
                    f"edge target '{edge.target_id}' does not reference a known entity."
                ),
                entity_id=edge.target_id,
            )


def _validate_forbidden_metadata_absence(
    document: MotifMlIrDocument,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for path, key in _iter_forbidden_metadata_paths(document):
        _append_issue(
            issues_by_rule,
            IrValidationRule.FORBIDDEN_METADATA_ABSENT,
            path=path,
            message=f"forbidden metadata field '{key}' must not appear in the IR.",
            entity_id=key,
        )


def _validate_phrase_spans(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    for phrase_index, phrase_span in enumerate(document.optional_overlays.phrase_spans):
        path_prefix = f"optional_overlays.phrase_spans[{phrase_index}]"
        if isinstance(phrase_span, PhraseSpan):
            phrase_id = phrase_span.phrase_id
            scope_ref = phrase_span.scope_ref
            start_time = phrase_span.start_time
            end_time = phrase_span.end_time
            voice_lane_chain_id = phrase_span.voice_lane_chain_id
        elif isinstance(phrase_span, Mapping):
            phrase_id = _required_phrase_text(
                phrase_span,
                "phrase_id",
                path_prefix,
                issues_by_rule,
            )
            scope_ref = _required_phrase_text(
                phrase_span,
                "scope_ref",
                path_prefix,
                issues_by_rule,
            )
            start_time = _coerce_phrase_score_time(
                phrase_span.get("start_time"),
                path=f"{path_prefix}.start_time",
                issues_by_rule=issues_by_rule,
            )
            end_time = _coerce_phrase_score_time(
                phrase_span.get("end_time"),
                path=f"{path_prefix}.end_time",
                issues_by_rule=issues_by_rule,
            )
            voice_lane_chain_id = phrase_span.get("voice_lane_chain_id")
        else:
            _append_issue(
                issues_by_rule,
                IrValidationRule.PHRASE_SPAN_VALIDITY,
                path=path_prefix,
                message="phrase spans must be PhraseSpan objects when present.",
            )
            continue

        if start_time is not None and end_time is not None and end_time <= start_time:
            _append_issue(
                issues_by_rule,
                IrValidationRule.PHRASE_SPAN_VALIDITY,
                path=path_prefix,
                message="phrase spans must satisfy start_time < end_time.",
                entity_id=phrase_id,
            )

        if scope_ref is not None and not _is_valid_phrase_scope_ref(scope_ref, context):
            _append_issue(
                issues_by_rule,
                IrValidationRule.PHRASE_SPAN_VALIDITY,
                path=f"{path_prefix}.scope_ref",
                message=f"phrase span scope_ref '{scope_ref}' does not reference a valid scope.",
                entity_id=phrase_id,
            )

        if voice_lane_chain_id is not None and (
            not isinstance(voice_lane_chain_id, str)
            or voice_lane_chain_id not in context.voice_lane_chain_ids
        ):
            _append_issue(
                issues_by_rule,
                IrValidationRule.PHRASE_SPAN_VALIDITY,
                path=f"{path_prefix}.voice_lane_chain_id",
                message=(
                    "phrase span voice_lane_chain_id must reference a known "
                    "voice-lane chain when provided."
                ),
                entity_id=phrase_id,
            )


def _validate_fretted_string_collisions(
    document: MotifMlIrDocument,
    context: _ValidationContext,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> None:
    del document

    for onset_id, note_entries in context.notes_by_onset.items():
        string_counts: Counter[int] = Counter()
        for _, note in note_entries:
            if (
                note.staff_id in context.fretted_staff_ids
                and note.string_number is not None
            ):
                string_counts[note.string_number] += 1

        for string_number, count in string_counts.items():
            if count <= 1:
                continue

            note_index = note_entries[0][0]
            _append_issue(
                issues_by_rule,
                IrValidationRule.FRETTED_STRING_COLLISION,
                path=f"note_events[{note_index}]",
                message=(
                    f"onset '{onset_id}' assigns string number {string_number} to "
                    "multiple notes on a fretted staff."
                ),
                entity_id=onset_id,
            )


def _required_phrase_text(
    phrase_span: Mapping[str, Any],
    field_name: str,
    path_prefix: str,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> str | None:
    value = phrase_span.get(field_name)
    if not isinstance(value, str) or not value.strip():
        _append_issue(
            issues_by_rule,
            IrValidationRule.PHRASE_SPAN_VALIDITY,
            path=f"{path_prefix}.{field_name}",
            message=f"phrase spans require a non-empty '{field_name}' field.",
        )
        return None

    return value.strip()


def _coerce_phrase_score_time(
    value: Any,
    *,
    path: str,
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
) -> ScoreTime | None:
    if isinstance(value, ScoreTime):
        return value

    if not isinstance(value, Mapping):
        _append_issue(
            issues_by_rule,
            IrValidationRule.PHRASE_SPAN_VALIDITY,
            path=path,
            message="phrase span times must be ScoreTime mappings.",
        )
        return None

    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        _append_issue(
            issues_by_rule,
            IrValidationRule.PHRASE_SPAN_VALIDITY,
            path=path,
            message="phrase span times must include integer numerator and denominator.",
        )
        return None

    try:
        return ScoreTime(numerator=numerator, denominator=denominator)
    except ValueError as exc:
        _append_issue(
            issues_by_rule,
            IrValidationRule.PHRASE_SPAN_VALIDITY,
            path=path,
            message=str(exc),
        )
        return None


def _is_valid_phrase_scope_ref(scope_ref: str, context: _ValidationContext) -> bool:
    if scope_ref.startswith("part:"):
        return scope_ref in context.all_entity_ids
    if scope_ref.startswith("staff:"):
        return scope_ref in context.all_entity_ids
    if scope_ref.startswith("voice-chain:"):
        return scope_ref in context.voice_lane_chain_ids

    return False


def _iter_forbidden_metadata_paths(
    value: Any,
    *,
    path: str = "",
) -> list[tuple[str, str]]:
    matches: list[tuple[str, str]] = []
    if is_dataclass(value):
        for field_info in fields(value):
            item = getattr(value, field_info.name)
            next_path = field_info.name if not path else f"{path}.{field_info.name}"
            if field_info.name.casefold() in FORBIDDEN_METADATA_FIELDS:
                matches.append((next_path, field_info.name))
            matches.extend(_iter_forbidden_metadata_paths(item, path=next_path))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_path = key_text if not path else f"{path}.{key_text}"
            if key_text.casefold() in FORBIDDEN_METADATA_FIELDS:
                matches.append((next_path, key_text))
            matches.extend(_iter_forbidden_metadata_paths(item, path=next_path))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            next_path = f"{path}[{index}]"
            matches.extend(_iter_forbidden_metadata_paths(item, path=next_path))

    return matches


def _append_issue(
    issues_by_rule: defaultdict[IrValidationRule, list[IrValidationIssue]],
    rule: IrValidationRule,
    *,
    path: str,
    message: str,
    entity_id: str | None = None,
) -> None:
    issues_by_rule[rule].append(
        IrValidationIssue(path=path, message=message, entity_id=entity_id)
    )


def _voice_lane_index(context: _ValidationContext, voice_lane_id: str) -> int:
    for index, candidate_id in enumerate(context.voice_lane_by_id):
        if candidate_id == voice_lane_id:
            return index

    return 0


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


def _coerce_validation_severity(
    value: IrValidationSeverity | str,
) -> IrValidationSeverity:
    if isinstance(value, IrValidationSeverity):
        return value

    normalized = str(value).strip().casefold()
    if normalized == "warning":
        normalized = IrValidationSeverity.WARN.value

    return IrValidationSeverity(normalized)


__all__ = [
    "DEFAULT_RULE_SEVERITIES",
    "IrDocumentValidationReport",
    "IrValidationIssue",
    "IrValidationRule",
    "IrValidationRuleReport",
    "IrValidationSeverity",
    "build_document_validation_report",
    "coerce_rule_severities",
    "validate_document",
]
