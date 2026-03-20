"""Typed value models shared across the MotifML IR."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, TypeVar

from motifml.ir.ids import (
    BAR_PREFIX,
    NOTE_PREFIX,
    ONSET_PREFIX,
    PART_PREFIX,
    PHRASE_PREFIX,
    POINT_CONTROL_PREFIX,
    SPAN_CONTROL_PREFIX,
    STAFF_PREFIX,
    VOICE_LANE_CHAIN_PREFIX,
    VOICE_LANE_PREFIX,
    edge_sort_key,
)
from motifml.ir.time import ScoreTime

_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class RhythmBaseValue(StrEnum):
    """Canonical onset-level note-value families preserved from the source score."""

    UNKNOWN = "Unknown"
    WHOLE = "Whole"
    HALF = "Half"
    QUARTER = "Quarter"
    EIGHTH = "Eighth"
    SIXTEENTH = "Sixteenth"
    THIRTY_SECOND = "ThirtySecond"
    SIXTY_FOURTH = "SixtyFourth"
    ONE_HUNDRED_TWENTY_EIGHTH = "OneHundredTwentyEighth"
    TWO_HUNDRED_FIFTY_SIXTH = "TwoHundredFiftySixth"


class PitchStep(StrEnum):
    """Canonical diatonic pitch-step labels."""

    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    A = "A"
    B = "B"


class HairpinDirection(StrEnum):
    """Supported v1 hairpin directions."""

    CRESCENDO = "crescendo"
    DECRESCENDO = "decrescendo"


class ControlScope(StrEnum):
    """Supported scope targets for score control events."""

    SCORE = "score"
    PART = "part"
    STAFF = "staff"
    VOICE = "voice"


class PointControlKind(StrEnum):
    """Supported v1 point control kinds."""

    TEMPO_CHANGE = "tempo_change"
    DYNAMIC_CHANGE = "dynamic_change"
    FERMATA = "fermata"


class SpanControlKind(StrEnum):
    """Supported v1 span control kinds."""

    HAIRPIN = "hairpin"
    OTTAVA = "ottava"


class EdgeType(StrEnum):
    """Intrinsic edge families allowed in the canonical IR document."""

    CONTAINS = "contains"
    NEXT_IN_VOICE = "next_in_voice"
    TIE_TO = "tie_to"
    TECHNIQUE_TO = "technique_to"


class DerivedEdgeType(StrEnum):
    """Optional derived edge families that may be materialized in view layers."""

    VERTICAL_OVERLAP = "vertical_overlap"
    MELODIC_INTERVAL_TO = "melodic_interval_to"
    HARMONIC_INTERVAL_TO = "harmonic_interval_to"
    RECURS_WITH = "recurs_with"
    PLAYBACK_NEXT = "playback_next"
    NEXT_PHRASE = "next_phrase"
    REPEATS = "repeats"
    VARIES = "varies"
    ALIGNS_WITH = "aligns_with"


class TimeUnit(StrEnum):
    """Supported canonical score-time units for persisted IR documents."""

    WHOLE_NOTE_FRACTION = "whole_note_fraction"


class PhraseKind(StrEnum):
    """Supported coarse phrase-overlay categories."""

    MELODIC = "melodic"
    ACCOMPANIMENT = "accompaniment"
    RIFF = "riff"
    CADENTIAL = "cadential"
    GESTURE = "gesture"
    PATTERN = "pattern"
    UNKNOWN = "unknown"


class PhraseSource(StrEnum):
    """Supported provenance labels for phrase overlays."""

    AUTHORED = "authored"
    MANUAL_ANNOTATION = "manual_annotation"
    DERIVED_RULE_BASED = "derived_rule_based"
    DERIVED_MODEL_BASED = "derived_model_based"


class IrManifestDiagnosticCategory(StrEnum):
    """High-level categories for grouped IR build diagnostics in the manifest."""

    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    EXCLUDED = "excluded"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Transposition:
    """Written-to-sounding transposition context for one part."""

    chromatic: int = 0
    octave: int = 0

    @property
    def written_minus_sounding_semitones(self) -> int:
        """Return the derived semitone offset preserved by the source model."""
        return self.chromatic + (self.octave * 12)


@dataclass(frozen=True, slots=True)
class TimeSignature:
    """Canonical bar-local time signature."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError("TimeSignature numerator must be greater than zero.")

        if self.denominator <= 0:
            raise ValueError("TimeSignature denominator must be greater than zero.")

    @property
    def bar_duration(self) -> ScoreTime:
        """Return the whole-note duration implied by the meter."""
        return ScoreTime(self.numerator, self.denominator)


@dataclass(frozen=True, slots=True)
class TupletRatio:
    """Tuplet ratio metadata used by written rhythm shapes."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError("TupletRatio numerator must be greater than zero.")

        if self.denominator <= 0:
            raise ValueError("TupletRatio denominator must be greater than zero.")


@dataclass(frozen=True, slots=True)
class RhythmShape:
    """Onset-level written rhythm shape metadata."""

    base_value: RhythmBaseValue
    augmentation_dots: int = 0
    primary_tuplet: TupletRatio | None = None
    secondary_tuplet: TupletRatio | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_value",
            _coerce_str_enum(self.base_value, RhythmBaseValue, "base_value"),
        )
        if self.augmentation_dots < 0:
            raise ValueError("RhythmShape augmentation_dots must be non-negative.")

        if self.secondary_tuplet is not None and self.primary_tuplet is None:
            raise ValueError("RhythmShape secondary_tuplet requires a primary_tuplet.")


@dataclass(frozen=True, slots=True)
class Pitch:
    """Canonical sounding pitch spelling for one note."""

    step: PitchStep
    octave: int
    accidental: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _coerce_str_enum(self.step, PitchStep, "step"))

        if self.accidental is not None:
            accidental = self.accidental.strip()
            if not accidental:
                raise ValueError("Pitch accidental must be non-empty when provided.")

            object.__setattr__(self, "accidental", accidental)

    def sort_key(self) -> tuple[int, str, str]:
        """Return a stable pitch-order key for canonical note sorting."""
        accidental = "" if self.accidental is None else self.accidental.casefold()
        return (self.octave, self.step.value, accidental)


@dataclass(frozen=True, slots=True)
class TempoChangeValue:
    """Structured payload for a tempo point control."""

    beats_per_minute: float

    def __post_init__(self) -> None:
        if self.beats_per_minute <= 0:
            raise ValueError("TempoChangeValue beats_per_minute must be positive.")


@dataclass(frozen=True, slots=True)
class DynamicChangeValue:
    """Structured payload for a dynamic point control."""

    marking: str

    def __post_init__(self) -> None:
        normalized = self.marking.strip()
        if not normalized:
            raise ValueError("DynamicChangeValue marking must be non-empty.")

        object.__setattr__(self, "marking", normalized)


@dataclass(frozen=True, slots=True)
class FermataValue:
    """Structured payload for a fermata point control."""

    fermata_type: str | None = None
    length_scale: float | None = None

    def __post_init__(self) -> None:
        if self.fermata_type is not None:
            normalized = self.fermata_type.strip()
            if not normalized:
                raise ValueError("FermataValue fermata_type must be non-empty.")

            object.__setattr__(self, "fermata_type", normalized)

        if self.length_scale is not None and self.length_scale <= 0:
            raise ValueError("FermataValue length_scale must be positive.")


@dataclass(frozen=True, slots=True)
class HairpinValue:
    """Structured payload for a hairpin span control."""

    direction: HairpinDirection
    niente: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            _coerce_str_enum(self.direction, HairpinDirection, "direction"),
        )


@dataclass(frozen=True, slots=True)
class OttavaValue:
    """Structured payload for an ottava span control."""

    octave_shift: int

    def __post_init__(self) -> None:
        if self.octave_shift == 0:
            raise ValueError("OttavaValue octave_shift must be non-zero.")


@dataclass(frozen=True, slots=True)
class GenericTechniqueFlags:
    """Generic note- or onset-local technique flags shared across instruments."""

    tie_origin: bool = False
    tie_destination: bool = False
    legato_origin: bool = False
    legato_destination: bool = False
    accent: int | None = None
    ornament: str | None = None
    vibrato: str | None = None
    let_ring: bool = False
    muted: bool = False
    palm_muted: bool = False
    trill: int | None = None

    def __post_init__(self) -> None:
        if self.accent is not None and self.accent < 0:
            raise ValueError("GenericTechniqueFlags accent must be non-negative.")

        if self.trill is not None and self.trill < 0:
            raise ValueError("GenericTechniqueFlags trill must be non-negative.")

        if self.ornament is not None:
            object.__setattr__(
                self, "ornament", _normalize_optional_text(self.ornament, "ornament")
            )

        if self.vibrato is not None:
            object.__setattr__(
                self, "vibrato", _normalize_optional_text(self.vibrato, "vibrato")
            )


@dataclass(frozen=True, slots=True)
class GeneralTechniquePayload:
    """Namespace container for non-family-specific structured techniques."""

    ornament: str | None = None

    def __post_init__(self) -> None:
        if self.ornament is not None:
            object.__setattr__(
                self, "ornament", _normalize_optional_text(self.ornament, "ornament")
            )


@dataclass(frozen=True, slots=True)
class StringFrettedTechniquePayload:
    """Namespace container for fretted-string-specific technique payloads."""

    slide_types: tuple[int, ...] = ()
    hopo_type: int | None = None
    tapped: bool = False
    left_hand_tapped: bool = False
    harmonic_type: int | None = None
    harmonic_kind: int | None = None
    harmonic_fret: float | None = None
    bend_enabled: bool = False
    whammy_enabled: bool = False

    def __post_init__(self) -> None:
        if any(slide_type < 0 for slide_type in self.slide_types):
            raise ValueError(
                "StringFrettedTechniquePayload slide_types must be non-negative."
            )

        _require_non_negative_optional_integer(self.hopo_type, "hopo_type")
        _require_non_negative_optional_integer(self.harmonic_type, "harmonic_type")
        _require_non_negative_optional_integer(self.harmonic_kind, "harmonic_kind")

        if self.harmonic_fret is not None and self.harmonic_fret < 0:
            raise ValueError(
                "StringFrettedTechniquePayload harmonic_fret must be non-negative."
            )


@dataclass(frozen=True, slots=True)
class TechniquePayload:
    """Structured note/onset technique container.

    The IR keeps generic flags separate from optional namespace payloads so the base
    document stays typed without committing early to every future instrument family.
    """

    generic: GenericTechniqueFlags = field(default_factory=GenericTechniqueFlags)
    general: GeneralTechniquePayload | None = None
    string_fretted: StringFrettedTechniquePayload | None = None


@dataclass(frozen=True, slots=True)
class Part:
    """Track-level IR structure without free-form textual metadata."""

    part_id: str
    instrument_family: int
    instrument_kind: int
    role: int
    transposition: Transposition
    staff_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.part_id, PART_PREFIX, "part_id")
        normalized_staff_ids = tuple(self.staff_ids)
        if not normalized_staff_ids:
            raise ValueError("Part staff_ids must contain at least one staff.")

        if len(set(normalized_staff_ids)) != len(normalized_staff_ids):
            raise ValueError("Part staff_ids must be unique within the part.")

        for staff_identifier in normalized_staff_ids:
            _require_identifier_prefix(staff_identifier, STAFF_PREFIX, "staff_ids")
            if not staff_identifier.startswith(f"{STAFF_PREFIX}:{self.part_id}:"):
                raise ValueError(
                    "Part staff_ids must belong to the owning part identifier."
                )

        object.__setattr__(self, "staff_ids", normalized_staff_ids)


@dataclass(frozen=True, slots=True)
class Staff:
    """Staff-level structure and performance context owned by one part."""

    staff_id: str
    part_id: str
    staff_index: int
    tuning_pitches: tuple[int, ...] | None = None
    capo_fret: int | None = None

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.staff_id, STAFF_PREFIX, "staff_id")
        _require_identifier_prefix(self.part_id, PART_PREFIX, "part_id")
        if not self.staff_id.startswith(f"{STAFF_PREFIX}:{self.part_id}:"):
            raise ValueError("Staff staff_id must encode the owning part_id.")

        if self.staff_index < 0:
            raise ValueError("Staff staff_index must be non-negative.")

        if self.tuning_pitches is not None:
            tuning_pitches = tuple(self.tuning_pitches)
            if not tuning_pitches:
                raise ValueError(
                    "Staff tuning_pitches must be non-empty when provided."
                )

            object.__setattr__(self, "tuning_pitches", tuning_pitches)

        if self.capo_fret is not None and self.capo_fret < 0:
            raise ValueError("Staff capo_fret must be non-negative when provided.")


@dataclass(frozen=True, slots=True)
class Bar:
    """Score-wide written bar geometry and local meter context."""

    bar_id: str
    bar_index: int
    start: ScoreTime
    duration: ScoreTime
    time_signature: TimeSignature
    key_accidental_count: int | None = None
    key_mode: str | None = None
    triplet_feel: str | None = None
    anacrusis_context: str | None = None

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.bar_id, BAR_PREFIX, "bar_id")
        if self.bar_index < 0:
            raise ValueError("Bar bar_index must be non-negative.")

        self.start.require_non_negative("Bar start")
        if self.duration.numerator <= 0:
            raise ValueError("Bar duration must be positive.")

        if self.key_mode is not None:
            object.__setattr__(
                self, "key_mode", _normalize_optional_text(self.key_mode, "key_mode")
            )

        if self.triplet_feel is not None:
            object.__setattr__(
                self,
                "triplet_feel",
                _normalize_optional_text(self.triplet_feel, "triplet_feel"),
            )

        if self.anacrusis_context is not None:
            object.__setattr__(
                self,
                "anacrusis_context",
                _normalize_optional_text(self.anacrusis_context, "anacrusis_context"),
            )


@dataclass(frozen=True, slots=True)
class VoiceLane:
    """Bar-scoped authored voice lane with a deterministic continuity chain."""

    voice_lane_id: str
    voice_lane_chain_id: str
    part_id: str
    staff_id: str
    bar_id: str
    voice_index: int

    def __post_init__(self) -> None:
        _require_identifier_prefix(
            self.voice_lane_id, VOICE_LANE_PREFIX, "voice_lane_id"
        )
        _require_identifier_prefix(
            self.voice_lane_chain_id,
            VOICE_LANE_CHAIN_PREFIX,
            "voice_lane_chain_id",
        )
        _require_identifier_prefix(self.part_id, PART_PREFIX, "part_id")
        _require_identifier_prefix(self.staff_id, STAFF_PREFIX, "staff_id")
        _require_identifier_prefix(self.bar_id, BAR_PREFIX, "bar_id")

        if not self.staff_id.startswith(f"{STAFF_PREFIX}:{self.part_id}:"):
            raise ValueError("VoiceLane staff_id must belong to the owning part_id.")

        if not self.voice_lane_id.startswith(f"{VOICE_LANE_PREFIX}:{self.staff_id}:"):
            raise ValueError("VoiceLane voice_lane_id must encode the owning staff_id.")

        if not self.voice_lane_chain_id.startswith(
            f"{VOICE_LANE_CHAIN_PREFIX}:{self.part_id}:{self.staff_id}:"
        ):
            raise ValueError(
                "VoiceLane voice_lane_chain_id must encode the owning part and staff."
            )

        if self.voice_index < 0:
            raise ValueError("VoiceLane voice_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class OnsetGroup:
    """One authored onset or rest slot inside a voice lane."""

    onset_id: str
    voice_lane_id: str
    bar_id: str
    time: ScoreTime
    duration_notated: ScoreTime
    is_rest: bool
    attack_order_in_voice: int
    duration_sounding_max: ScoreTime | None = None
    grace_type: str | None = None
    dynamic_local: str | None = None
    techniques: TechniquePayload | None = None
    rhythm_shape: RhythmShape | None = None

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.onset_id, ONSET_PREFIX, "onset_id")
        _require_identifier_prefix(
            self.voice_lane_id, VOICE_LANE_PREFIX, "voice_lane_id"
        )
        _require_identifier_prefix(self.bar_id, BAR_PREFIX, "bar_id")
        if not self.onset_id.startswith(f"{ONSET_PREFIX}:{self.voice_lane_id}:"):
            raise ValueError(
                "OnsetGroup onset_id must encode the owning voice_lane_id."
            )

        self.time.require_non_negative("OnsetGroup time")
        if self.duration_notated.numerator <= 0:
            raise ValueError("OnsetGroup duration_notated must be positive.")

        if self.attack_order_in_voice < 0:
            raise ValueError("OnsetGroup attack_order_in_voice must be non-negative.")

        if self.duration_sounding_max is not None:
            if self.duration_sounding_max.numerator <= 0:
                raise ValueError("OnsetGroup duration_sounding_max must be positive.")

            if self.is_rest:
                raise ValueError(
                    "OnsetGroup duration_sounding_max cannot be set for a rest onset."
                )

        if self.grace_type is not None:
            object.__setattr__(
                self,
                "grace_type",
                _normalize_optional_text(self.grace_type, "grace_type"),
            )

        if self.dynamic_local is not None:
            object.__setattr__(
                self,
                "dynamic_local",
                _normalize_optional_text(self.dynamic_local, "dynamic_local"),
            )


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """One note attached to an onset group."""

    note_id: str
    onset_id: str
    part_id: str
    staff_id: str
    time: ScoreTime
    attack_duration: ScoreTime
    sounding_duration: ScoreTime
    pitch: Pitch | None = None
    velocity: int | None = None
    string_number: int | None = None
    show_string_number: bool | None = None
    techniques: TechniquePayload | None = None

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.note_id, NOTE_PREFIX, "note_id")
        _require_identifier_prefix(self.onset_id, ONSET_PREFIX, "onset_id")
        _require_identifier_prefix(self.part_id, PART_PREFIX, "part_id")
        _require_identifier_prefix(self.staff_id, STAFF_PREFIX, "staff_id")
        if not self.note_id.startswith(f"{NOTE_PREFIX}:{self.onset_id}:"):
            raise ValueError("NoteEvent note_id must encode the owning onset_id.")

        if not self.staff_id.startswith(f"{STAFF_PREFIX}:{self.part_id}:"):
            raise ValueError("NoteEvent staff_id must belong to the owning part_id.")

        self.time.require_non_negative("NoteEvent time")
        if self.attack_duration.numerator <= 0:
            raise ValueError("NoteEvent attack_duration must be positive.")

        if self.sounding_duration.numerator <= 0:
            raise ValueError("NoteEvent sounding_duration must be positive.")

        if self.velocity is not None and self.velocity < 0:
            raise ValueError("NoteEvent velocity must be non-negative when provided.")

        if self.string_number is not None and self.string_number <= 0:
            raise ValueError("NoteEvent string_number must be positive when provided.")

        if self.show_string_number and self.string_number is None:
            raise ValueError(
                "NoteEvent show_string_number requires string_number to be present."
            )


@dataclass(frozen=True, slots=True)
class PointControlEvent:
    """One point-local score control event."""

    control_id: str
    kind: PointControlKind
    scope: ControlScope
    target_ref: str
    time: ScoreTime
    value: PointControlValue

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.control_id, POINT_CONTROL_PREFIX, "control_id")
        object.__setattr__(
            self,
            "kind",
            _coerce_str_enum(self.kind, PointControlKind, "kind"),
        )
        object.__setattr__(
            self,
            "scope",
            _coerce_str_enum(self.scope, ControlScope, "scope"),
        )
        self.time.require_non_negative("PointControlEvent time")
        _validate_control_target_ref(self.scope, self.target_ref)
        _validate_point_control_value(self.kind, self.value)


@dataclass(frozen=True, slots=True)
class SpanControlEvent:
    """One span-local score control event."""

    control_id: str
    kind: SpanControlKind
    scope: ControlScope
    target_ref: str
    start_time: ScoreTime
    end_time: ScoreTime
    value: SpanControlValue
    start_anchor_ref: str | None = None
    end_anchor_ref: str | None = None

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.control_id, SPAN_CONTROL_PREFIX, "control_id")
        object.__setattr__(
            self,
            "kind",
            _coerce_str_enum(self.kind, SpanControlKind, "kind"),
        )
        object.__setattr__(
            self,
            "scope",
            _coerce_str_enum(self.scope, ControlScope, "scope"),
        )
        self.start_time.require_non_negative("SpanControlEvent start_time")
        self.end_time.require_non_negative("SpanControlEvent end_time")
        if self.end_time <= self.start_time:
            raise ValueError("SpanControlEvent end_time must be after start_time.")

        _validate_control_target_ref(self.scope, self.target_ref)
        _validate_span_control_value(self.kind, self.value)

        if self.start_anchor_ref is not None:
            _validate_anchor_ref(self.start_anchor_ref, "start_anchor_ref")

        if self.end_anchor_ref is not None:
            _validate_anchor_ref(self.end_anchor_ref, "end_anchor_ref")


CANONICAL_CONTAINMENT_PATHS: tuple[tuple[str, str], ...] = (
    (PART_PREFIX, STAFF_PREFIX),
    (BAR_PREFIX, VOICE_LANE_PREFIX),
    (VOICE_LANE_PREFIX, ONSET_PREFIX),
    (ONSET_PREFIX, NOTE_PREFIX),
)


@dataclass(frozen=True, slots=True)
class Edge:
    """Sparse intrinsic edge between two IR entities."""

    source_id: str
    target_id: str
    edge_type: EdgeType

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "edge_type",
            _coerce_str_enum(self.edge_type, EdgeType, "edge_type"),
        )
        source_prefix = _identifier_prefix(self.source_id, "source_id")
        target_prefix = _identifier_prefix(self.target_id, "target_id")

        if self.edge_type is EdgeType.CONTAINS:
            if (source_prefix, target_prefix) not in CANONICAL_CONTAINMENT_PATHS:
                raise ValueError(
                    "Edge contains relationships must follow the canonical "
                    "Part->Staff, Bar->VoiceLane, VoiceLane->OnsetGroup, or "
                    "OnsetGroup->NoteEvent paths."
                )
            return

        if self.edge_type is EdgeType.NEXT_IN_VOICE:
            _require_edge_endpoint_prefix(
                self.source_id, ONSET_PREFIX, "source_id", self.edge_type
            )
            _require_edge_endpoint_prefix(
                self.target_id, ONSET_PREFIX, "target_id", self.edge_type
            )
            return

        _require_edge_endpoint_prefix(
            self.source_id, NOTE_PREFIX, "source_id", self.edge_type
        )
        _require_edge_endpoint_prefix(
            self.target_id, NOTE_PREFIX, "target_id", self.edge_type
        )


@dataclass(frozen=True, slots=True)
class IrDocumentMetadata:
    """Build and schema metadata for one IR document."""

    ir_schema_version: str
    corpus_build_version: str
    generator_version: str
    source_document_hash: str
    time_unit: TimeUnit = TimeUnit.WHOLE_NOTE_FRACTION
    compiled_resolution_hint: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_unit",
            _coerce_str_enum(self.time_unit, TimeUnit, "time_unit"),
        )
        object.__setattr__(
            self,
            "ir_schema_version",
            _normalize_optional_text(self.ir_schema_version, "ir_schema_version"),
        )
        object.__setattr__(
            self,
            "corpus_build_version",
            _normalize_optional_text(self.corpus_build_version, "corpus_build_version"),
        )
        object.__setattr__(
            self,
            "generator_version",
            _normalize_optional_text(self.generator_version, "generator_version"),
        )
        object.__setattr__(
            self,
            "source_document_hash",
            _normalize_optional_text(self.source_document_hash, "source_document_hash"),
        )

        if self.compiled_resolution_hint is not None:
            if self.compiled_resolution_hint <= 0:
                raise ValueError(
                    "IrDocumentMetadata compiled_resolution_hint must be positive "
                    "when provided."
                )


@dataclass(frozen=True, slots=True)
class IrManifestDiagnosticSummary:
    """Grouped build diagnostic information attached to one manifest entry."""

    category: IrManifestDiagnosticCategory
    severity: str
    code: str
    count: int
    paths: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category",
            _coerce_str_enum(
                self.category,
                IrManifestDiagnosticCategory,
                "category",
            ),
        )

        severity = _normalize_optional_text(self.severity, "severity").casefold()
        if severity not in {"error", "warning"}:
            raise ValueError("severity must be either 'error' or 'warning'.")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", _normalize_optional_text(self.code, "code"))

        if self.count <= 0:
            raise ValueError("count must be positive.")

        object.__setattr__(
            self,
            "paths",
            _normalize_text_sequence(self.paths, "paths"),
        )
        object.__setattr__(
            self,
            "messages",
            _normalize_text_sequence(self.messages, "messages"),
        )


@dataclass(frozen=True, slots=True)
class IrManifestEntry:
    """File-level build manifest entry for one emitted IR document."""

    source_path: str
    source_hash: str
    ir_document_path: str
    build_timestamp: str
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    unsupported_features_dropped: tuple[str, ...] = ()
    conversion_diagnostics: tuple[IrManifestDiagnosticSummary, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_path",
            _normalize_optional_text(self.source_path, "source_path"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _normalize_optional_text(self.source_hash, "source_hash"),
        )
        object.__setattr__(
            self,
            "ir_document_path",
            _normalize_optional_text(self.ir_document_path, "ir_document_path"),
        )
        object.__setattr__(
            self,
            "build_timestamp",
            _normalize_optional_text(self.build_timestamp, "build_timestamp"),
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
        object.__setattr__(
            self,
            "unsupported_features_dropped",
            _normalize_text_sequence(
                self.unsupported_features_dropped, "unsupported_features_dropped"
            ),
        )
        object.__setattr__(
            self,
            "conversion_diagnostics",
            tuple(
                sorted(
                    self.conversion_diagnostics,
                    key=lambda item: (
                        item.category.value,
                        item.severity,
                        item.code,
                        item.paths,
                        item.messages,
                    ),
                )
            ),
        )


PhraseConfidence: TypeAlias = str | float


@dataclass(frozen=True, slots=True)
class PhraseSpan:
    """Optional overlay describing one phrase span over a score scope."""

    phrase_id: str
    scope_ref: str
    start_time: ScoreTime
    end_time: ScoreTime
    phrase_kind: PhraseKind
    source: PhraseSource
    confidence: PhraseConfidence
    voice_lane_chain_id: str | None = None
    anchor_bar_start: str | None = None
    anchor_bar_end: str | None = None

    def __post_init__(self) -> None:
        _require_identifier_prefix(self.phrase_id, PHRASE_PREFIX, "phrase_id")
        _validate_phrase_scope_ref(self.scope_ref, "scope_ref")
        self.start_time.require_non_negative("PhraseSpan start_time")
        self.end_time.require_non_negative("PhraseSpan end_time")
        if self.end_time <= self.start_time:
            raise ValueError("PhraseSpan end_time must be after start_time.")

        object.__setattr__(
            self,
            "phrase_kind",
            _coerce_str_enum(self.phrase_kind, PhraseKind, "phrase_kind"),
        )
        object.__setattr__(
            self,
            "source",
            _coerce_str_enum(self.source, PhraseSource, "source"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_phrase_confidence(self.confidence),
        )

        if self.voice_lane_chain_id is not None:
            _require_identifier_prefix(
                self.voice_lane_chain_id,
                VOICE_LANE_CHAIN_PREFIX,
                "voice_lane_chain_id",
            )

        if self.anchor_bar_start is not None:
            _require_identifier_prefix(
                self.anchor_bar_start, BAR_PREFIX, "anchor_bar_start"
            )

        if self.anchor_bar_end is not None:
            _require_identifier_prefix(
                self.anchor_bar_end, BAR_PREFIX, "anchor_bar_end"
            )


@dataclass(frozen=True, slots=True)
class PlaybackInstance:
    """Placeholder playback-unrolled event span for future derived traversals."""

    instance_id: str
    source_ref: str
    start_time: ScoreTime
    end_time: ScoreTime
    voice_lane_chain_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instance_id",
            _normalize_optional_text(self.instance_id, "instance_id"),
        )
        object.__setattr__(
            self,
            "source_ref",
            _normalize_optional_text(self.source_ref, "source_ref"),
        )
        self.start_time.require_non_negative("PlaybackInstance start_time")
        self.end_time.require_non_negative("PlaybackInstance end_time")
        if self.end_time <= self.start_time:
            raise ValueError("PlaybackInstance end_time must be after start_time.")

        if self.voice_lane_chain_id is not None:
            _require_identifier_prefix(
                self.voice_lane_chain_id,
                VOICE_LANE_CHAIN_PREFIX,
                "voice_lane_chain_id",
            )

    def sort_key(self) -> tuple[ScoreTime, ScoreTime, str, str]:
        """Return a stable ordering key for canonical serialization."""
        return (self.start_time, self.end_time, self.instance_id, self.source_ref)


@dataclass(frozen=True, slots=True)
class DerivedEdge:
    """Optional derived relation between IR entities or overlays."""

    source_id: str
    target_id: str
    edge_type: DerivedEdgeType

    def __post_init__(self) -> None:
        _identifier_prefix(self.source_id, "source_id")
        _identifier_prefix(self.target_id, "target_id")
        object.__setattr__(
            self,
            "edge_type",
            _coerce_str_enum(self.edge_type, DerivedEdgeType, "edge_type"),
        )

    def sort_key(
        self,
    ) -> tuple[
        tuple[str, tuple[tuple[int, int | str], ...]],
        str,
        tuple[str, tuple[tuple[int, int | str], ...]],
    ]:
        """Return a stable ordering key for canonical serialization."""
        return edge_sort_key(self.source_id, self.edge_type.value, self.target_id)


@dataclass(frozen=True, slots=True)
class DerivedEdgeSet:
    """Named container for one optional derived-edge family or analytical slice."""

    name: str
    kind: str
    edges: tuple[DerivedEdge, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_optional_text(self.name, "name"))
        object.__setattr__(self, "kind", _normalize_optional_text(self.kind, "kind"))
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(tuple(self.edges), key=lambda edge: edge.sort_key())),
        )

    def sort_key(self) -> tuple[str, str]:
        """Return a stable ordering key for canonical serialization."""
        return (self.kind.casefold(), self.name.casefold())


@dataclass(frozen=True, slots=True)
class OptionalOverlays:
    """Optional overlay containers kept separate from the canonical backbone."""

    phrase_spans: tuple[PhraseSpan, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phrase_spans", tuple(self.phrase_spans))


@dataclass(frozen=True, slots=True)
class OptionalViews:
    """Optional derived-view containers kept separate from canonical entities."""

    playback_instances: tuple[PlaybackInstance, ...] = ()
    derived_edge_sets: tuple[DerivedEdgeSet, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "playback_instances",
            tuple(
                sorted(tuple(self.playback_instances), key=lambda item: item.sort_key())
            ),
        )
        object.__setattr__(
            self,
            "derived_edge_sets",
            tuple(
                sorted(tuple(self.derived_edge_sets), key=lambda item: item.sort_key())
            ),
        )


@dataclass(frozen=True, slots=True)
class MotifMlIrDocument:
    """Top-level canonical IR document for one source score."""

    metadata: IrDocumentMetadata
    parts: tuple[Part, ...] = ()
    staves: tuple[Staff, ...] = ()
    bars: tuple[Bar, ...] = ()
    voice_lanes: tuple[VoiceLane, ...] = ()
    point_control_events: tuple[PointControlEvent, ...] = ()
    span_control_events: tuple[SpanControlEvent, ...] = ()
    onset_groups: tuple[OnsetGroup, ...] = ()
    note_events: tuple[NoteEvent, ...] = ()
    edges: tuple[Edge, ...] = ()
    optional_overlays: OptionalOverlays = field(default_factory=OptionalOverlays)
    optional_views: OptionalViews = field(default_factory=OptionalViews)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "staves", tuple(self.staves))
        object.__setattr__(self, "bars", tuple(self.bars))
        object.__setattr__(self, "voice_lanes", tuple(self.voice_lanes))
        object.__setattr__(
            self, "point_control_events", tuple(self.point_control_events)
        )
        object.__setattr__(self, "span_control_events", tuple(self.span_control_events))
        object.__setattr__(self, "onset_groups", tuple(self.onset_groups))
        object.__setattr__(self, "note_events", tuple(self.note_events))
        object.__setattr__(self, "edges", tuple(self.edges))


# Control values are intentionally modeled as strongly typed payload dataclasses
# grouped by union aliases, rather than as free-form tagged dictionaries.
PointControlValue: TypeAlias = TempoChangeValue | DynamicChangeValue | FermataValue
SpanControlValue: TypeAlias = HairpinValue | OttavaValue
ControlValue: TypeAlias = PointControlValue | SpanControlValue

__all__ = [
    "Bar",
    "ControlScope",
    "ControlValue",
    "CANONICAL_CONTAINMENT_PATHS",
    "DerivedEdge",
    "DerivedEdgeSet",
    "DerivedEdgeType",
    "DynamicChangeValue",
    "Edge",
    "EdgeType",
    "FermataValue",
    "GeneralTechniquePayload",
    "GenericTechniqueFlags",
    "HairpinDirection",
    "HairpinValue",
    "IrDocumentMetadata",
    "IrManifestDiagnosticCategory",
    "IrManifestDiagnosticSummary",
    "IrManifestEntry",
    "MotifMlIrDocument",
    "NoteEvent",
    "OnsetGroup",
    "OttavaValue",
    "OptionalOverlays",
    "OptionalViews",
    "Part",
    "PlaybackInstance",
    "Pitch",
    "PitchStep",
    "PhraseConfidence",
    "PhraseKind",
    "PhraseSource",
    "PhraseSpan",
    "PointControlEvent",
    "PointControlKind",
    "PointControlValue",
    "RhythmBaseValue",
    "RhythmShape",
    "Staff",
    "SpanControlEvent",
    "SpanControlKind",
    "SpanControlValue",
    "StringFrettedTechniquePayload",
    "TechniquePayload",
    "TempoChangeValue",
    "TimeSignature",
    "TimeUnit",
    "Transposition",
    "TupletRatio",
    "VoiceLane",
]


def _coerce_str_enum(
    value: _StrEnumT | str, enum_type: type[_StrEnumT], field_name: str
) -> _StrEnumT:
    if isinstance(value, enum_type):
        return value

    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(member.value for member in enum_type)}."
        ) from exc


def _normalize_optional_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when provided.")

    return normalized


def _normalize_count_mapping(value: dict[str, int], field_name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping of names to counts.")

    normalized: dict[str, int] = {}
    for raw_key, raw_count in sorted(
        value.items(), key=lambda item: str(item[0]).casefold()
    ):
        key = _normalize_optional_text(str(raw_key), f"{field_name} key")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"{field_name} values must be non-negative.")

        normalized[key] = count

    return normalized


def _normalize_text_sequence(
    value: tuple[str, ...], field_name: str
) -> tuple[str, ...]:
    normalized = tuple(
        _normalize_optional_text(str(item), f"{field_name} entry") for item in value
    )
    return tuple(sorted(normalized, key=str.casefold))


def _require_non_negative_optional_integer(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")


def _require_identifier_prefix(value: str, prefix: str, field_name: str) -> None:
    if not value.startswith(f"{prefix}:"):
        raise ValueError(f"{field_name} must start with '{prefix}:'.")


def _identifier_prefix(value: str, field_name: str) -> str:
    prefix, separator, _ = value.partition(":")
    if not separator:
        raise ValueError(f"{field_name} must contain a namespaced identifier prefix.")

    return prefix


def _require_edge_endpoint_prefix(
    value: str, prefix: str, field_name: str, edge_type: EdgeType
) -> None:
    if not value.startswith(f"{prefix}:"):
        raise ValueError(
            f"Edge type '{edge_type.value}' requires {field_name} to start with '{prefix}:'."
        )


def _validate_control_target_ref(scope: ControlScope, target_ref: str) -> None:
    if scope is ControlScope.SCORE:
        if target_ref != ControlScope.SCORE.value:
            raise ValueError("Score-scoped controls must target the literal 'score'.")
        return

    scope_prefix_map = {
        ControlScope.PART: PART_PREFIX,
        ControlScope.STAFF: STAFF_PREFIX,
        ControlScope.VOICE: VOICE_LANE_PREFIX,
    }
    prefix = scope_prefix_map[scope]
    if not target_ref.startswith(f"{prefix}:"):
        raise ValueError(
            f"{scope.value}-scoped controls must target a '{prefix}:' reference."
        )


def _validate_point_control_value(
    kind: PointControlKind, value: PointControlValue
) -> None:
    value_type_by_kind = {
        PointControlKind.TEMPO_CHANGE: TempoChangeValue,
        PointControlKind.DYNAMIC_CHANGE: DynamicChangeValue,
        PointControlKind.FERMATA: FermataValue,
    }
    expected_type = value_type_by_kind[kind]
    if not isinstance(value, expected_type):
        raise ValueError(
            f"PointControlEvent kind '{kind.value}' requires {expected_type.__name__}."
        )


def _validate_span_control_value(
    kind: SpanControlKind, value: SpanControlValue
) -> None:
    value_type_by_kind = {
        SpanControlKind.HAIRPIN: HairpinValue,
        SpanControlKind.OTTAVA: OttavaValue,
    }
    expected_type = value_type_by_kind[kind]
    if not isinstance(value, expected_type):
        raise ValueError(
            f"SpanControlEvent kind '{kind.value}' requires {expected_type.__name__}."
        )


def _validate_anchor_ref(value: str, field_name: str) -> None:
    if value.startswith(f"{ONSET_PREFIX}:") or value.startswith(f"{NOTE_PREFIX}:"):
        return

    raise ValueError(f"{field_name} must reference an onset or note identifier.")


def _validate_phrase_scope_ref(value: str, field_name: str) -> None:
    if value.startswith(f"{PART_PREFIX}:"):
        return
    if value.startswith(f"{STAFF_PREFIX}:"):
        return
    if value.startswith(f"{VOICE_LANE_CHAIN_PREFIX}:"):
        return

    raise ValueError(
        f"{field_name} must reference a part, staff, or voice-lane chain identifier."
    )


def _normalize_phrase_confidence(value: PhraseConfidence) -> PhraseConfidence:
    if isinstance(value, str):
        return _normalize_optional_text(value, "confidence")

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("confidence must be a non-empty string or numeric score.")

    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("confidence must be finite when provided as a numeric score.")

    return normalized
