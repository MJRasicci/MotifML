"""Typed value models shared across the MotifML IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, TypeVar

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


@dataclass(frozen=True)
class Transposition:
    """Written-to-sounding transposition context for one part."""

    chromatic: int = 0
    octave: int = 0

    @property
    def written_minus_sounding_semitones(self) -> int:
        """Return the derived semitone offset preserved by the source model."""
        return self.chromatic + (self.octave * 12)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TupletRatio:
    """Tuplet ratio metadata used by written rhythm shapes."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError("TupletRatio numerator must be greater than zero.")

        if self.denominator <= 0:
            raise ValueError("TupletRatio denominator must be greater than zero.")


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TempoChangeValue:
    """Structured payload for a tempo point control."""

    beats_per_minute: float

    def __post_init__(self) -> None:
        if self.beats_per_minute <= 0:
            raise ValueError("TempoChangeValue beats_per_minute must be positive.")


@dataclass(frozen=True)
class DynamicChangeValue:
    """Structured payload for a dynamic point control."""

    marking: str

    def __post_init__(self) -> None:
        normalized = self.marking.strip()
        if not normalized:
            raise ValueError("DynamicChangeValue marking must be non-empty.")

        object.__setattr__(self, "marking", normalized)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class OttavaValue:
    """Structured payload for an ottava span control."""

    octave_shift: int

    def __post_init__(self) -> None:
        if self.octave_shift == 0:
            raise ValueError("OttavaValue octave_shift must be non-zero.")


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class GeneralTechniquePayload:
    """Namespace container for non-family-specific structured techniques."""

    ornament: str | None = None

    def __post_init__(self) -> None:
        if self.ornament is not None:
            object.__setattr__(
                self, "ornament", _normalize_optional_text(self.ornament, "ornament")
            )


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TechniquePayload:
    """Structured note/onset technique container.

    The IR keeps generic flags separate from optional namespace payloads so the base
    document stays typed without committing early to every future instrument family.
    """

    generic: GenericTechniqueFlags = field(default_factory=GenericTechniqueFlags)
    general: GeneralTechniquePayload | None = None
    string_fretted: StringFrettedTechniquePayload | None = None


# Control values are intentionally modeled as strongly typed payload dataclasses
# grouped by union aliases, rather than as free-form tagged dictionaries.
PointControlValue: TypeAlias = TempoChangeValue | DynamicChangeValue | FermataValue
SpanControlValue: TypeAlias = HairpinValue | OttavaValue
ControlValue: TypeAlias = PointControlValue | SpanControlValue

__all__ = [
    "ControlValue",
    "DynamicChangeValue",
    "FermataValue",
    "GeneralTechniquePayload",
    "GenericTechniqueFlags",
    "HairpinDirection",
    "HairpinValue",
    "OttavaValue",
    "Pitch",
    "PitchStep",
    "PointControlValue",
    "RhythmBaseValue",
    "RhythmShape",
    "SpanControlValue",
    "StringFrettedTechniquePayload",
    "TechniquePayload",
    "TempoChangeValue",
    "TimeSignature",
    "Transposition",
    "TupletRatio",
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


def _require_non_negative_optional_integer(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative when provided.")
