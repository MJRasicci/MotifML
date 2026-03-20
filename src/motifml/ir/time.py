"""Exact rational time primitives used by MotifML IR entities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from math import gcd


@total_ordering
@dataclass(frozen=True, slots=True)
class ScoreTime:
    """Exact score time stored as a reduced whole-note fraction.

    Args:
        numerator: Signed numerator for the rational value.
        denominator: Positive denominator for the rational value.

    Raises:
        ValueError: If ``denominator`` is not strictly positive.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ValueError("ScoreTime denominator must be greater than zero.")

        if self.numerator == 0:
            object.__setattr__(self, "numerator", 0)
            object.__setattr__(self, "denominator", 1)
            return

        common_divisor = gcd(abs(self.numerator), self.denominator)
        object.__setattr__(self, "numerator", self.numerator // common_divisor)
        object.__setattr__(self, "denominator", self.denominator // common_divisor)

    @classmethod
    def from_fraction(cls, numerator: int, denominator: int) -> ScoreTime:
        """Build a ``ScoreTime`` from an explicit numerator/denominator pair."""
        return cls(numerator=numerator, denominator=denominator)

    @classmethod
    def from_numerator_denominator(cls, numerator: int, denominator: int) -> ScoreTime:
        """Build a ``ScoreTime`` from integer numerator and denominator values."""
        return cls.from_fraction(numerator=numerator, denominator=denominator)

    def to_float(self) -> float:
        """Convert the rational value to ``float`` for display or debugging."""
        return self.numerator / self.denominator

    def require_non_negative(self, field_name: str = "score time") -> ScoreTime:
        """Reject negative values when a caller is validating a duration field.

        Args:
            field_name: Human-readable field name to include in the exception.

        Returns:
            The same ``ScoreTime`` instance for convenient inline validation.

        Raises:
            ValueError: If the rational value is negative.
        """
        if self.numerator < 0:
            raise ValueError(f"{field_name} must be non-negative.")

        return self

    def __add__(self, other: object) -> ScoreTime:
        if not isinstance(other, ScoreTime):
            return NotImplemented

        return ScoreTime(
            numerator=(
                self.numerator * other.denominator + other.numerator * self.denominator
            ),
            denominator=self.denominator * other.denominator,
        )

    def __sub__(self, other: object) -> ScoreTime:
        if not isinstance(other, ScoreTime):
            return NotImplemented

        return ScoreTime(
            numerator=(
                self.numerator * other.denominator - other.numerator * self.denominator
            ),
            denominator=self.denominator * other.denominator,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ScoreTime):
            return NotImplemented

        return self.numerator * other.denominator < other.numerator * self.denominator
