"""Typed models used by the IR build pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DiagnosticSeverity(StrEnum):
    """Supported severities for IR build diagnostics."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class IrBuildDiagnostic:
    """One deterministic diagnostic emitted during IR build validation."""

    severity: DiagnosticSeverity
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "severity",
            DiagnosticSeverity(self.severity),
        )
        object.__setattr__(self, "code", _normalize_text(self.code, "code"))
        object.__setattr__(self, "path", _normalize_text(self.path, "path"))
        object.__setattr__(self, "message", _normalize_text(self.message, "message"))

    def sort_key(self) -> tuple[int, str, str, str]:
        """Return a stable diagnostic sort key."""
        severity_rank = 0 if self.severity is DiagnosticSeverity.ERROR else 1
        return (severity_rank, self.path, self.code, self.message)


@dataclass(frozen=True)
class CanonicalScoreValidationResult:
    """Typed validation result for one raw Motif JSON score surface."""

    relative_path: str
    source_hash: str
    diagnostics: tuple[IrBuildDiagnostic, ...] = ()
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

        diagnostics = tuple(sorted(self.diagnostics, key=lambda item: item.sort_key()))
        object.__setattr__(self, "diagnostics", diagnostics)

        error_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.ERROR
        )
        warning_count = sum(
            1
            for diagnostic in diagnostics
            if diagnostic.severity is DiagnosticSeverity.WARNING
        )
        object.__setattr__(self, "error_count", error_count)
        object.__setattr__(self, "warning_count", warning_count)
        object.__setattr__(self, "passed", error_count == 0)


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


__all__ = [
    "CanonicalScoreValidationResult",
    "DiagnosticSeverity",
    "IrBuildDiagnostic",
]
