"""Typed configuration and outputs for IR tokenization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from motifml.pipelines.feature_extraction.models import ProjectionType
from motifml.training.token_codec import coerce_frozen_vocabulary
from motifml.training.token_families import SPECIAL_TOKENS, TokenFamily


class PaddingStrategy(StrEnum):
    """Supported padding strategies for model-input preparation."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class TokenizationParameters:
    """Configuration surface for the tokenization pipeline."""

    vocabulary_strategy: str = "projection_native"
    max_sequence_length: int = 8
    padding_strategy: PaddingStrategy = PaddingStrategy.RIGHT
    time_resolution: int = 96

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary_strategy",
            _normalize_text(self.vocabulary_strategy, "vocabulary_strategy"),
        )
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        object.__setattr__(
            self,
            "padding_strategy",
            PaddingStrategy(self.padding_strategy),
        )
        if self.time_resolution <= 0:
            raise ValueError("time_resolution must be positive.")


@dataclass(frozen=True)
class VocabularyParameters:
    """Configuration surface for deterministic vocabulary construction."""

    time_resolution: int = 96
    minimum_frequency: int = 1
    maximum_size: int = 65536
    special_tokens: dict[str, str] | None = None
    guardrails: VocabularyGuardrailParameters | Mapping[str, Any] = field(
        default_factory=lambda: VocabularyGuardrailParameters()
    )

    def __post_init__(self) -> None:
        if self.time_resolution <= 0:
            raise ValueError("time_resolution must be positive.")
        if self.minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be positive.")
        if self.maximum_size <= 0:
            raise ValueError("maximum_size must be positive.")
        normalized_special_tokens = dict(SPECIAL_TOKENS)
        if self.special_tokens is not None:
            for key, value in self.special_tokens.items():
                normalized_special_tokens[
                    _normalize_text(str(key), "special_tokens")
                ] = _normalize_text(str(value), "special_tokens")
        object.__setattr__(self, "special_tokens", normalized_special_tokens)
        object.__setattr__(
            self,
            "guardrails",
            coerce_vocabulary_guardrail_parameters(self.guardrails),
        )


@dataclass(frozen=True)
class VocabularyGuardrailParameters:
    """Acceptance thresholds for frozen-vocabulary health checks."""

    minimum_vocabulary_size: int = 7
    required_token_families: tuple[str, ...] = (
        TokenFamily.NOTE_DURATION.value,
        TokenFamily.NOTE_PITCH.value,
        TokenFamily.STRUCTURE.value,
    )
    maximum_top_token_fraction: float = 0.6
    maximum_unk_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.minimum_vocabulary_size <= 0:
            raise ValueError("minimum_vocabulary_size must be positive.")
        normalized_families = tuple(
            sorted(
                {
                    _normalize_text(str(family), "required_token_families").upper()
                    for family in self.required_token_families
                }
            )
        )
        object.__setattr__(self, "required_token_families", normalized_families)
        object.__setattr__(
            self,
            "maximum_top_token_fraction",
            _normalize_fraction(
                self.maximum_top_token_fraction,
                "maximum_top_token_fraction",
            ),
        )
        object.__setattr__(
            self,
            "maximum_unk_fraction",
            _normalize_fraction(
                self.maximum_unk_fraction,
                "maximum_unk_fraction",
            ),
        )


@dataclass(frozen=True)
class ModelInputRecord:
    """One tokenized example derived from a projected feature record."""

    relative_path: str
    projection_type: ProjectionType
    vocabulary_strategy: str
    time_resolution: int
    original_token_count: int
    tokens: tuple[str, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _normalize_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "projection_type",
            ProjectionType(self.projection_type),
        )
        object.__setattr__(
            self,
            "vocabulary_strategy",
            _normalize_text(self.vocabulary_strategy, "vocabulary_strategy"),
        )
        if self.time_resolution <= 0:
            raise ValueError("time_resolution must be positive.")
        if self.original_token_count < 0:
            raise ValueError("original_token_count must be non-negative.")
        if len(self.tokens) != len(self.attention_mask):
            raise ValueError("tokens and attention_mask must have matching lengths.")


@dataclass(frozen=True)
class ModelInputSet:
    """Collection of model-input records emitted by tokenization."""

    parameters: TokenizationParameters
    records: tuple[ModelInputRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda item: item.relative_path.casefold())),
        )


@dataclass(frozen=True)
class TokenCountEntry:
    """One canonical token-frequency pair for reducer-friendly shard output."""

    token: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _normalize_text(self.token, "token"))
        if self.count < 0:
            raise ValueError("count must be non-negative.")


@dataclass(frozen=True)
class ShardTokenCounts:
    """Shard-local token frequencies derived from training-split sequence features."""

    feature_version: str
    split_version: str
    time_resolution: int
    special_token_policy: dict[str, Any]
    counted_document_count: int
    total_token_count: int
    counted_relative_paths: tuple[str, ...] = ()
    token_counts: tuple[TokenCountEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_version",
            _normalize_text(self.feature_version, "feature_version"),
        )
        object.__setattr__(
            self,
            "split_version",
            _normalize_text(self.split_version, "split_version"),
        )
        if self.time_resolution <= 0:
            raise ValueError("time_resolution must be positive.")
        object.__setattr__(
            self,
            "special_token_policy",
            {
                _normalize_text(str(key), "special_token_policy"): value
                for key, value in sorted(
                    self.special_token_policy.items(),
                    key=lambda item: str(item[0]),
                )
            },
        )
        if self.counted_document_count < 0:
            raise ValueError("counted_document_count must be non-negative.")
        if self.total_token_count < 0:
            raise ValueError("total_token_count must be non-negative.")
        normalized_paths = tuple(
            sorted(
                (
                    _normalize_text(relative_path, "counted_relative_paths")
                    for relative_path in self.counted_relative_paths
                ),
                key=str.casefold,
            )
        )
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("counted_relative_paths must be unique.")
        object.__setattr__(self, "counted_relative_paths", normalized_paths)
        normalized_counts = tuple(
            sorted(
                (
                    entry
                    if isinstance(entry, TokenCountEntry)
                    else TokenCountEntry(
                        token=str(entry["token"]),
                        count=int(entry["count"]),
                    )
                    for entry in self.token_counts
                ),
                key=lambda item: item.token,
            )
        )
        if self.counted_document_count != len(normalized_paths):
            raise ValueError(
                "counted_document_count must match counted_relative_paths length."
            )
        if self.total_token_count != sum(entry.count for entry in normalized_counts):
            raise ValueError("total_token_count must match the summed token counts.")
        object.__setattr__(self, "token_counts", normalized_counts)


@dataclass(frozen=True)
class TokenFamilyCoverageEntry:
    """Coverage summary for one retained token family in the frozen vocabulary."""

    family: str
    vocabulary_size: int
    token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _normalize_text(self.family, "family"))
        if self.vocabulary_size < 0:
            raise ValueError("vocabulary_size must be non-negative.")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative.")


@dataclass(frozen=True)
class VocabularyGuardrailReport:
    """Human-reviewable summary of frozen-vocabulary acceptance checks."""

    observed_vocabulary_size: int
    minimum_vocabulary_size: int
    required_token_families: tuple[str, ...]
    missing_required_token_families: tuple[str, ...]
    top_token: str | None
    top_token_count: int
    top_token_fraction: float
    maximum_top_token_fraction: float
    estimated_unk_token_count: int
    estimated_unk_fraction: float
    maximum_unk_fraction: float
    passed: bool

    def __post_init__(self) -> None:
        if self.observed_vocabulary_size < 0:
            raise ValueError("observed_vocabulary_size must be non-negative.")
        if self.minimum_vocabulary_size <= 0:
            raise ValueError("minimum_vocabulary_size must be positive.")
        object.__setattr__(
            self,
            "required_token_families",
            tuple(
                sorted(
                    _normalize_text(str(family), "required_token_families").upper()
                    for family in self.required_token_families
                )
            ),
        )
        object.__setattr__(
            self,
            "missing_required_token_families",
            tuple(
                sorted(
                    _normalize_text(
                        str(family), "missing_required_token_families"
                    ).upper()
                    for family in self.missing_required_token_families
                )
            ),
        )
        normalized_top_token = None
        if self.top_token is not None:
            normalized_top_token = _normalize_text(self.top_token, "top_token")
        object.__setattr__(self, "top_token", normalized_top_token)
        if self.top_token_count < 0:
            raise ValueError("top_token_count must be non-negative.")
        if normalized_top_token is None and self.top_token_count != 0:
            raise ValueError("top_token_count must be zero when top_token is absent.")
        object.__setattr__(
            self,
            "top_token_fraction",
            _normalize_fraction(self.top_token_fraction, "top_token_fraction"),
        )
        object.__setattr__(
            self,
            "maximum_top_token_fraction",
            _normalize_fraction(
                self.maximum_top_token_fraction,
                "maximum_top_token_fraction",
            ),
        )
        if self.estimated_unk_token_count < 0:
            raise ValueError("estimated_unk_token_count must be non-negative.")
        object.__setattr__(
            self,
            "estimated_unk_fraction",
            _normalize_fraction(self.estimated_unk_fraction, "estimated_unk_fraction"),
        )
        object.__setattr__(
            self,
            "maximum_unk_fraction",
            _normalize_fraction(
                self.maximum_unk_fraction,
                "maximum_unk_fraction",
            ),
        )
        object.__setattr__(self, "passed", bool(self.passed))


@dataclass(frozen=True)
class VocabularyArtifact:
    """Frozen vocabulary surface persisted for deterministic tokenization."""

    vocabulary_version: str
    feature_version: str
    split_version: str
    token_count: int
    vocabulary_size: int
    token_to_id: dict[str, int]
    token_counts: tuple[TokenCountEntry, ...]
    construction_parameters: dict[str, Any]
    special_token_policy: dict[str, Any]
    id_to_token: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary_version",
            _normalize_text(self.vocabulary_version, "vocabulary_version"),
        )
        object.__setattr__(
            self,
            "feature_version",
            _normalize_text(self.feature_version, "feature_version"),
        )
        object.__setattr__(
            self,
            "split_version",
            _normalize_text(self.split_version, "split_version"),
        )
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative.")
        if self.vocabulary_size < 0:
            raise ValueError("vocabulary_size must be non-negative.")
        normalized_counts = tuple(
            sorted(
                (
                    entry
                    if isinstance(entry, TokenCountEntry)
                    else TokenCountEntry(
                        token=str(entry["token"]),
                        count=int(entry["count"]),
                    )
                    for entry in self.token_counts
                ),
                key=lambda item: item.token,
            )
        )
        if self.token_count != sum(entry.count for entry in normalized_counts):
            raise ValueError("token_count must match the summed retained token counts.")
        if len(normalized_counts) != self.vocabulary_size:
            raise ValueError("vocabulary_size must match the retained token count.")
        frozen_vocabulary = coerce_frozen_vocabulary({"token_to_id": self.token_to_id})
        if len(frozen_vocabulary.token_to_id) != self.vocabulary_size:
            raise ValueError("token_to_id must match vocabulary_size.")
        count_tokens = {entry.token for entry in normalized_counts}
        vocabulary_tokens = set(frozen_vocabulary.token_to_id)
        if count_tokens != vocabulary_tokens:
            raise ValueError(
                "token_counts and token_to_id must cover the same retained token set."
            )
        object.__setattr__(
            self,
            "token_to_id",
            dict(frozen_vocabulary.token_to_id),
        )
        object.__setattr__(self, "id_to_token", frozen_vocabulary.id_to_token)
        object.__setattr__(self, "token_counts", normalized_counts)
        object.__setattr__(
            self,
            "construction_parameters",
            {
                _normalize_text(str(key), "construction_parameters"): value
                for key, value in sorted(
                    self.construction_parameters.items(),
                    key=lambda item: str(item[0]),
                )
            },
        )
        object.__setattr__(
            self,
            "special_token_policy",
            {
                _normalize_text(str(key), "special_token_policy"): value
                for key, value in sorted(
                    self.special_token_policy.items(),
                    key=lambda item: str(item[0]),
                )
            },
        )


@dataclass(frozen=True)
class VocabularyStatsReport:
    """Human-reviewable statistics for one frozen vocabulary reduction."""

    vocabulary_version: str
    feature_version: str
    split_version: str
    token_count: int
    vocabulary_size: int
    token_family_coverage: tuple[TokenFamilyCoverageEntry, ...]
    top_tokens: tuple[TokenCountEntry, ...]
    unk_token_count: int
    unk_token_fraction: float
    guardrails: VocabularyGuardrailReport | Mapping[str, Any]
    construction_parameters: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vocabulary_version",
            _normalize_text(self.vocabulary_version, "vocabulary_version"),
        )
        object.__setattr__(
            self,
            "feature_version",
            _normalize_text(self.feature_version, "feature_version"),
        )
        object.__setattr__(
            self,
            "split_version",
            _normalize_text(self.split_version, "split_version"),
        )
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative.")
        if self.vocabulary_size < 0:
            raise ValueError("vocabulary_size must be non-negative.")
        object.__setattr__(
            self,
            "token_family_coverage",
            tuple(
                sorted(
                    (
                        entry
                        if isinstance(entry, TokenFamilyCoverageEntry)
                        else TokenFamilyCoverageEntry(
                            family=str(entry["family"]),
                            vocabulary_size=int(entry["vocabulary_size"]),
                            token_count=int(entry["token_count"]),
                        )
                        for entry in self.token_family_coverage
                    ),
                    key=lambda item: item.family,
                )
            ),
        )
        object.__setattr__(
            self,
            "top_tokens",
            tuple(
                entry
                if isinstance(entry, TokenCountEntry)
                else TokenCountEntry(
                    token=str(entry["token"]), count=int(entry["count"])
                )
                for entry in self.top_tokens
            ),
        )
        if self.unk_token_count < 0:
            raise ValueError("unk_token_count must be non-negative.")
        if self.unk_token_count > self.token_count:
            raise ValueError("unk_token_count must not exceed token_count.")
        object.__setattr__(
            self,
            "unk_token_fraction",
            _normalize_fraction(self.unk_token_fraction, "unk_token_fraction"),
        )
        object.__setattr__(
            self,
            "guardrails",
            (
                self.guardrails
                if isinstance(self.guardrails, VocabularyGuardrailReport)
                else VocabularyGuardrailReport(
                    observed_vocabulary_size=int(
                        self.guardrails["observed_vocabulary_size"]
                    ),
                    minimum_vocabulary_size=int(
                        self.guardrails["minimum_vocabulary_size"]
                    ),
                    required_token_families=tuple(
                        self.guardrails.get("required_token_families", ())
                    ),
                    missing_required_token_families=tuple(
                        self.guardrails.get("missing_required_token_families", ())
                    ),
                    top_token=None
                    if self.guardrails.get("top_token") is None
                    else str(self.guardrails["top_token"]),
                    top_token_count=int(self.guardrails["top_token_count"]),
                    top_token_fraction=float(self.guardrails["top_token_fraction"]),
                    maximum_top_token_fraction=float(
                        self.guardrails["maximum_top_token_fraction"]
                    ),
                    estimated_unk_token_count=int(
                        self.guardrails["estimated_unk_token_count"]
                    ),
                    estimated_unk_fraction=float(
                        self.guardrails["estimated_unk_fraction"]
                    ),
                    maximum_unk_fraction=float(self.guardrails["maximum_unk_fraction"]),
                    passed=bool(self.guardrails["passed"]),
                )
            ),
        )
        object.__setattr__(
            self,
            "construction_parameters",
            {
                _normalize_text(str(key), "construction_parameters"): value
                for key, value in sorted(
                    self.construction_parameters.items(),
                    key=lambda item: str(item[0]),
                )
            },
        )


def coerce_tokenization_parameters(
    value: TokenizationParameters | Mapping[str, Any],
) -> TokenizationParameters:
    """Coerce Kedro parameter mappings into the typed tokenization config."""
    if isinstance(value, TokenizationParameters):
        return value

    return TokenizationParameters(
        vocabulary_strategy=str(value.get("vocabulary_strategy", "projection_native")),
        max_sequence_length=int(value.get("max_sequence_length", 8)),
        padding_strategy=value.get("padding_strategy", PaddingStrategy.RIGHT),
        time_resolution=int(value.get("time_resolution", 96)),
    )


def coerce_vocabulary_parameters(
    value: VocabularyParameters | Mapping[str, Any],
) -> VocabularyParameters:
    """Coerce Kedro vocabulary parameter mappings into the typed config."""
    if isinstance(value, VocabularyParameters):
        return value

    return VocabularyParameters(
        time_resolution=int(value.get("time_resolution", 96)),
        minimum_frequency=int(value.get("minimum_frequency", 1)),
        maximum_size=int(value.get("maximum_size", 65536)),
        special_tokens=(
            None
            if value.get("special_tokens") is None
            else {
                str(key): str(token)
                for key, token in value.get("special_tokens", {}).items()
            }
        ),
        guardrails=value.get("guardrails", {}),
    )


def coerce_vocabulary_guardrail_parameters(
    value: VocabularyGuardrailParameters | Mapping[str, Any] | None,
) -> VocabularyGuardrailParameters:
    """Coerce Kedro vocabulary guardrail mappings into the typed config."""
    if value is None:
        return VocabularyGuardrailParameters()
    if isinstance(value, VocabularyGuardrailParameters):
        return value

    defaults = VocabularyGuardrailParameters()
    return VocabularyGuardrailParameters(
        minimum_vocabulary_size=int(
            value.get("minimum_vocabulary_size", defaults.minimum_vocabulary_size)
        ),
        required_token_families=tuple(
            value.get("required_token_families", defaults.required_token_families)
        ),
        maximum_top_token_fraction=float(
            value.get(
                "maximum_top_token_fraction",
                defaults.maximum_top_token_fraction,
            )
        ),
        maximum_unk_fraction=float(
            value.get("maximum_unk_fraction", defaults.maximum_unk_fraction)
        ),
    )


def coerce_shard_token_counts(
    value: ShardTokenCounts | Mapping[str, Any],
) -> ShardTokenCounts:
    """Coerce JSON-loaded shard token-count payloads into the typed model."""
    if isinstance(value, ShardTokenCounts):
        return value

    return ShardTokenCounts(
        feature_version=str(value["feature_version"]),
        split_version=str(value["split_version"]),
        time_resolution=int(value["time_resolution"]),
        special_token_policy={
            str(key): policy_value
            for key, policy_value in value.get("special_token_policy", {}).items()
        },
        counted_document_count=int(value["counted_document_count"]),
        total_token_count=int(value["total_token_count"]),
        counted_relative_paths=tuple(
            str(relative_path)
            for relative_path in value.get("counted_relative_paths", ())
        ),
        token_counts=tuple(value.get("token_counts", ())),
    )


def coerce_vocabulary_artifact(
    value: VocabularyArtifact | Mapping[str, Any],
) -> VocabularyArtifact:
    """Coerce JSON-loaded frozen vocabulary payloads into the typed model."""
    if isinstance(value, VocabularyArtifact):
        return value

    return VocabularyArtifact(
        vocabulary_version=str(value["vocabulary_version"]),
        feature_version=str(value["feature_version"]),
        split_version=str(value["split_version"]),
        token_count=int(value["token_count"]),
        vocabulary_size=int(value["vocabulary_size"]),
        token_to_id={
            str(token): int(token_id)
            for token, token_id in value["token_to_id"].items()
        },
        token_counts=tuple(value.get("token_counts", ())),
        construction_parameters={
            str(key): item
            for key, item in value.get("construction_parameters", {}).items()
        },
        special_token_policy={
            str(key): item
            for key, item in value.get("special_token_policy", {}).items()
        },
    )


def coerce_vocabulary_stats_report(
    value: VocabularyStatsReport | Mapping[str, Any],
) -> VocabularyStatsReport:
    """Coerce JSON-loaded vocabulary stats payloads into the typed model."""
    if isinstance(value, VocabularyStatsReport):
        return value

    top_tokens = tuple(value.get("top_tokens", ()))
    top_token_payload = top_tokens[0] if top_tokens else None
    top_token = None
    top_token_count = 0
    if isinstance(top_token_payload, Mapping):
        top_token = (
            None
            if top_token_payload.get("token") is None
            else str(top_token_payload["token"])
        )
        top_token_count = int(top_token_payload.get("count", 0))
    elif isinstance(top_token_payload, TokenCountEntry):
        top_token = top_token_payload.token
        top_token_count = top_token_payload.count
    token_count = int(value["token_count"])
    vocabulary_size = int(value["vocabulary_size"])
    unk_token_count = int(value.get("unk_token_count", 0))

    return VocabularyStatsReport(
        vocabulary_version=str(value["vocabulary_version"]),
        feature_version=str(value["feature_version"]),
        split_version=str(value["split_version"]),
        token_count=token_count,
        vocabulary_size=vocabulary_size,
        token_family_coverage=tuple(value.get("token_family_coverage", ())),
        top_tokens=top_tokens,
        unk_token_count=unk_token_count,
        unk_token_fraction=float(value.get("unk_token_fraction", 0.0)),
        guardrails=value.get(
            "guardrails",
            {
                "observed_vocabulary_size": vocabulary_size,
                "minimum_vocabulary_size": vocabulary_size,
                "required_token_families": (),
                "missing_required_token_families": (),
                "top_token": top_token,
                "top_token_count": top_token_count,
                "top_token_fraction": (
                    0.0 if token_count <= 0 else top_token_count / token_count
                ),
                "maximum_top_token_fraction": 1.0,
                "estimated_unk_token_count": 0,
                "estimated_unk_fraction": 0.0,
                "maximum_unk_fraction": 1.0,
                "passed": True,
            },
        ),
        construction_parameters={
            str(key): item
            for key, item in value.get("construction_parameters", {}).items()
        },
    )


def _normalize_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")

    return normalized


def _normalize_fraction(value: float, field_name: str) -> float:
    normalized = float(value)
    if normalized < 0 or normalized > 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return normalized
