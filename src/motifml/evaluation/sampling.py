"""Deterministic prompt and continuation sampling for baseline evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from motifml.datasets.json_dataset import to_json_compatible
from motifml.training.contracts import DatasetSplit
from motifml.training.data_loading import LoadedTokenizedDocument
from motifml.training.model_input import TokenizedDocumentRow
from motifml.training.token_codec import (
    FrozenVocabulary,
    coerce_frozen_vocabulary,
    decode_token_ids_to_strings,
)


@dataclass(frozen=True, slots=True)
class QualitativeSample:
    """One deterministic prompt / continuation review sample."""

    split: DatasetSplit
    relative_path: str
    document_id: str
    prompt_token_ids: tuple[int, ...]
    reference_continuation_token_ids: tuple[int, ...]
    generated_continuation_token_ids: tuple[int, ...]
    prompt_tokens: tuple[str, ...]
    reference_continuation_tokens: tuple[str, ...]
    generated_continuation_tokens: tuple[str, ...]
    prompt_summary: str
    reference_summary: str
    generated_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", DatasetSplit(self.split))
        object.__setattr__(
            self,
            "relative_path",
            _normalize_non_empty_text(self.relative_path, "relative_path"),
        )
        object.__setattr__(
            self,
            "document_id",
            _normalize_non_empty_text(self.document_id, "document_id"),
        )
        object.__setattr__(
            self,
            "prompt_token_ids",
            _normalize_token_ids(self.prompt_token_ids, "prompt_token_ids"),
        )
        object.__setattr__(
            self,
            "reference_continuation_token_ids",
            _normalize_token_ids(
                self.reference_continuation_token_ids,
                "reference_continuation_token_ids",
            ),
        )
        object.__setattr__(
            self,
            "generated_continuation_token_ids",
            _normalize_token_ids(
                self.generated_continuation_token_ids,
                "generated_continuation_token_ids",
            ),
        )
        object.__setattr__(
            self,
            "prompt_tokens",
            tuple(
                _normalize_non_empty_text(token, "prompt_tokens")
                for token in self.prompt_tokens
            ),
        )
        object.__setattr__(
            self,
            "reference_continuation_tokens",
            tuple(
                _normalize_non_empty_text(token, "reference_continuation_tokens")
                for token in self.reference_continuation_tokens
            ),
        )
        object.__setattr__(
            self,
            "generated_continuation_tokens",
            tuple(
                _normalize_non_empty_text(token, "generated_continuation_tokens")
                for token in self.generated_continuation_tokens
            ),
        )
        object.__setattr__(
            self,
            "prompt_summary",
            _normalize_non_empty_text(self.prompt_summary, "prompt_summary"),
        )
        object.__setattr__(
            self,
            "reference_summary",
            _normalize_non_empty_text(self.reference_summary, "reference_summary"),
        )
        object.__setattr__(
            self,
            "generated_summary",
            _normalize_non_empty_text(self.generated_summary, "generated_summary"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the qualitative sample for persisted review artifacts."""
        return to_json_compatible(self)


def generate_greedy_continuation(  # noqa: PLR0913
    model: nn.Module,
    *,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    device: torch.device,
    context_length: int,
    eos_token_id: int | None = None,
) -> tuple[int, ...]:
    """Generate one greedy continuation from a prompt token sequence."""
    normalized_prompt = _normalize_token_ids(prompt_token_ids, "prompt_token_ids")
    normalized_max_new_tokens = _require_positive_int(max_new_tokens, "max_new_tokens")
    normalized_context_length = _require_positive_int(context_length, "context_length")
    if len(normalized_prompt) > normalized_context_length:
        raise ValueError(
            "prompt_token_ids must not exceed the configured context length."
        )

    generated: list[int] = []
    current_tokens = list(normalized_prompt)
    model.eval()
    with torch.no_grad():
        for _ in range(normalized_max_new_tokens):
            model_input = current_tokens[-normalized_context_length:]
            input_ids = torch.tensor([model_input], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            logits = model(input_ids, attention_mask=attention_mask)
            next_token_id = int(logits[0, -1].argmax(dim=-1).item())
            generated.append(next_token_id)
            current_tokens.append(next_token_id)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
    return tuple(generated)


def build_prompt_continuation_samples(  # noqa: PLR0913
    model: nn.Module,
    *,
    documents: Iterable[LoadedTokenizedDocument],
    vocabulary: FrozenVocabulary | Mapping[str, Any],
    samples_per_split: int,
    prompt_token_count: int,
    continuation_token_count: int,
    summary_token_limit: int,
    device: torch.device,
    context_length: int,
    eos_token_id: int | None = None,
) -> tuple[QualitativeSample, ...]:
    """Build deterministic prompt / continuation samples from lazy documents."""
    typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
    normalized_samples_per_split = _require_positive_int(
        samples_per_split,
        "samples_per_split",
    )
    normalized_prompt_token_count = _require_positive_int(
        prompt_token_count,
        "prompt_token_count",
    )
    normalized_continuation_token_count = _require_positive_int(
        continuation_token_count,
        "continuation_token_count",
    )
    normalized_summary_token_limit = _require_positive_int(
        summary_token_limit,
        "summary_token_limit",
    )

    samples: list[QualitativeSample] = []
    for document in documents:
        if len(samples) >= normalized_samples_per_split:
            break
        row = document.row
        prompt_length = min(
            normalized_prompt_token_count,
            max(len(row.token_ids) - 1, 1),
        )
        if prompt_length >= len(row.token_ids):
            continue
        prompt_token_ids = row.token_ids[:prompt_length]
        reference_continuation_token_ids = row.token_ids[
            prompt_length : prompt_length + normalized_continuation_token_count
        ]
        generated_continuation_token_ids = generate_greedy_continuation(
            model,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=normalized_continuation_token_count,
            device=device,
            context_length=context_length,
            eos_token_id=eos_token_id,
        )
        prompt_tokens = decode_token_ids_to_strings(
            prompt_token_ids,
            vocabulary=typed_vocabulary,
        )
        reference_continuation_tokens = decode_token_ids_to_strings(
            reference_continuation_token_ids,
            vocabulary=typed_vocabulary,
        )
        generated_continuation_tokens = decode_token_ids_to_strings(
            generated_continuation_token_ids,
            vocabulary=typed_vocabulary,
        )
        samples.append(
            QualitativeSample(
                split=row.split,
                relative_path=row.relative_path,
                document_id=row.document_id,
                prompt_token_ids=prompt_token_ids,
                reference_continuation_token_ids=reference_continuation_token_ids,
                generated_continuation_token_ids=generated_continuation_token_ids,
                prompt_tokens=prompt_tokens,
                reference_continuation_tokens=reference_continuation_tokens,
                generated_continuation_tokens=generated_continuation_tokens,
                prompt_summary=summarize_decoded_tokens(
                    prompt_tokens,
                    max_tokens=normalized_summary_token_limit,
                ),
                reference_summary=summarize_decoded_tokens(
                    reference_continuation_tokens,
                    max_tokens=normalized_summary_token_limit,
                ),
                generated_summary=summarize_decoded_tokens(
                    generated_continuation_tokens,
                    max_tokens=normalized_summary_token_limit,
                ),
            )
        )
    return tuple(samples)


def summarize_decoded_tokens(
    tokens: Sequence[str],
    *,
    max_tokens: int,
) -> str:
    """Render one decoded token sequence into a compact human-reviewable summary."""
    normalized_max_tokens = _require_positive_int(max_tokens, "max_tokens")
    normalized_tokens = tuple(
        _normalize_non_empty_text(token, "tokens") for token in tokens
    )
    if not normalized_tokens:
        return "<empty>"
    if len(normalized_tokens) <= normalized_max_tokens:
        return " ".join(normalized_tokens)
    visible_tokens = " ".join(normalized_tokens[:normalized_max_tokens])
    omitted_count = len(normalized_tokens) - normalized_max_tokens
    return f"{visible_tokens} ... (+{omitted_count} more)"


def coerce_loaded_tokenized_documents(
    documents: Iterable[
        LoadedTokenizedDocument | TokenizedDocumentRow | Mapping[str, Any]
    ],
    *,
    shard_id: str = "global",
) -> tuple[LoadedTokenizedDocument, ...]:
    """Coerce document-like inputs into the loaded-document contract used by sampling."""
    normalized_shard_id = _normalize_non_empty_text(shard_id, "shard_id")
    coerced: list[LoadedTokenizedDocument] = []
    for document in documents:
        if isinstance(document, LoadedTokenizedDocument):
            coerced.append(document)
            continue
        row = (
            document
            if isinstance(document, TokenizedDocumentRow)
            else TokenizedDocumentRow.from_row_dict(document)
        )
        coerced.append(
            LoadedTokenizedDocument(
                shard_id=normalized_shard_id,
                record_path=(
                    f"records/{row.split.value}/{normalized_shard_id}/"
                    f"{row.relative_path}.model_input.parquet"
                ),
                row=row,
            )
        )
    return tuple(coerced)


def _normalize_token_ids(value: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes):
        raise ValueError(f"{field_name} must be a token-id sequence.")
    normalized: list[int] = []
    for index, token_id in enumerate(value):
        normalized.append(_require_non_negative_int(token_id, f"{field_name}[{index}]"))
    return tuple(normalized)


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


__all__ = [
    "QualitativeSample",
    "build_prompt_continuation_samples",
    "coerce_loaded_tokenized_documents",
    "generate_greedy_continuation",
    "summarize_decoded_tokens",
]
