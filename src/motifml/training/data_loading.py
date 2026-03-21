"""Lazy training-time loaders over Parquet-backed ``05_model_input`` rows."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from kedro.io import DatasetError
from torch.utils.data import IterableDataset

from motifml.datasets.model_input_storage import (
    MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
    ModelInputStorageSchema,
    coerce_model_input_storage_schema,
)
from motifml.training.contracts import DatasetSplit
from motifml.training.model_input import (
    TokenizedDocumentRow,
    coerce_tokenized_document_row,
)
from motifml.training.special_token_policy import (
    PaddingInteraction,
    coerce_special_token_policy,
)
from motifml.training.token_codec import FrozenVocabulary, coerce_frozen_vocabulary
from motifml.training.token_families import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


@dataclass(frozen=True, slots=True)
class LoadedTokenizedDocument:
    """One lazily loaded tokenized document plus its persisted shard metadata."""

    shard_id: str
    record_path: str
    row: TokenizedDocumentRow


@dataclass(frozen=True, slots=True)
class SpecialTokenIds:
    """Frozen token ids for the special-token family required at load time."""

    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pad_token_id",
            _normalize_non_negative_int(self.pad_token_id, "pad_token_id"),
        )
        object.__setattr__(
            self,
            "bos_token_id",
            _normalize_non_negative_int(self.bos_token_id, "bos_token_id"),
        )
        object.__setattr__(
            self,
            "eos_token_id",
            _normalize_non_negative_int(self.eos_token_id, "eos_token_id"),
        )
        object.__setattr__(
            self,
            "unk_token_id",
            _normalize_non_negative_int(self.unk_token_id, "unk_token_id"),
        )

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: FrozenVocabulary | Mapping[str, Any],
    ) -> SpecialTokenIds:
        """Resolve special token ids from one frozen vocabulary artifact."""
        typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
        return cls(
            pad_token_id=typed_vocabulary.token_to_id[PAD_TOKEN],
            bos_token_id=typed_vocabulary.token_to_id[BOS_TOKEN],
            eos_token_id=typed_vocabulary.token_to_id[EOS_TOKEN],
            unk_token_id=typed_vocabulary.token_to_id[UNK_TOKEN],
        )


@dataclass(frozen=True, slots=True)
class TokenWindowExample:
    """One lazily reconstructed next-token training example."""

    split: DatasetSplit
    shard_id: str
    relative_path: str
    document_id: str
    window_index: int
    window_start_offset: int
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", DatasetSplit(self.split))
        object.__setattr__(
            self, "shard_id", _normalize_non_empty_text(self.shard_id, "shard_id")
        )
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
            "window_index",
            _normalize_non_negative_int(self.window_index, "window_index"),
        )
        object.__setattr__(
            self,
            "window_start_offset",
            _normalize_non_negative_int(
                self.window_start_offset,
                "window_start_offset",
            ),
        )
        normalized_input_ids = tuple(
            _normalize_non_negative_int(token_id, f"input_ids[{index}]")
            for index, token_id in enumerate(self.input_ids)
        )
        normalized_target_ids = tuple(
            _normalize_non_negative_int(token_id, f"target_ids[{index}]")
            for index, token_id in enumerate(self.target_ids)
        )
        normalized_attention_mask = tuple(
            _normalize_attention_value(mask_value, index)
            for index, mask_value in enumerate(self.attention_mask)
        )
        if len(normalized_input_ids) != len(normalized_target_ids):
            raise ValueError("input_ids and target_ids must have matching lengths.")
        if len(normalized_input_ids) != len(normalized_attention_mask):
            raise ValueError("input_ids and attention_mask must have matching lengths.")
        object.__setattr__(self, "input_ids", normalized_input_ids)
        object.__setattr__(self, "target_ids", normalized_target_ids)
        object.__setattr__(self, "attention_mask", normalized_attention_mask)


def load_tokenized_document_row_file(
    path: str | Path,
) -> TokenizedDocumentRow:
    """Load one persisted Parquet row from the tokenized model-input dataset."""
    record_path = Path(path)
    table = pq.read_table(record_path)
    rows = table.to_pylist()
    if len(rows) != 1:
        raise DatasetError(
            "Each tokenized model-input Parquet file must contain exactly one row: "
            f"{record_path.as_posix()}."
        )
    return coerce_tokenized_document_row(rows[0])


def discover_model_input_shards(
    dataset_root: str | Path,
    *,
    split: DatasetSplit | str,
) -> tuple[str, ...]:
    """Return the available shard ids for one split in deterministic order."""
    normalized_split = DatasetSplit(split)
    split_root = Path(dataset_root) / "records" / normalized_split.value
    if not split_root.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in split_root.iterdir()
            if path.is_dir() and path.name not in {".", ".."}
        )
    )


class LazyTokenizedDocumentDataset(IterableDataset[LoadedTokenizedDocument]):
    """Iterate tokenized document rows lazily by split and shard.

    The dataset only discovers shard directories up front. Each Parquet row is loaded
    on demand as the iterator advances, so callers do not need to materialize the full
    ``05_model_input`` corpus before training.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: DatasetSplit | str,
        shard_ids: Sequence[str] | None = None,
        row_loader: Callable[[str | Path], TokenizedDocumentRow] | None = None,
        storage_schema: ModelInputStorageSchema | dict[str, Any] | None = None,
    ) -> None:
        self._dataset_root = Path(dataset_root)
        self._split = DatasetSplit(split)
        self._shard_ids = (
            tuple(_normalize_shard_id(shard_id) for shard_id in shard_ids)
            if shard_ids is not None
            else None
        )
        self._row_loader = (
            load_tokenized_document_row_file if row_loader is None else row_loader
        )
        self._storage_schema = self._load_storage_schema(storage_schema)

    def __iter__(self) -> Iterator[LoadedTokenizedDocument]:
        """Yield tokenized document rows lazily in deterministic split/shard order."""
        for shard_id in self.shard_ids:
            for record_path in self._iter_shard_record_paths(shard_id):
                row = self._row_loader(record_path)
                if row.split is not self._split:
                    raise DatasetError(
                        "Persisted row split does not match the requested split: "
                        f"{record_path.as_posix()}."
                    )
                yield LoadedTokenizedDocument(
                    shard_id=shard_id,
                    record_path=record_path.relative_to(self._dataset_root).as_posix(),
                    row=row,
                )

    @property
    def split(self) -> DatasetSplit:
        """Return the split this lazy dataset is scoped to."""
        return self._split

    @property
    def shard_ids(self) -> tuple[str, ...]:
        """Return the deterministic shard order visible to this dataset."""
        if self._shard_ids is None:
            return discover_model_input_shards(self._dataset_root, split=self._split)
        return self._shard_ids

    def _iter_shard_record_paths(self, shard_id: str) -> Iterator[Path]:
        shard_root = self._dataset_root / "records" / self._split.value / shard_id
        if not shard_root.exists():
            return
        yield from sorted(shard_root.rglob(f"*{self._storage_schema.record_suffix}"))

    def _load_storage_schema(
        self,
        configured_schema: ModelInputStorageSchema | dict[str, Any] | None,
    ) -> ModelInputStorageSchema:
        schema_path = self._dataset_root / MODEL_INPUT_STORAGE_SCHEMA_FILENAME
        if schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as stream:
                persisted_schema = json.load(stream)
            return coerce_model_input_storage_schema(persisted_schema)
        return coerce_model_input_storage_schema(configured_schema)


class LazyTokenWindowDataset(IterableDataset[TokenWindowExample]):
    """Iterate next-token training windows lazily from persisted document rows."""

    def __init__(
        self,
        documents: Iterable[LoadedTokenizedDocument],
        *,
        special_token_ids: SpecialTokenIds | FrozenVocabulary | Mapping[str, Any],
        drop_empty_targets: bool = True,
    ) -> None:
        self._documents = documents
        self._special_token_ids = coerce_special_token_ids(special_token_ids)
        self._drop_empty_targets = bool(drop_empty_targets)

    def __iter__(self) -> Iterator[TokenWindowExample]:
        """Yield reconstructed windows in persisted offset order."""
        for document in self._documents:
            for window_index, window_start_offset in enumerate(
                document.row.window_start_offsets
            ):
                example = build_token_window_example(
                    document.row,
                    shard_id=document.shard_id,
                    window_index=window_index,
                    window_start_offset=window_start_offset,
                    special_token_ids=self._special_token_ids,
                )
                if self._drop_empty_targets and not any(example.attention_mask):
                    continue
                yield example


def build_token_window_example(
    row: TokenizedDocumentRow,
    *,
    shard_id: str,
    window_index: int,
    window_start_offset: int,
    special_token_ids: SpecialTokenIds | FrozenVocabulary | Mapping[str, Any],
) -> TokenWindowExample:
    """Reconstruct one next-token training window from a persisted document row."""
    normalized_shard_id = _normalize_shard_id(shard_id)
    typed_special_token_ids = coerce_special_token_ids(special_token_ids)
    normalized_window_index = _normalize_non_negative_int(
        window_index,
        "window_index",
    )
    normalized_window_start_offset = _normalize_non_negative_int(
        window_start_offset,
        "window_start_offset",
    )
    if normalized_window_start_offset not in row.window_start_offsets:
        raise ValueError(
            "window_start_offset must be one of the persisted window_start_offsets."
        )

    raw_input_ids = tuple(
        row.token_ids[
            normalized_window_start_offset : normalized_window_start_offset
            + row.context_length
        ]
    )
    raw_target_ids = tuple(
        row.token_ids[
            normalized_window_start_offset + 1 : normalized_window_start_offset
            + row.context_length
            + 1
        ]
    )
    if row.padding_strategy == "none":
        semantic_length = len(raw_target_ids)
        return TokenWindowExample(
            split=row.split,
            shard_id=normalized_shard_id,
            relative_path=row.relative_path,
            document_id=row.document_id,
            window_index=normalized_window_index,
            window_start_offset=normalized_window_start_offset,
            input_ids=raw_input_ids[:semantic_length],
            target_ids=raw_target_ids,
            attention_mask=tuple(1 for _ in range(semantic_length)),
        )

    aligned_target_ids = raw_target_ids + (typed_special_token_ids.pad_token_id,) * max(
        0, len(raw_input_ids) - len(raw_target_ids)
    )
    semantic_positions = _build_semantic_positions(
        raw_input_ids=raw_input_ids,
        row=row,
        window_start_offset=normalized_window_start_offset,
        special_token_ids=typed_special_token_ids,
    )
    input_ids = [typed_special_token_ids.pad_token_id] * row.context_length
    target_ids = [typed_special_token_ids.pad_token_id] * row.context_length
    attention_mask = [0] * row.context_length

    for semantic_index, semantic_position in enumerate(semantic_positions):
        input_ids[semantic_position] = raw_input_ids[semantic_index]
        target_ids[semantic_position] = aligned_target_ids[semantic_index]
        if semantic_index < len(raw_target_ids):
            attention_mask[semantic_position] = 1

    return TokenWindowExample(
        split=row.split,
        shard_id=normalized_shard_id,
        relative_path=row.relative_path,
        document_id=row.document_id,
        window_index=normalized_window_index,
        window_start_offset=normalized_window_start_offset,
        input_ids=tuple(input_ids),
        target_ids=tuple(target_ids),
        attention_mask=tuple(attention_mask),
    )


def coerce_special_token_ids(
    value: SpecialTokenIds | FrozenVocabulary | Mapping[str, Any],
) -> SpecialTokenIds:
    """Coerce vocabulary-backed special token ids into the typed loader contract."""
    if isinstance(value, SpecialTokenIds):
        return value

    if isinstance(value, FrozenVocabulary) or "token_to_id" in value:
        return SpecialTokenIds.from_vocabulary(value)

    return SpecialTokenIds(
        pad_token_id=int(value["pad_token_id"]),
        bos_token_id=int(value["bos_token_id"]),
        eos_token_id=int(value["eos_token_id"]),
        unk_token_id=int(value["unk_token_id"]),
    )


def _build_semantic_positions(
    *,
    raw_input_ids: tuple[int, ...],
    row: TokenizedDocumentRow,
    window_start_offset: int,
    special_token_ids: SpecialTokenIds,
) -> tuple[int, ...]:
    semantic_positions: tuple[int, ...] = ()
    if raw_input_ids:
        pad_count = row.context_length - len(raw_input_ids)
        if pad_count <= 0:
            semantic_positions = tuple(range(len(raw_input_ids)))
        else:
            policy = coerce_special_token_policy(row.special_token_policy)
            is_first_window = window_start_offset == row.window_start_offsets[0]
            is_last_window = window_start_offset == row.window_start_offsets[-1]
            prefix_tokens, suffix_tokens = policy.boundary_tokens_for_window(
                is_first_window=is_first_window,
                is_last_window=is_last_window,
            )
            has_leading_boundary = (
                bool(prefix_tokens)
                and raw_input_ids[0] == special_token_ids.bos_token_id
            )
            has_trailing_boundary = (
                bool(suffix_tokens)
                and raw_input_ids[-1] == special_token_ids.eos_token_id
            )
            if row.padding_strategy == "left":
                semantic_positions = tuple(range(pad_count, row.context_length))
                if (
                    policy.padding_interaction is PaddingInteraction.INSIDE_BOUNDARIES
                    and has_leading_boundary
                ):
                    semantic_positions = (
                        (0,)
                        if len(raw_input_ids) == 1
                        else (0,)
                        + tuple(
                            range(
                                row.context_length - (len(raw_input_ids) - 1),
                                row.context_length,
                            )
                        )
                    )
            else:
                semantic_positions = tuple(range(len(raw_input_ids)))
                if (
                    policy.padding_interaction is PaddingInteraction.INSIDE_BOUNDARIES
                    and has_trailing_boundary
                ):
                    semantic_positions = (
                        (row.context_length - 1,)
                        if len(raw_input_ids) == 1
                        else tuple(range(len(raw_input_ids) - 1))
                        + (row.context_length - 1,)
                    )
    return semantic_positions


def _normalize_shard_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("shard_ids must be non-empty.")
    if "/" in normalized or normalized in {".", ".."}:
        raise ValueError("shard_ids must be safe path segments.")
    return normalized


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _normalize_attention_value(value: Any, index: int) -> int:
    normalized = _normalize_non_negative_int(value, f"attention_mask[{index}]")
    if normalized not in {0, 1}:
        raise ValueError("attention_mask values must be binary.")
    return normalized


__all__ = [
    "LazyTokenWindowDataset",
    "LazyTokenizedDocumentDataset",
    "LoadedTokenizedDocument",
    "SpecialTokenIds",
    "TokenWindowExample",
    "build_token_window_example",
    "coerce_special_token_ids",
    "discover_model_input_shards",
    "load_tokenized_document_row_file",
]
