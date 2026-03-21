"""Lazy runtime accessors for persisted tokenized model-input artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from motifml.datasets.model_input_storage import ModelInputStorageSchema
from motifml.training.contracts import DatasetSplit, ModelInputMetadata
from motifml.training.data_loading import (
    LazyTokenizedDocumentDataset,
    LazyTokenWindowDataset,
    LoaderIterationOptions,
    SpecialTokenIds,
    TokenWindowBatch,
    build_token_window_data_loader,
    coerce_loader_iteration_options,
)
from motifml.training.token_codec import FrozenVocabulary, coerce_frozen_vocabulary


@dataclass(frozen=True, slots=True)
class TokenizedModelInputRuntimeHandle:
    """Lazy runtime handle over one persisted tokenized model-input dataset root."""

    dataset_root: str
    metadata: ModelInputMetadata
    storage_schema: ModelInputStorageSchema

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_root",
            _normalize_non_empty_text(self.dataset_root, "dataset_root"),
        )

    def build_document_dataset(
        self,
        *,
        split: DatasetSplit | str,
        shard_ids: Sequence[str] | None = None,
        iteration_options: LoaderIterationOptions | Mapping[str, Any] | None = None,
    ) -> LazyTokenizedDocumentDataset:
        """Build one lazy split-scoped document dataset from the persisted root."""
        return LazyTokenizedDocumentDataset(
            self.dataset_root,
            split=split,
            shard_ids=shard_ids,
            iteration_options=iteration_options,
        )

    def build_window_dataset(
        self,
        *,
        split: DatasetSplit | str,
        vocabulary: FrozenVocabulary | Mapping[str, Any],
        shard_ids: Sequence[str] | None = None,
        iteration_options: LoaderIterationOptions | Mapping[str, Any] | None = None,
        drop_empty_targets: bool = True,
    ) -> LazyTokenWindowDataset:
        """Build one lazy split-scoped window dataset from persisted rows."""
        typed_iteration_options = coerce_loader_iteration_options(iteration_options)
        typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
        documents = self.build_document_dataset(
            split=split,
            shard_ids=shard_ids,
            iteration_options=typed_iteration_options,
        )
        return LazyTokenWindowDataset(
            documents,
            split=split,
            special_token_ids=SpecialTokenIds.from_vocabulary(typed_vocabulary),
            drop_empty_targets=drop_empty_targets,
            iteration_options=typed_iteration_options,
        )

    def build_window_data_loader(  # noqa: PLR0913
        self,
        *,
        split: DatasetSplit | str,
        vocabulary: FrozenVocabulary | Mapping[str, Any],
        batch_size: int,
        shard_ids: Sequence[str] | None = None,
        iteration_options: LoaderIterationOptions | Mapping[str, Any] | None = None,
        drop_empty_targets: bool = True,
    ) -> DataLoader[TokenWindowBatch]:
        """Build one lazy bounded-memory training DataLoader for a split."""
        typed_vocabulary = coerce_frozen_vocabulary(vocabulary)
        return build_token_window_data_loader(
            self.build_window_dataset(
                split=split,
                vocabulary=typed_vocabulary,
                shard_ids=shard_ids,
                iteration_options=iteration_options,
                drop_empty_targets=drop_empty_targets,
            ),
            batch_size=batch_size,
            pad_token_id=typed_vocabulary.token_to_id["<pad>"],
        )

    def count_batches(  # noqa: PLR0913
        self,
        *,
        split: DatasetSplit | str,
        vocabulary: FrozenVocabulary | Mapping[str, Any],
        batch_size: int,
        shard_ids: Sequence[str] | None = None,
        iteration_options: LoaderIterationOptions | Mapping[str, Any] | None = None,
        drop_empty_targets: bool = True,
    ) -> int:
        """Count the lazy loader batches without materializing the corpus at once."""
        return sum(
            1
            for _ in self.build_window_data_loader(
                split=split,
                vocabulary=vocabulary,
                batch_size=batch_size,
                shard_ids=shard_ids,
                iteration_options=iteration_options,
                drop_empty_targets=drop_empty_targets,
            )
        )

    @property
    def dataset_path(self) -> Path:
        """Return the dataset root as a ``Path`` for diagnostics and tests."""
        return Path(self.dataset_root)


def _normalize_non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


__all__ = ["TokenizedModelInputRuntimeHandle"]
