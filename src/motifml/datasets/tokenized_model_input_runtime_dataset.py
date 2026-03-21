"""Read-only Kedro dataset that returns a lazy runtime handle for ``05_model_input``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset, DatasetError

from motifml.datasets.model_input_storage import (
    MODEL_INPUT_PARAMETERS_FILENAME,
    MODEL_INPUT_STORAGE_SCHEMA_FILENAME,
    coerce_model_input_storage_schema,
)
from motifml.training.contracts import ModelInputMetadata
from motifml.training.model_input_runtime import TokenizedModelInputRuntimeHandle


class TokenizedModelInputRuntimeDataset(
    AbstractDataset[TokenizedModelInputRuntimeHandle, Any]
):
    """Expose the tokenized model-input root as a lazy runtime handle."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> TokenizedModelInputRuntimeHandle:
        """Load the runtime handle without materializing persisted document rows."""
        if not self._filepath.exists():
            raise DatasetError(
                "Tokenized model-input directory does not exist: "
                f"{self._filepath.as_posix()}"
            )

        metadata_payload = self._load_json(
            self._filepath / MODEL_INPUT_PARAMETERS_FILENAME
        )
        if not isinstance(metadata_payload, dict):
            raise DatasetError(
                "Tokenized model-input metadata is missing or malformed at "
                f"{(self._filepath / MODEL_INPUT_PARAMETERS_FILENAME).as_posix()}."
            )
        schema_payload = self._load_json(
            self._filepath / MODEL_INPUT_STORAGE_SCHEMA_FILENAME
        )
        if not isinstance(schema_payload, dict):
            raise DatasetError(
                "Tokenized model-input storage schema is missing or malformed at "
                f"{(self._filepath / MODEL_INPUT_STORAGE_SCHEMA_FILENAME).as_posix()}."
            )
        return TokenizedModelInputRuntimeHandle(
            dataset_root=self._filepath.as_posix(),
            metadata=ModelInputMetadata.from_json_dict(metadata_payload),
            storage_schema=coerce_model_input_storage_schema(schema_payload),
        )

    def save(self, data: Any) -> None:
        """Reject save calls because the runtime handle is read-only."""
        del data
        raise DatasetError(
            "TokenizedModelInputRuntimeDataset is read-only and cannot be saved."
        )

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath.as_posix()}

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)


__all__ = ["TokenizedModelInputRuntimeDataset"]
