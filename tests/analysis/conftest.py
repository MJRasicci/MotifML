"""Shared pytest fixtures for notebook integration coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.analysis.notebook_test_support import build_training_runtime_artifact_root


@pytest.fixture(scope="session")
def training_runtime_artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one shared tiny end-to-end runtime artifact tree for notebook tests."""
    return build_training_runtime_artifact_root(
        tmp_path_factory.mktemp("training_runtime_artifacts")
    )
