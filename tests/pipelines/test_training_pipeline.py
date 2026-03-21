"""Integration smoke tests for the baseline training pipeline."""

from __future__ import annotations

from pathlib import Path

from motifml.datasets.tokenized_model_input_runtime_dataset import (
    TokenizedModelInputRuntimeDataset,
)
from motifml.datasets.training_checkpoint_dataset import TrainingCheckpointDataset
from tests.pipelines.ir_test_support import (
    load_json,
    load_text,
    run_session,
    write_test_conf,
)
from tests.pipelines.training_test_support import (
    baseline_training_runtime_overrides,
    materialize_training_fixture_corpus,
)


def test_training_pipeline_persists_cpu_baseline_artifacts(tmp_path: Path) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(
        conf_source,
        ["__default__"],
        runtime_params=baseline_training_runtime_overrides(),
    )
    run_session(
        conf_source,
        ["training"],
        runtime_params=baseline_training_runtime_overrides(),
    )

    _assert_persisted_training_artifacts(output_root)


def test_baseline_training_pipeline_runs_in_one_command(tmp_path: Path) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(
        conf_source,
        ["baseline_training"],
        runtime_params=baseline_training_runtime_overrides(),
    )

    _assert_persisted_training_artifacts(output_root)


def test_default_pipeline_outputs_support_lazy_model_input_iteration(
    tmp_path: Path,
) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(
        conf_source,
        ["__default__"],
        runtime_params=baseline_training_runtime_overrides(),
    )

    runtime = TokenizedModelInputRuntimeDataset(
        filepath=str(output_root / "model_input")
    ).load()
    vocabulary = load_json(output_root / "vocabulary.json")

    first_document = next(
        iter(
            runtime.build_document_dataset(
                split="train",
                iteration_options={
                    "shuffle_documents": False,
                },
            )
        )
    )
    first_batch = next(
        iter(
            runtime.build_window_data_loader(
                split="train",
                vocabulary=vocabulary,
                batch_size=1,
                iteration_options={
                    "shuffle_documents": False,
                    "shuffle_windows": False,
                },
            )
        )
    )

    assert first_document.row.split.value == "train"
    assert first_document.row.window_start_offsets
    assert list(first_batch.document_ids) == [first_document.row.document_id]
    assert list(first_batch.relative_paths) == [first_document.row.relative_path]


def _assert_persisted_training_artifacts(output_root: Path) -> None:
    history = load_json(output_root / "training_history.json")
    model_input_report = load_text(output_root / "model_input_report.md")
    run_metadata = load_json(output_root / "training_run_metadata.json")
    training_artifacts = TrainingCheckpointDataset(
        filepath=str(output_root / "training" / "baseline")
    ).load()

    assert history["best_epoch_index"] == 0
    assert history["epochs"][0]["epoch_index"] == 0
    assert "# Model Input Pathology Report" in model_input_report
    assert run_metadata["training_run_id"]
    assert run_metadata["model_input_version"]
    assert training_artifacts["best_checkpoint"]["checkpoint_name"] == "epoch-0000.pt"
    assert training_artifacts["checkpoints"][0]["state"]["model_state_dict"]
