"""Integration smoke tests for the baseline training pipeline."""

from __future__ import annotations

from pathlib import Path

from motifml.datasets.training_checkpoint_dataset import TrainingCheckpointDataset
from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    load_json,
    run_session,
    write_test_conf,
)


def test_training_pipeline_persists_cpu_baseline_artifacts(tmp_path: Path) -> None:
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)
    runtime_overrides = {
        "data_split": {
            "ratios": {
                "train": 0.5,
                "validation": 0.25,
                "test": 0.25,
            },
            "hash_seed": 17,
            "grouping_strategy": "document_id",
            "grouping_key_fallback": "relative_path",
        },
        "model_input": {
            "projection_type": "sequence",
            "sequence_mode": "baseline_v1",
            "context_length": 64,
            "stride": 32,
            "padding_strategy": "right",
            "special_token_policy": {
                "bos": "document",
                "eos": "document",
                "padding_interaction": "outside_boundaries",
                "unknown_token_mapping": "map_to_unk",
            },
            "storage": {
                "backend": "parquet",
                "schema_version": "parquet-v1",
            },
            "reporting": {
                "worst_document_limit": 10,
                "oversized_token_count_threshold": 8192,
            },
        },
        "model": {
            "architecture": "decoder_only_transformer",
            "embedding_dim": 32,
            "hidden_size": 64,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "positional_encoding": "learned",
        },
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "optimizer": "adamw",
            "lr_scheduler": {
                "name": "constant",
                "warmup_steps": 0,
            },
        },
    }

    run_session(conf_source, ["__default__"], runtime_params=runtime_overrides)
    run_session(conf_source, ["training"], runtime_params=runtime_overrides)

    history = load_json(output_root / "training_history.json")
    run_metadata = load_json(output_root / "training_run_metadata.json")
    training_artifacts = TrainingCheckpointDataset(
        filepath=str(output_root / "training" / "baseline")
    ).load()

    assert history["best_epoch_index"] == 0
    assert history["epochs"][0]["epoch_index"] == 0
    assert run_metadata["training_run_id"]
    assert run_metadata["model_input_version"]
    assert training_artifacts["best_checkpoint"]["checkpoint_name"] == "epoch-0000.pt"
    assert training_artifacts["checkpoints"][0]["state"]["model_state_dict"]
