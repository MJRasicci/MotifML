"""Tests for project performance hooks."""

from __future__ import annotations

import json
from pathlib import Path

from kedro.io import CachedDataset, DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline

from motifml.datasets.json_directory_dataset import JsonDirectoryDataset
from motifml.hooks import ProjectHooks
from motifml.pipelines.feature_extraction.models import (
    FeatureExtractionParameters,
    IrFeatureRecord,
    IrFeatureSet,
    ProjectionType,
)


def test_project_hooks_write_node_and_dataset_timings(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    catalog = DataCatalog(
        datasets={
            "params:performance": MemoryDataset(
                data={
                    "reporting": {
                        "enabled": True,
                        "report_dir": report_dir.as_posix(),
                        "report_basename": "hook-test",
                        "top_n": 5,
                    },
                    "dataset_run_cache": {
                        "enabled": False,
                        "copy_mode": "assign",
                        "release_when_unused": True,
                        "dataset_names": [],
                    },
                }
            )
        }
    )
    pipeline_node = node(
        func=lambda source: source,
        inputs="source",
        outputs="sink",
        name="echo_node",
        tags={"tokenization"},
    )
    hooks = ProjectHooks()
    run_params = {
        "pipeline_names": ["__default__"],
        "runner": "SequentialRunner",
        "is_async": False,
    }

    hooks.before_pipeline_run(run_params, pipeline=None, catalog=catalog)
    hooks.before_dataset_loaded("source", pipeline_node)
    hooks.after_dataset_loaded("source", {"value": 1}, pipeline_node)
    hooks.before_node_run(
        pipeline_node,
        catalog=catalog,
        inputs={"source": {"value": 1}},
        is_async=False,
        run_id="node-run-id",
    )
    hooks.after_node_run(
        pipeline_node,
        catalog=catalog,
        inputs={"source": {"value": 1}},
        outputs={"sink": {"value": 1}},
        is_async=False,
        run_id="node-run-id",
    )
    hooks.before_dataset_saved("sink", {"value": 1}, pipeline_node)
    hooks.after_dataset_saved("sink", {"value": 1}, pipeline_node)
    hooks.after_pipeline_run(run_params, run_result={}, pipeline=None, catalog=catalog)

    report_path = next(report_dir.glob("hook-test_*.json"))
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["run"]["partial"] is False
    assert payload["run"]["pipeline_name"] == "__default__"
    assert payload["run"]["pipeline_names"] == ["__default__"]
    assert len(payload["node_timings"]) == 1
    assert payload["node_timings"][0]["node_name"] == "echo_node"
    assert payload["node_timings"][0]["stage"] == "tokenization"
    assert {record["dataset_name"] for record in payload["dataset_timings"]} == {
        "source",
        "sink",
    }


def test_before_pipeline_run_wraps_configured_datasets_with_cache(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    catalog = DataCatalog(
        datasets={
            "params:performance": MemoryDataset(
                data={
                    "reporting": {
                        "enabled": True,
                        "report_dir": report_dir.as_posix(),
                        "report_basename": "cache-test",
                        "top_n": 5,
                    },
                    "dataset_run_cache": {
                        "enabled": True,
                        "copy_mode": "assign",
                        "release_when_unused": True,
                        "dataset_names": ["demo_dataset"],
                    },
                }
            ),
            "demo_dataset": MemoryDataset(data={"value": 1}),
        }
    )
    hooks = ProjectHooks()

    hooks.before_pipeline_run(
        {
            "pipeline_names": ["__default__"],
            "runner": "SequentialRunner",
            "is_async": False,
        },
        pipeline=None,
        catalog=catalog,
    )

    assert isinstance(catalog.get("demo_dataset"), CachedDataset)


def test_after_node_run_releases_wrapped_cache_after_last_consumer(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    source_dataset = MemoryDataset(data={"value": 1})
    catalog = DataCatalog(
        datasets={
            "params:performance": MemoryDataset(
                data={
                    "reporting": {
                        "enabled": True,
                        "report_dir": report_dir.as_posix(),
                        "report_basename": "release-test",
                        "top_n": 5,
                    },
                    "dataset_run_cache": {
                        "enabled": True,
                        "copy_mode": "assign",
                        "release_when_unused": True,
                        "dataset_names": ["demo_dataset"],
                    },
                }
            ),
            "demo_dataset": source_dataset,
        }
    )
    pipeline_node = node(
        func=lambda demo_dataset: demo_dataset,
        inputs="demo_dataset",
        outputs="sink",
        name="consume_demo_dataset",
    )
    hooks = ProjectHooks()
    run_params = {
        "pipeline_names": ["__default__"],
        "runner": "SequentialRunner",
        "is_async": False,
    }
    demo_pipeline = pipeline([pipeline_node])

    hooks.before_pipeline_run(run_params, pipeline=demo_pipeline, catalog=catalog)

    wrapped_dataset = catalog.get("demo_dataset")
    assert isinstance(wrapped_dataset, CachedDataset)
    wrapped_dataset.load()
    assert wrapped_dataset._cache.exists()

    hooks.before_node_run(
        pipeline_node,
        catalog=catalog,
        inputs={"demo_dataset": {"value": 1}},
        is_async=False,
        run_id="release-run-id",
    )
    hooks.after_node_run(
        pipeline_node,
        catalog=catalog,
        inputs={"demo_dataset": {"value": 1}},
        outputs={"sink": {"value": 1}},
        is_async=False,
        run_id="release-run-id",
    )

    assert wrapped_dataset._cache.exists() is False


def test_project_hooks_write_kedro_viz_stats_for_directory_datasets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool.kedro]\n", encoding="utf-8")

    payload_root = tmp_path / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    first_path = payload_root / "a.json"
    second_path = payload_root / "b.json"
    first_path.write_text('{"name":"alpha","value":1}\n', encoding="utf-8")
    second_path.write_text('{"name":"beta","value":2}\n', encoding="utf-8")
    expected_size = first_path.stat().st_size + second_path.stat().st_size

    catalog = DataCatalog(
        datasets={
            "params:performance": MemoryDataset(
                data={
                    "reporting": {
                        "enabled": False,
                    },
                    "dataset_run_cache": {
                        "enabled": False,
                        "copy_mode": "assign",
                        "release_when_unused": True,
                        "dataset_names": [],
                    },
                }
            ),
            "sample_dir": JsonDirectoryDataset(filepath=str(payload_root)),
        }
    )
    hooks = ProjectHooks()
    pipeline_node = node(
        func=lambda sample_dir: sample_dir,
        inputs="sample_dir",
        outputs="sink",
        name="load_sample_dir",
        tags={"ingestion"},
    )
    run_params = {
        "pipeline_names": ["__default__"],
        "runner": "SequentialRunner",
        "is_async": False,
    }

    hooks.before_pipeline_run(run_params, pipeline=None, catalog=catalog)
    loaded = catalog.load("sample_dir")
    hooks.before_dataset_loaded("sample_dir", pipeline_node)
    hooks.after_dataset_loaded("sample_dir", loaded, pipeline_node)

    viz_dir = tmp_path / ".viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    (viz_dir / "stats.json").write_text(
        json.dumps({"pandas_like": {"rows": 4, "columns": 2, "file_size": 10}}),
        encoding="utf-8",
    )

    hooks.after_pipeline_run(run_params, run_result={}, pipeline=None, catalog=catalog)

    viz_stats = json.loads((viz_dir / "stats.json").read_text(encoding="utf-8"))
    assert viz_stats["pandas_like"] == {"rows": 4, "columns": 2, "file_size": 10}
    assert viz_stats["sample_dir"] == {
        "rows": 2,
        "columns": 2,
        "file_size": expected_size,
    }


def test_project_hooks_write_kedro_viz_stats_for_record_sets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool.kedro]\n", encoding="utf-8")

    catalog = DataCatalog(
        datasets={
            "params:performance": MemoryDataset(
                data={
                    "reporting": {
                        "enabled": False,
                    },
                    "dataset_run_cache": {
                        "enabled": False,
                        "copy_mode": "assign",
                        "release_when_unused": True,
                        "dataset_names": [],
                    },
                }
            ),
            "ir_features": MemoryDataset(),
        }
    )
    hooks = ProjectHooks()
    pipeline_node = node(
        func=lambda ir_features: ir_features,
        inputs="ir_features",
        outputs="sink",
        name="extract_features",
        tags={"feature_extraction"},
    )
    run_params = {
        "pipeline_names": ["__default__"],
        "runner": "SequentialRunner",
        "is_async": False,
    }
    feature_set = IrFeatureSet(
        parameters=FeatureExtractionParameters(),
        records=(
            IrFeatureRecord(
                relative_path="Artist A/Alpha",
                projection_type=ProjectionType.SEQUENCE,
                projection={},
            ),
        ),
    )

    hooks.before_pipeline_run(run_params, pipeline=None, catalog=catalog)
    hooks.before_dataset_saved("ir_features", feature_set, pipeline_node)
    hooks.after_dataset_saved("ir_features", feature_set, pipeline_node)
    hooks.after_pipeline_run(run_params, run_result={}, pipeline=None, catalog=catalog)

    viz_stats = json.loads(
        (tmp_path / ".viz" / "stats.json").read_text(encoding="utf-8")
    )
    assert viz_stats["ir_features"] == {"rows": 1, "columns": 3}
