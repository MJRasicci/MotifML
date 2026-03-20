"""Tests for project performance hooks."""

from __future__ import annotations

import json
from pathlib import Path

from kedro.io import CachedDataset, DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline

from motifml.hooks import ProjectHooks


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
        "pipeline_name": "__default__",
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
            "pipeline_name": "__default__",
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
        "pipeline_name": "__default__",
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
