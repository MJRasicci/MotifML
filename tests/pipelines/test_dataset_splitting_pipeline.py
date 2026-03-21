"""Integration tests for the dataset splitting pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

from motifml.sharding import shard_ids_from_entries
from tests.pipelines.ir_test_support import (
    MOTIF_JSON_FIXTURE_ROOT,
    load_json,
    load_partition_index,
    run_session,
    write_test_conf,
)


def test_dataset_splitting_pipeline_persists_stable_manifests_for_unchanged_inputs(
    tmp_path: Path,
) -> None:
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    run_session(conf_source, ["dataset_splitting"])
    first_bytes = (output_root / "split_manifest.json").read_bytes()
    first_stats_bytes = (output_root / "split_stats.json").read_bytes()

    run_session(conf_source, ["dataset_splitting"])
    second_bytes = (output_root / "split_manifest.json").read_bytes()
    second_stats_bytes = (output_root / "split_stats.json").read_bytes()

    assert first_bytes == second_bytes
    assert first_stats_bytes == second_stats_bytes


def test_dataset_splitting_pipeline_is_independent_of_shard_execution(
    tmp_path: Path,
) -> None:
    global_conf, global_output = write_test_conf(
        tmp_path / "global",
        MOTIF_JSON_FIXTURE_ROOT,
    )
    sharded_conf, sharded_output = write_test_conf(
        tmp_path / "sharded",
        MOTIF_JSON_FIXTURE_ROOT,
    )

    run_session(global_conf, ["ir_build"])
    run_session(global_conf, ["normalization"])
    run_session(global_conf, ["dataset_splitting"])
    global_manifest = load_json(global_output / "split_manifest.json")
    global_stats = load_json(global_output / "split_stats.json")

    run_session(sharded_conf, ["ingestion"])
    partition_index = load_partition_index(sharded_output / "raw_partition_index.json")
    for shard_id in shard_ids_from_entries(partition_index):
        run_session(
            sharded_conf,
            ["shard_processing"],
            runtime_params={"execution": {"shard_id": shard_id}},
        )
    run_session(sharded_conf, ["dataset_splitting"])
    sharded_manifest = load_json(sharded_output / "split_manifest.json")
    sharded_stats = load_json(sharded_output / "split_stats.json")

    assert sharded_manifest == global_manifest
    assert sharded_stats == global_stats


def test_dataset_splitting_pipeline_honors_ratio_overrides(tmp_path: Path) -> None:
    conf_source, output_root = write_test_conf(tmp_path, MOTIF_JSON_FIXTURE_ROOT)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    run_session(
        conf_source,
        ["dataset_splitting"],
        runtime_params={
            "data_split": {
                "ratios": {"train": 0.0, "validation": 1.0, "test": 0.0},
            }
        },
    )

    manifest = load_json(output_root / "split_manifest.json")
    split_stats = load_json(output_root / "split_stats.json")

    assert manifest
    assert {entry["split"] for entry in manifest} == {"validation"}
    assert split_stats["splits"][1]["document_count"] == len(manifest)


def test_dataset_splitting_pipeline_keeps_parent_directory_groups_together(
    tmp_path: Path,
) -> None:
    raw_root = _build_parent_group_fixture_corpus(tmp_path / "raw_grouped")
    conf_source, output_root = write_test_conf(tmp_path / "conf_grouped", raw_root)

    run_session(conf_source, ["ir_build"])
    run_session(conf_source, ["normalization"])
    run_session(
        conf_source,
        ["dataset_splitting"],
        runtime_params={
            "data_split": {
                "grouping_strategy": "parent_directory",
                "grouping_key_fallback": "relative_path",
            }
        },
    )

    manifest = load_json(output_root / "split_manifest.json")
    split_stats = load_json(output_root / "split_stats.json")
    by_path = {entry["relative_path"]: entry for entry in manifest}

    assert by_path["collection_a/song_one.json"]["group_key"] == "collection_a"
    assert by_path["collection_a/song_two.json"]["group_key"] == "collection_a"
    assert (
        by_path["collection_a/song_one.json"]["split"]
        == by_path["collection_a/song_two.json"]["split"]
    )
    assert split_stats["split_version"] == manifest[0]["split_version"]


def _build_parent_group_fixture_corpus(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    copy_plan = {
        "collection_a/song_one.json": MOTIF_JSON_FIXTURE_ROOT
        / "single_track_monophonic_pickup.json",
        "collection_a/song_two.json": MOTIF_JSON_FIXTURE_ROOT / "voice_reentry.json",
        "collection_b/song_three.json": MOTIF_JSON_FIXTURE_ROOT
        / "ensemble_polyphony_controls.json",
    }
    for relative_path, source_path in copy_plan.items():
        target_path = destination / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    return destination
