"""Integration tests for shard-local vocabulary counting and reducer wiring."""

from __future__ import annotations

from pathlib import Path

from motifml.sharding import shard_ids_from_entries
from tests.pipelines.ir_test_support import (
    load_json,
    load_partition_index,
    run_session,
    write_test_conf,
)
from tests.pipelines.training_test_support import materialize_training_fixture_corpus


def test_vocabulary_counting_shards_reduce_into_frozen_vocabulary(
    tmp_path: Path,
) -> None:
    raw_root = materialize_training_fixture_corpus(tmp_path / "raw_training")
    conf_source, output_root = write_test_conf(tmp_path, raw_root)

    run_session(conf_source, ["ingestion"])
    partition_index = load_partition_index(output_root / "raw_partition_index.json")
    shard_ids = shard_ids_from_entries(partition_index)

    for shard_id in shard_ids:
        run_session(
            conf_source,
            ["shard_processing"],
            runtime_params={"execution": {"shard_id": shard_id}},
        )

    run_session(conf_source, ["dataset_splitting"])

    for shard_id in shard_ids:
        run_session(
            conf_source,
            ["vocabulary_counting_shard"],
            runtime_params={"execution": {"shard_id": shard_id}},
        )

    run_session(conf_source, ["partitioned_reduce"])

    token_count_paths = sorted((output_root / "token_counts").glob("*.json"))
    vocabulary = load_json(output_root / "vocabulary.json")
    vocabulary_version = load_json(output_root / "vocabulary_version.json")
    vocab_stats = load_json(output_root / "vocab_stats.json")

    assert len(token_count_paths) == len(shard_ids)
    assert vocabulary["token_to_id"]["<pad>"] == 0
    assert vocabulary["token_to_id"]["<bos>"] == 1
    assert vocabulary["vocabulary_version"] == vocabulary_version["vocabulary_version"]
    assert vocabulary["vocabulary_size"] == len(vocabulary["token_to_id"])
    assert vocab_stats["vocabulary_version"] == vocabulary["vocabulary_version"]
    assert vocab_stats["token_family_coverage"]
    assert vocab_stats["guardrails"]["passed"] is True
