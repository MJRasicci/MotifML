"""Run the shard-scoped MotifML Kedro pipelines end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

from motifml.sharding import shard_ids_from_entries


def main() -> int:
    """Run ingestion, shard processing, and global reducers."""
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    partition_index_path = repo_root / args.partition_index_path

    if not args.skip_partition:
        _run_kedro_pipeline(
            repo_root=repo_root,
            pipeline_name="ingestion",
            conf_source=args.conf_source,
        )

    partition_index = json.loads(partition_index_path.read_text(encoding="utf-8"))
    shard_ids = list(shard_ids_from_entries(partition_index))
    if args.shard_limit is not None:
        shard_ids = shard_ids[: args.shard_limit]
    if not shard_ids:
        raise SystemExit("No shard IDs were found in the partition index.")

    logs_root = repo_root / "temp" / "shard_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    if args.max_workers <= 1:
        for shard_id in shard_ids:
            _run_shard(repo_root, shard_id, args.conf_source, logs_root)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    _run_shard,
                    repo_root,
                    shard_id,
                    args.conf_source,
                    logs_root,
                ): shard_id
                for shard_id in shard_ids
            }
            done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                future.result()
            for future in not_done:
                future.result()

    if not args.skip_reduce:
        _run_kedro_pipeline(
            repo_root=repo_root,
            pipeline_name="partitioned_reduce",
            conf_source=args.conf_source,
        )

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shard-scoped MotifML Kedro pipelines."
    )
    parser.add_argument(
        "--conf-source",
        help="Optional Kedro conf source directory.",
    )
    parser.add_argument(
        "--partition-index-path",
        default="data/02_intermediate/ingestion/raw_partition_index.json",
        help="Partition-index path relative to the repository root.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of shard Kedro runs to execute concurrently.",
    )
    parser.add_argument(
        "--shard-limit",
        type=int,
        help="Optional limit on the number of shards to process.",
    )
    parser.add_argument(
        "--skip-partition",
        action="store_true",
        help="Skip the ingestion step that refreshes the partition index.",
    )
    parser.add_argument(
        "--skip-reduce",
        action="store_true",
        help="Skip the global reducer step.",
    )
    return parser.parse_args()


def _run_shard(
    repo_root: Path,
    shard_id: str,
    conf_source: str | None,
    logs_root: Path,
) -> None:
    log_path = logs_root / f"{shard_id}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        _run_kedro_pipeline(
            repo_root=repo_root,
            pipeline_name="shard_processing",
            conf_source=conf_source,
            runtime_param=f"execution.shard_id={shard_id}",
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def _run_kedro_pipeline(  # noqa: PLR0913
    *,
    repo_root: Path,
    pipeline_name: str,
    conf_source: str | None,
    runtime_param: str | None = None,
    stdout: object | None = None,
    stderr: object | None = None,
) -> None:
    command = [sys.executable, "-m", "kedro", "run", f"--pipeline={pipeline_name}"]
    if conf_source is not None:
        command.append(f"--conf-source={conf_source}")
    if runtime_param is not None:
        command.append(f"--params={runtime_param}")

    completed = subprocess.run(  # noqa: S603
        command,
        cwd=repo_root,
        check=False,
        stdout=stdout,
        stderr=stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Kedro pipeline '{pipeline_name}' failed with exit code "
            f"{completed.returncode}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
