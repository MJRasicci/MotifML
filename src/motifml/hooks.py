"""Project hooks for performance profiling and run-local dataset caching."""

from __future__ import annotations

import json
import logging
import resource
import sys
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns, thread_time_ns
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import CachedDataset, DatasetNotFoundError
from kedro.pipeline import Node, Pipeline

LOGGER = logging.getLogger(__name__)

_KNOWN_PIPELINE_TAGS = (
    "ingestion",
    "ir_build",
    "ir_validation",
    "normalization",
    "feature_extraction",
    "tokenization",
)


@dataclass(frozen=True)
class _ActiveTiming:
    wall_start_ns: int
    cpu_start_ns: int
    rss_start_bytes: int | None
    peak_rss_start_bytes: int | None


@dataclass(frozen=True)
class _NodeTimingRecord:
    node_name: str
    stage: str
    tags: tuple[str, ...]
    is_async: bool
    wall_seconds: float
    cpu_seconds: float
    cpu_utilization_ratio: float
    rss_start_bytes: int | None
    rss_end_bytes: int | None
    rss_delta_bytes: int | None
    peak_rss_start_bytes: int | None
    peak_rss_end_bytes: int | None
    peak_rss_delta_bytes: int | None
    input_dataset_names: tuple[str, ...]
    output_dataset_names: tuple[str, ...]


@dataclass(frozen=True)
class _DatasetTimingRecord:
    dataset_name: str
    node_name: str
    stage: str
    operation: str
    wall_seconds: float
    cpu_seconds: float
    cpu_utilization_ratio: float
    rss_start_bytes: int | None
    rss_end_bytes: int | None
    rss_delta_bytes: int | None
    peak_rss_start_bytes: int | None
    peak_rss_end_bytes: int | None
    peak_rss_delta_bytes: int | None
    data_type: str
    item_count: int | None


@dataclass(frozen=True)
class _DatasetCacheReleaseRecord:
    dataset_name: str
    released_after_node: str


@dataclass
class _PerformanceConfig:
    reporting_enabled: bool = True
    report_dir: str = "data/08_reporting/performance"
    report_basename: str = "pipeline_profile"
    top_n: int = 10
    dataset_run_cache_enabled: bool = True
    dataset_run_cache_copy_mode: str = "assign"
    dataset_run_cache_release_when_unused: bool = True
    dataset_run_cache_dataset_names: tuple[str, ...] = (
        "raw_motif_json_corpus",
        "motif_ir_corpus",
        "normalized_ir_corpus",
        "motif_ir_manifest",
        "ir_features",
    )


class _PerformanceRunState:
    """Thread-safe timing state for a single Kedro pipeline run."""

    def __init__(
        self,
        *,
        run_params: dict[str, Any],
        config: _PerformanceConfig,
    ) -> None:
        self.run_params = dict(run_params)
        self.config = config
        self.started_at_utc = datetime.now(UTC)
        run_id = str(self.run_params.get("run_id", "unknown-run"))
        timestamp = self.started_at_utc.strftime("%Y%m%dT%H%M%SZ")
        self.report_dir = Path(self.config.report_dir)
        self.report_path = self.report_dir / (
            f"{self.config.report_basename}_{timestamp}_{run_id}.json"
        )
        self.partial_report_path = self.report_dir / (
            f"{self.config.report_basename}_{timestamp}_{run_id}.partial.json"
        )
        self.pipeline_wall_start_ns = perf_counter_ns()
        self.pipeline_cpu_start_ns = thread_time_ns()
        self.node_timings: list[_NodeTimingRecord] = []
        self.dataset_timings: list[_DatasetTimingRecord] = []
        self._active_node_timings: dict[tuple[int, str], _ActiveTiming] = {}
        self._active_dataset_timings: dict[
            tuple[int, str, str, str], _ActiveTiming
        ] = {}
        self._wrapped_datasets: list[str] = []
        self._wrapped_dataset_names: set[str] = set()
        self._released_wrapped_dataset_names: set[str] = set()
        self.dataset_cache_releases: list[_DatasetCacheReleaseRecord] = []
        self._remaining_dataset_loads: Counter[str] = Counter()
        self._finalized = False
        self._lock = threading.Lock()

    def record_wrapped_dataset(self, dataset_name: str) -> None:
        with self._lock:
            self._wrapped_datasets.append(dataset_name)
            self._wrapped_dataset_names.add(dataset_name)

    def initialize_pipeline(self, pipeline: Pipeline | None) -> None:
        if pipeline is None:
            return

        with self._lock:
            for node in pipeline.nodes:
                self._remaining_dataset_loads.update(node.inputs)

    def releasable_wrapped_datasets(self, node: Node) -> tuple[str, ...]:
        releasable: list[str] = []
        with self._lock:
            for dataset_name in node.inputs:
                if dataset_name not in self._remaining_dataset_loads:
                    continue

                self._remaining_dataset_loads[dataset_name] -= 1
                if (
                    self._remaining_dataset_loads[dataset_name] <= 0
                    and dataset_name in self._wrapped_dataset_names
                    and dataset_name not in self._released_wrapped_dataset_names
                ):
                    self._released_wrapped_dataset_names.add(dataset_name)
                    releasable.append(dataset_name)
                    self.dataset_cache_releases.append(
                        _DatasetCacheReleaseRecord(
                            dataset_name=dataset_name,
                            released_after_node=node.name,
                        )
                    )

        return tuple(sorted(releasable))

    def start_node(self, node: Node, *, is_async: bool) -> None:
        del is_async
        rss_start_bytes, peak_rss_start_bytes = _capture_memory_snapshot()
        with self._lock:
            self._active_node_timings[(threading.get_ident(), node.name)] = (
                _ActiveTiming(
                    wall_start_ns=perf_counter_ns(),
                    cpu_start_ns=thread_time_ns(),
                    rss_start_bytes=rss_start_bytes,
                    peak_rss_start_bytes=peak_rss_start_bytes,
                )
            )

    def finish_node(
        self,
        node: Node,
        *,
        is_async: bool,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        key = (threading.get_ident(), node.name)
        with self._lock:
            started = self._active_node_timings.pop(key, None)
            if started is None:
                return

        wall_seconds, cpu_seconds, cpu_utilization_ratio = _elapsed_metrics(started)
        rss_end_bytes, peak_rss_end_bytes = _capture_memory_snapshot()
        self.node_timings.append(
            _NodeTimingRecord(
                node_name=node.name,
                stage=_node_stage(node),
                tags=tuple(sorted(node.tags)),
                is_async=is_async,
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                cpu_utilization_ratio=cpu_utilization_ratio,
                rss_start_bytes=started.rss_start_bytes,
                rss_end_bytes=rss_end_bytes,
                rss_delta_bytes=_optional_delta(rss_end_bytes, started.rss_start_bytes),
                peak_rss_start_bytes=started.peak_rss_start_bytes,
                peak_rss_end_bytes=peak_rss_end_bytes,
                peak_rss_delta_bytes=_optional_delta(
                    peak_rss_end_bytes, started.peak_rss_start_bytes
                ),
                input_dataset_names=tuple(inputs),
                output_dataset_names=tuple(outputs),
            )
        )
        self.write_checkpoint()

    def start_dataset(self, dataset_name: str, node: Node, *, operation: str) -> None:
        rss_start_bytes, peak_rss_start_bytes = _capture_memory_snapshot()
        with self._lock:
            self._active_dataset_timings[
                (threading.get_ident(), dataset_name, node.name, operation)
            ] = _ActiveTiming(
                wall_start_ns=perf_counter_ns(),
                cpu_start_ns=thread_time_ns(),
                rss_start_bytes=rss_start_bytes,
                peak_rss_start_bytes=peak_rss_start_bytes,
            )

    def finish_dataset(
        self,
        dataset_name: str,
        node: Node,
        *,
        operation: str,
        data: Any,
    ) -> None:
        key = (threading.get_ident(), dataset_name, node.name, operation)
        with self._lock:
            started = self._active_dataset_timings.pop(key, None)
            if started is None:
                return

        wall_seconds, cpu_seconds, cpu_utilization_ratio = _elapsed_metrics(started)
        rss_end_bytes, peak_rss_end_bytes = _capture_memory_snapshot()
        data_type, item_count = _summarize_data(data)
        self.dataset_timings.append(
            _DatasetTimingRecord(
                dataset_name=dataset_name,
                node_name=node.name,
                stage=_node_stage(node),
                operation=operation,
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                cpu_utilization_ratio=cpu_utilization_ratio,
                rss_start_bytes=started.rss_start_bytes,
                rss_end_bytes=rss_end_bytes,
                rss_delta_bytes=_optional_delta(rss_end_bytes, started.rss_start_bytes),
                peak_rss_start_bytes=started.peak_rss_start_bytes,
                peak_rss_end_bytes=peak_rss_end_bytes,
                peak_rss_delta_bytes=_optional_delta(
                    peak_rss_end_bytes, started.peak_rss_start_bytes
                ),
                data_type=data_type,
                item_count=item_count,
            )
        )

    def finalize(self, *, error: str | None = None) -> Path | None:
        with self._lock:
            if self._finalized:
                return None
            self._finalized = True

        report_payload = self._build_report_payload(error=error, partial=False)

        _log_hotspots(report_payload)

        if not self.config.reporting_enabled:
            return None

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        if self.partial_report_path.exists():
            self.partial_report_path.unlink()
        LOGGER.info("Wrote performance profile to '%s'.", self.report_path.as_posix())
        return self.report_path

    def write_checkpoint(self) -> None:
        if not self.config.reporting_enabled:
            return

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.partial_report_path.write_text(
            json.dumps(self._build_report_payload(error=None, partial=True), indent=2)
            + "\n",
            encoding="utf-8",
        )

    def _build_report_payload(
        self,
        *,
        error: str | None,
        partial: bool,
    ) -> dict[str, Any]:
        pipeline_wall_seconds = (
            perf_counter_ns() - self.pipeline_wall_start_ns
        ) / 1_000_000_000
        pipeline_cpu_seconds = (
            thread_time_ns() - self.pipeline_cpu_start_ns
        ) / 1_000_000_000

        return {
            "run": {
                "run_id": self.run_params.get("run_id"),
                "pipeline_name": self.run_params.get("pipeline_name", "__default__"),
                "runner": self.run_params.get("runner"),
                "is_async": bool(self.run_params.get("is_async", False)),
                "started_at_utc": self.started_at_utc.isoformat(),
                "finished_at_utc": None if partial else datetime.now(UTC).isoformat(),
                "pipeline_wall_seconds": pipeline_wall_seconds,
                "pipeline_cpu_seconds": pipeline_cpu_seconds,
                "pipeline_cpu_utilization_ratio": _safe_ratio(
                    pipeline_cpu_seconds, pipeline_wall_seconds
                ),
                "current_rss_bytes": _current_rss_bytes(),
                "peak_rss_bytes": _peak_rss_bytes(),
                "error": error,
                "partial": partial,
            },
            "config": asdict(self.config),
            "wrapped_datasets": sorted(set(self._wrapped_datasets)),
            "released_wrapped_datasets": [
                asdict(record) for record in self.dataset_cache_releases
            ],
            "pipeline_stage_breakdown": _aggregate_stage_timings(self.node_timings),
            "dataset_operation_breakdown": _aggregate_dataset_timings(
                self.dataset_timings
            ),
            "slowest_nodes": _top_records(
                self.node_timings,
                key=lambda record: record.wall_seconds,
                top_n=self.config.top_n,
            ),
            "slowest_dataset_operations": _top_records(
                self.dataset_timings,
                key=lambda record: record.wall_seconds,
                top_n=self.config.top_n,
            ),
            "node_timings": [asdict(record) for record in self.node_timings],
            "dataset_timings": [asdict(record) for record in self.dataset_timings],
        }


class ProjectHooks:
    """Register performance profiling and run-local dataset caching hooks."""

    def __init__(self) -> None:
        self._states_by_run_id: dict[str, _PerformanceRunState] = {}
        self._state_lock = threading.Lock()

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog: Any,
    ) -> None:
        config = _load_performance_config(catalog)
        state = _PerformanceRunState(run_params=run_params, config=config)
        state.initialize_pipeline(pipeline)

        run_id = str(run_params.get("run_id", "unknown-run"))
        with self._state_lock:
            self._states_by_run_id[run_id] = state

        if (
            config.dataset_run_cache_enabled
            and run_params.get("runner") != "ParallelRunner"
        ):
            for dataset_name in config.dataset_run_cache_dataset_names:
                _wrap_catalog_dataset_with_cache(
                    catalog=catalog,
                    dataset_name=dataset_name,
                    copy_mode=config.dataset_run_cache_copy_mode,
                    state=state,
                )
        elif config.dataset_run_cache_enabled:
            LOGGER.warning(
                "Skipping run-local dataset caching because ParallelRunner does not "
                "support CachedDataset wrappers."
            )

    @hook_impl
    def after_pipeline_run(
        self,
        run_params: dict[str, Any],
        run_result: dict[str, Any],
        pipeline: Pipeline,
        catalog: Any,
    ) -> None:
        del run_result, pipeline, catalog
        state = self._pop_state(run_params)
        if state is not None:
            state.finalize()

    @hook_impl
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: dict[str, Any],
        pipeline: Pipeline,
        catalog: Any,
    ) -> None:
        del pipeline, catalog
        state = self._pop_state(run_params)
        if state is not None:
            state.finalize(error=f"{type(error).__name__}: {error}")

    @hook_impl
    def before_node_run(
        self,
        node: Node,
        catalog: Any,
        inputs: dict[str, Any],
        is_async: bool,
        run_id: str,
    ) -> None:
        del catalog, inputs
        state = self._get_state(run_id)
        if state is not None:
            state.start_node(node, is_async=is_async)

    # Kedro hook signatures are fixed and intentionally exceed Ruff's local limit.
    @hook_impl
    def after_node_run(  # noqa: PLR0913
        self,
        node: Node,
        catalog: Any,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        is_async: bool,
        run_id: str,
    ) -> None:
        state = self._get_state(run_id)
        if state is not None:
            state.finish_node(
                node,
                is_async=is_async,
                inputs=inputs,
                outputs=outputs,
            )
            if state.config.dataset_run_cache_release_when_unused:
                for dataset_name in state.releasable_wrapped_datasets(node):
                    _release_catalog_dataset(catalog=catalog, dataset_name=dataset_name)

    @hook_impl
    def on_node_error(  # noqa: PLR0913
        self,
        error: Exception,
        node: Node,
        catalog: Any,
        inputs: dict[str, Any],
        is_async: bool,
        run_id: str,
    ) -> None:
        del error, catalog, inputs, is_async
        state = self._get_state(run_id)
        if state is not None:
            state.finish_node(node, is_async=False, inputs={}, outputs={})

    @hook_impl
    def before_dataset_loaded(self, dataset_name: str, node: Node) -> None:
        state = self._sole_active_state()
        if state is not None:
            state.start_dataset(dataset_name, node, operation="load")

    @hook_impl
    def after_dataset_loaded(self, dataset_name: str, data: Any, node: Node) -> None:
        state = self._sole_active_state()
        if state is not None:
            state.finish_dataset(dataset_name, node, operation="load", data=data)

    @hook_impl
    def before_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None:
        del data
        state = self._sole_active_state()
        if state is not None:
            state.start_dataset(dataset_name, node, operation="save")

    @hook_impl
    def after_dataset_saved(self, dataset_name: str, data: Any, node: Node) -> None:
        state = self._sole_active_state()
        if state is not None:
            state.finish_dataset(dataset_name, node, operation="save", data=data)

    def _get_state(self, run_id: str) -> _PerformanceRunState | None:
        with self._state_lock:
            state = self._states_by_run_id.get(str(run_id))
            if state is not None:
                return state
            if len(self._states_by_run_id) == 1:
                return next(iter(self._states_by_run_id.values()))
            return None

    def _sole_active_state(self) -> _PerformanceRunState | None:
        with self._state_lock:
            if len(self._states_by_run_id) != 1:
                return None
            return next(iter(self._states_by_run_id.values()))

    def _pop_state(self, run_params: dict[str, Any]) -> _PerformanceRunState | None:
        with self._state_lock:
            return self._states_by_run_id.pop(
                str(run_params.get("run_id", "unknown-run")),
                None,
            )


def _load_performance_config(catalog: Any) -> _PerformanceConfig:
    try:
        loaded = catalog.load("params:performance")
    except DatasetNotFoundError:
        return _PerformanceConfig()
    except Exception as exc:  # pragma: no cover - defensive config fallback
        LOGGER.warning("Falling back to default performance config: %s", exc)
        return _PerformanceConfig()

    reporting = loaded.get("reporting", {}) if isinstance(loaded, dict) else {}
    dataset_run_cache = (
        loaded.get("dataset_run_cache", {}) if isinstance(loaded, dict) else {}
    )

    dataset_names = dataset_run_cache.get(
        "dataset_names",
        _PerformanceConfig.dataset_run_cache_dataset_names,
    )

    return _PerformanceConfig(
        reporting_enabled=bool(reporting.get("enabled", True)),
        report_dir=str(reporting.get("report_dir", "data/08_reporting/performance")),
        report_basename=str(reporting.get("report_basename", "pipeline_profile")),
        top_n=int(reporting.get("top_n", 10)),
        dataset_run_cache_enabled=bool(dataset_run_cache.get("enabled", True)),
        dataset_run_cache_copy_mode=str(dataset_run_cache.get("copy_mode", "assign")),
        dataset_run_cache_release_when_unused=bool(
            dataset_run_cache.get("release_when_unused", True)
        ),
        dataset_run_cache_dataset_names=tuple(str(name) for name in dataset_names),
    )


def _wrap_catalog_dataset_with_cache(
    *,
    catalog: Any,
    dataset_name: str,
    copy_mode: str,
    state: _PerformanceRunState,
) -> None:
    try:
        dataset = catalog.get(dataset_name)
    except DatasetNotFoundError:
        LOGGER.warning("Cannot cache unknown dataset '%s'.", dataset_name)
        return
    if dataset is None:
        LOGGER.warning(
            "Skipping run-local cache for dataset '%s' because catalog.get() "
            "returned None.",
            dataset_name,
        )
        return

    if isinstance(dataset, CachedDataset):
        return

    catalog._datasets[dataset_name] = CachedDataset(
        dataset=dataset, copy_mode=copy_mode
    )
    catalog._lazy_datasets.pop(dataset_name, None)
    state.record_wrapped_dataset(dataset_name)


def _release_catalog_dataset(*, catalog: Any, dataset_name: str) -> None:
    try:
        catalog.release(dataset_name)
        LOGGER.info(
            "Released cached dataset '%s' after its final consumer completed.",
            dataset_name,
        )
    except DatasetNotFoundError:
        LOGGER.warning("Cannot release unknown dataset '%s'.", dataset_name)
    except Exception as exc:  # pragma: no cover - defensive hook fallback
        LOGGER.warning("Failed to release dataset '%s': %s", dataset_name, exc)


def _elapsed_metrics(started: _ActiveTiming) -> tuple[float, float, float]:
    wall_seconds = (perf_counter_ns() - started.wall_start_ns) / 1_000_000_000
    cpu_seconds = (thread_time_ns() - started.cpu_start_ns) / 1_000_000_000
    return wall_seconds, cpu_seconds, _safe_ratio(cpu_seconds, wall_seconds)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _optional_delta(
    end_value: int | None,
    start_value: int | None,
) -> int | None:
    if end_value is None or start_value is None:
        return None
    return end_value - start_value


def _capture_memory_snapshot() -> tuple[int | None, int | None]:
    return _current_rss_bytes(), _peak_rss_bytes()


def _current_rss_bytes() -> int | None:
    statm_path = Path("/proc/self/statm")
    try:
        resident_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        return None

    return resident_pages * resource.getpagesize()


def _peak_rss_bytes() -> int | None:
    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None

    if sys.platform == "darwin":
        return int(max_rss)
    return int(max_rss) * 1024


def _node_stage(node: Node) -> str:
    for tag in _KNOWN_PIPELINE_TAGS:
        if tag in node.tags:
            return tag
    return "uncategorized"


def _summarize_data(data: Any) -> tuple[str, int | None]:
    data_type = type(data).__name__
    item_count: int | None = None
    if not isinstance(data, str | bytes | bytearray):
        try:
            item_count = len(data)  # type: ignore[arg-type]
        except TypeError:
            item_count = None

    return data_type, item_count


def _aggregate_stage_timings(
    node_timings: list[_NodeTimingRecord],
) -> list[dict[str, Any]]:
    by_stage: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"wall_seconds": 0.0, "cpu_seconds": 0.0, "node_count": 0}
    )
    for record in node_timings:
        stage = by_stage[record.stage]
        stage["wall_seconds"] += record.wall_seconds
        stage["cpu_seconds"] += record.cpu_seconds
        stage["node_count"] += 1

    return [
        {
            "stage": stage_name,
            "wall_seconds": values["wall_seconds"],
            "cpu_seconds": values["cpu_seconds"],
            "cpu_utilization_ratio": _safe_ratio(
                float(values["cpu_seconds"]),
                float(values["wall_seconds"]),
            ),
            "node_count": values["node_count"],
        }
        for stage_name, values in sorted(
            by_stage.items(),
            key=lambda item: float(item[1]["wall_seconds"]),
            reverse=True,
        )
    ]


def _aggregate_dataset_timings(
    dataset_timings: list[_DatasetTimingRecord],
) -> list[dict[str, Any]]:
    by_dataset: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"wall_seconds": 0.0, "cpu_seconds": 0.0, "count": 0}
    )
    for record in dataset_timings:
        totals = by_dataset[(record.dataset_name, record.operation)]
        totals["wall_seconds"] += record.wall_seconds
        totals["cpu_seconds"] += record.cpu_seconds
        totals["count"] += 1

    return [
        {
            "dataset_name": dataset_name,
            "operation": operation,
            "wall_seconds": values["wall_seconds"],
            "cpu_seconds": values["cpu_seconds"],
            "cpu_utilization_ratio": _safe_ratio(
                float(values["cpu_seconds"]),
                float(values["wall_seconds"]),
            ),
            "count": values["count"],
        }
        for (dataset_name, operation), values in sorted(
            by_dataset.items(),
            key=lambda item: float(item[1]["wall_seconds"]),
            reverse=True,
        )
    ]


def _top_records(
    records: list[_NodeTimingRecord] | list[_DatasetTimingRecord],
    *,
    key: Any,
    top_n: int,
) -> list[dict[str, Any]]:
    return [asdict(record) for record in sorted(records, key=key, reverse=True)[:top_n]]


def _log_hotspots(report_payload: dict[str, Any]) -> None:
    slowest_nodes = report_payload["slowest_nodes"]
    slowest_dataset_operations = report_payload["slowest_dataset_operations"]
    if slowest_nodes:
        LOGGER.info(
            "Slowest nodes: %s",
            ", ".join(
                f"{record['node_name']}={record['wall_seconds']:.2f}s"
                for record in slowest_nodes[:3]
            ),
        )
    if slowest_dataset_operations:
        LOGGER.info(
            "Slowest dataset operations: %s",
            ", ".join(
                (
                    f"{record['operation']}:{record['dataset_name']}="
                    f"{record['wall_seconds']:.2f}s"
                )
                for record in slowest_dataset_operations[:3]
            ),
        )
