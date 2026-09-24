"""Run orchestration for repeated benchmark evaluations.

This package replaces the older ``src/runs.py`` module.  Public callers
should keep importing from ``src.runs``; the per-file split exists for
internal organisation only.

Layout:
* ``_common`` – ``RunMode``, shared helpers, and per-result handlers used
  by every drain loop in the package.
* ``single`` – ``run_single`` (one independent rollout, fail-soft on refusal).
* ``baseline`` – per-instance baseline worker, merge logic, and runner.
* ``benchmark`` – multi-run rollouts and the unified baseline+rollout entry
  point that overlaps everything in a single process pool.
"""

from concurrent.futures import ProcessPoolExecutor  # re-exported for test patches

from .common import (
    RunMode,
    build_blocked_run_stub,
    build_blocked_task_result,
    filter_init_params,
    get_task_num_instances,
    serialize_blocked_baseline_instance,
    task_accepts_parallel_execution,
)
from .baseline import (
    _merge_baseline_instance_results,
    _run_baseline_instance,
    run_baseline,
)
from .benchmark import (
    derive_run_task_params,
    run_benchmark,
    run_benchmark_runs,
)
from .single import run_single

__all__ = [
    "ProcessPoolExecutor",
    "RunMode",
    "derive_run_task_params",
    "run_baseline",
    "run_benchmark",
    "run_benchmark_runs",
    "run_single",
    # Internal helpers re-exported for tests / deep callers.
    "build_blocked_run_stub",
    "build_blocked_task_result",
    "filter_init_params",
    "get_task_num_instances",
    "_merge_baseline_instance_results",
    "_run_baseline_instance",
    "serialize_blocked_baseline_instance",
    "task_accepts_parallel_execution",
]
