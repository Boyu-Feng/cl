"""Multi-run benchmark orchestration.

``run_benchmark_runs`` runs N rollouts only.  ``run_benchmark`` is the
unified entry point that runs the baseline and the rollouts together —
when both task and system are ``parallel_safe`` it submits everything to
a single ``ProcessPoolExecutor`` so baseline instances and rollout runs
overlap in wall-clock time.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..errors import ProviderRefusalError
from ..interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    TaskResult,
    TaskResults,
)
from ..logging_utils import bind_logging_context
from ..run_ids import new_timestamp_run_id
from ..trace_storage import _atomic_write_json
from .common import (
    BaselineSuccess,
    RunMode,
    RunSuccess,
    drain_run_futures,
    get_task_num_instances,
    record_baseline_outcome,
    record_run_outcome,
    task_accepts_parallel_execution,
)
from .baseline import (
    _merge_baseline_instance_results,
    _run_baseline_instance,
    run_baseline,
)
from .single import run_single

logger = logging.getLogger(__name__)


def derive_run_task_params(
    task_params: dict[str, Any],
    run_mode: RunMode,
    run_index: int,
) -> dict[str, Any]:
    """Return per-run task params per *run_mode*."""
    run_task_params = {**task_params}
    if run_mode is RunMode.REPLICATE:
        return run_task_params
    if run_mode in (RunMode.RESAMPLE, RunMode.PERMUTE):
        run_task_params["run_index"] = run_index
        run_task_params["rollout_index"] = run_index
    return run_task_params


def run_benchmark_runs(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    runs: int,
    max_workers: int,
    system_name: str,
    task_name: str,
    run_group_id: Optional[str] = None,
    output_path: Optional[Path] = None,
    run_mode: RunMode = RunMode.PERMUTE,
    verbose_runs: bool = False,
    live_trace_paths: Optional[dict[int, Path]] = None,
) -> tuple[TaskResults, list[dict[str, Any]]]:
    """Orchestrate one or more independent benchmark runs, optionally in parallel."""
    if run_group_id is None:
        run_group_id = new_timestamp_run_id()

    can_parallelize = task_accepts_parallel_execution(task_class, system_class)
    run_results: list[Optional[RunSuccess]] = [None] * runs
    effective_workers = min(max_workers, runs) if can_parallelize else 1

    print(
        f"Starting {runs} run(s) with max workers {effective_workers} "
        f"[group {run_group_id}, mode={run_mode.value}]"
    )
    logger.info(
        "runs.started",
        extra={
            "run_group_id": run_group_id,
            "task": task_name,
            "system": system_name,
            "runs": runs,
            "max_workers": max_workers,
            "effective_workers": effective_workers,
            "parallel_safe": can_parallelize,
            "run_mode": run_mode.value,
        },
    )

    if runs == 1 or effective_workers == 1:
        for i in range(runs):
            try:
                with bind_logging_context(run_index=i, phase="run"):
                    logger.info("run.submitted")
                    outcome: RunSuccess | None = run_single(
                        task_class=task_class,
                        task_params=derive_run_task_params(task_params, run_mode, i),
                        system_class=system_class,
                        system_params=system_params,
                        run_index=i,
                        run_group_id=run_group_id,
                        system_name=system_name,
                        task_name=task_name,
                        output_path=output_path,
                        show_progress=(runs == 1),
                        verbose_runs=verbose_runs,
                        live_trace_path=None
                        if live_trace_paths is None
                        else live_trace_paths.get(i),
                    )
                refusal: ProviderRefusalError | None = None
            except ProviderRefusalError as exc:
                outcome, refusal = None, exc
            except Exception as exc:
                logger.error(
                    "Run %d/%d failed: %s",
                    i + 1,
                    runs,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(f"Run {i + 1}/{runs} failed: {exc}") from exc
            record_run_outcome(
                i,
                outcome,
                refusal,
                runs=runs,
                run_results=run_results,
                run_group_id=run_group_id,
                live_trace_path=(
                    None if live_trace_paths is None else live_trace_paths.get(i)
                ),
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            logger.info(
                "run_pool.started",
                extra={"effective_workers": effective_workers},
            )
            futures = {
                pool.submit(
                    run_single,
                    task_class=task_class,
                    task_params=derive_run_task_params(task_params, run_mode, i),
                    system_class=system_class,
                    system_params=system_params,
                    run_index=i,
                    run_group_id=run_group_id,
                    system_name=system_name,
                    task_name=task_name,
                    verbose_runs=verbose_runs,
                    live_trace_path=(
                        None if live_trace_paths is None else live_trace_paths.get(i)
                    ),
                ): i
                for i in range(runs)
            }

            if verbose_runs:
                for i in range(runs):
                    print(f"  Run {i + 1}/{runs} starting")

            drain_run_futures(
                futures,
                runs=runs,
                run_results=run_results,
                run_group_id=run_group_id,
                live_trace_paths=live_trace_paths,
            )

    task_results_list: list[TaskResult] = []
    trace_data_list: list[dict[str, Any]] = []
    execution_summaries: list[dict[str, Any]] = []
    for entry in run_results:
        assert entry is not None
        _, result, trace_data, execution_summary = entry
        task_results_list.append(result)
        trace_data_list.append(trace_data)
        execution_summaries.append(execution_summary)

    task_results = TaskResults(
        run_group_id=run_group_id,
        results=task_results_list,
        execution_summaries=execution_summaries,
    )
    logger.info(
        "runs.finished",
        extra={"runs": len(task_results_list), "run_group_id": run_group_id},
    )

    return task_results, trace_data_list


def run_benchmark(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    runs: int,
    max_workers: int,
    system_name: str,
    task_name: str,
    run_group_id: Optional[str] = None,
    run_mode: "RunMode" = None,  # type: ignore[assignment]
    verbose_runs: bool = False,
    include_baseline: bool = True,
    baseline_task_params: Optional[dict[str, Any]] = None,
    baseline_live_path: Optional[Path] = None,
    live_trace_paths: Optional[dict[int, Path]] = None,
    output_path: Optional[Path] = None,
) -> tuple[
    Optional[tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]],
    TaskResults,
    list[dict[str, Any]],
]:
    """Unified benchmark entry point: baseline (optional) + rollout runs.

    When both classes are ``parallel_safe`` the baseline instances and all
    rollout runs are submitted to a single ``ProcessPoolExecutor`` so they
    overlap in wall-clock time.  Otherwise the baseline runs first (sequential
    or parallel-instances depending on task support) and the rollout runs
    follow.
    """
    if run_mode is None:
        run_mode = RunMode.PERMUTE
    if run_group_id is None:
        run_group_id = new_timestamp_run_id()

    can_parallelize = task_accepts_parallel_execution(task_class, system_class)
    logger.info(
        "started",
        extra={
            "run_group_id": run_group_id,
            "task": task_name,
            "system": system_name,
            "runs": runs,
            "max_workers": max_workers,
            "include_baseline": include_baseline,
            "parallel_safe": can_parallelize,
            "run_mode": run_mode.value,
        },
    )

    # ── Sequential baseline then parallel runs (safe fallback) ──────────────
    if not include_baseline or not can_parallelize:
        baseline_info = None
        if include_baseline:
            logger.info("phase.baseline.started")
            baseline_info = run_baseline(
                task_class=task_class,
                task_params=task_params,
                system_class=system_class,
                system_params=system_params,
                max_workers=max_workers,
                run_group_id=run_group_id,
                system_name=system_name,
                task_name=task_name,
                baseline_task_params=baseline_task_params,
                live_trace_path=baseline_live_path,
                verbose=verbose_runs,
            )

        logger.info("phase.rollout.started")
        task_results, run_trace_data = run_benchmark_runs(
            task_class=task_class,
            task_params=task_params,
            system_class=system_class,
            system_params=system_params,
            runs=runs,
            max_workers=max_workers,
            system_name=system_name,
            task_name=task_name,
            run_group_id=run_group_id,
            run_mode=run_mode,
            verbose_runs=verbose_runs,
            live_trace_paths=live_trace_paths,
            output_path=output_path,
        )
        logger.info("finished", extra={"path": "sequential"})
        return baseline_info, task_results, run_trace_data

    # ── Fully parallel: baseline instances + rollout runs in one pool ───────
    merged_baseline_params = {**task_params, **(baseline_task_params or {})}
    num_baseline_instances = get_task_num_instances(task_class, merged_baseline_params)
    total_jobs = num_baseline_instances + runs
    effective_workers = min(max_workers, total_jobs)
    print(
        f"Starting {runs} run(s) + {num_baseline_instances}-instance baseline "
        f"with {effective_workers} parallel workers "
        f"[group {run_group_id}, mode={run_mode.value}]"
    )
    logger.info(
        "pool.started",
        extra={
            "runs": runs,
            "baseline_instances": num_baseline_instances,
            "effective_workers": effective_workers,
        },
    )

    baseline_instance_results: list[Optional[BaselineSuccess]] = [
        None
    ] * num_baseline_instances
    blocked_baseline_instances: list[dict[str, Any]] = []
    run_results: list[Optional[RunSuccess]] = [None] * runs

    baseline_start_time = datetime.now().isoformat()

    def _write_partial_baseline_snapshot(
        completed_results: list[BaselineSuccess],
    ) -> None:
        if baseline_live_path is None:
            return
        _, partial_trace, _ = _merge_baseline_instance_results(
            completed_results,
            run_group_id=run_group_id,
            system_name=system_name,
            task_name=task_name,
            task_params=merged_baseline_params,
            system_params=system_params,
            start_time=baseline_start_time,
            end_time=datetime.now().isoformat(),
            blocked_instances=blocked_baseline_instances,
            expected_num_instances=num_baseline_instances,
        )
        _atomic_write_json(baseline_live_path, partial_trace)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
        # Submit rollout runs first so they immediately claim workers — they
        # are the sequential bottleneck, so minimising their queue time matters
        # most.
        run_futures = {
            pool.submit(
                run_single,
                task_class=task_class,
                task_params=derive_run_task_params(task_params, run_mode, i),
                system_class=system_class,
                system_params=system_params,
                run_index=i,
                run_group_id=run_group_id,
                system_name=system_name,
                task_name=task_name,
                verbose_runs=verbose_runs,
                live_trace_path=(
                    None if live_trace_paths is None else live_trace_paths.get(i)
                ),
            ): i
            for i in range(runs)
        }
        # Submit baseline instances after; they are fully parallel and can fill
        # whatever worker slots remain once the rollouts have started.
        baseline_futures = {
            pool.submit(
                _run_baseline_instance,
                task_class=task_class,
                task_params=merged_baseline_params,
                system_class=system_class,
                system_params=system_params,
                instance_index=i,
                run_group_id=run_group_id,
                system_name=system_name,
                task_name=task_name,
                verbose=verbose_runs,
            ): i
            for i in range(num_baseline_instances)
        }

        all_futures: dict[Any, tuple[str, int]] = {
            **{f: ("baseline", i) for f, i in baseline_futures.items()},
            **{f: ("run", i) for f, i in run_futures.items()},
        }

        completed_baseline_so_far: list[BaselineSuccess] = []
        for future in as_completed(all_futures):
            kind, idx = all_futures[future]
            try:
                payload = future.result()
                refusal: ProviderRefusalError | None = None
            except ProviderRefusalError as exc:
                payload, refusal = None, exc
            except Exception as exc:
                label = (
                    f"Baseline instance {idx + 1}/{num_baseline_instances}"
                    if kind == "baseline"
                    else f"Run {idx + 1}/{runs}"
                )
                logger.error("%s failed: %s", label, exc, exc_info=True)
                raise RuntimeError(f"{label} failed: {exc}") from exc

            if kind == "baseline":
                record_baseline_outcome(
                    idx,
                    payload,  # type: ignore[arg-type]
                    refusal,
                    num_instances=num_baseline_instances,
                    instance_results=baseline_instance_results,
                    blocked_instances=blocked_baseline_instances,
                    completed_so_far=completed_baseline_so_far,
                    on_progress=_write_partial_baseline_snapshot,
                )
            else:
                record_run_outcome(
                    idx,
                    payload,  # type: ignore[arg-type]
                    refusal,
                    runs=runs,
                    run_results=run_results,
                    run_group_id=run_group_id,
                    live_trace_path=(
                        None if live_trace_paths is None else live_trace_paths.get(idx)
                    ),
                )

    baseline_end_time = datetime.now().isoformat()

    # Merge baseline instances.
    completed_baseline = [r for r in baseline_instance_results if r is not None]
    merged_result, baseline_trace_data, baseline_execution = (
        _merge_baseline_instance_results(
            completed_baseline,
            run_group_id=run_group_id,
            system_name=system_name,
            task_name=task_name,
            task_params=merged_baseline_params,
            system_params=system_params,
            start_time=baseline_start_time,
            end_time=baseline_end_time,
            blocked_instances=blocked_baseline_instances,
            expected_num_instances=num_baseline_instances,
        )
    )
    baseline_info = (None, merged_result, baseline_trace_data, baseline_execution)

    # Write the completed baseline trace so the live viewer can display it.
    if baseline_live_path is not None:
        _atomic_write_json(baseline_live_path, baseline_trace_data)
        logger.debug(
            "baseline.live_trace.written",
            extra={"path": str(baseline_live_path)},
        )

    # Aggregate rollout runs.
    task_results_list: list[TaskResult] = []
    run_trace_data_list: list[dict[str, Any]] = []
    execution_summaries: list[dict[str, Any]] = []
    for entry in run_results:
        assert entry is not None
        _, result, trace_data, execution_summary = entry
        task_results_list.append(result)
        run_trace_data_list.append(trace_data)
        execution_summaries.append(execution_summary)

    task_results = TaskResults(
        run_group_id=run_group_id,
        results=task_results_list,
        execution_summaries=execution_summaries,
    )
    logger.info(
        "finished",
        extra={"path": "parallel", "runs": len(task_results_list)},
    )

    return baseline_info, task_results, run_trace_data_list
