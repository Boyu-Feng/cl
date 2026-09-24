"""Baseline orchestration: per-instance worker, merge, and runner."""

from __future__ import annotations

import logging
import multiprocessing as mp
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..artifacts import build_baseline_system_artifacts
from ..errors import ProviderRefusalError
from ..interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    EvalMetrics,
    InstanceOutcome,
    TaskResult,
    run_task,
    serialize_instance_outcome,
)
from ..trace_metrics import attach_interaction_metrics_to_outcomes
from ..trace_storage import TraceRecorder, _atomic_write_json
from ..usage import summarize_usage_events
from .common import (
    BaselineSuccess,
    drain_baseline_futures,
    filter_init_params,
    get_task_num_instances,
    record_baseline_outcome,
    task_accepts_parallel_execution,
)
from .single import run_single

logger = logging.getLogger(__name__)


def _run_baseline_instance(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    instance_index: int,
    run_group_id: str,
    system_name: str,
    task_name: str,
    verbose: bool = False,
) -> tuple[int, TaskResult, dict[str, Any]]:
    """Worker: run a single baseline instance in its own isolated process.

    Each call constructs a fresh task and system.  The task is sliced to only
    the instance at *instance_index* before execution so no reset is needed.
    """
    filtered_params = filter_init_params(task_class, task_params)
    task = task_class(**filtered_params)
    initial_query = task.reset_baseline_instance(instance_index)
    system = system_class(**system_params)
    task_brief = task.get_agent_brief()

    trace_task_params = dict(filtered_params)
    expected_instances = getattr(task, "num_instances", None)
    if isinstance(expected_instances, int) and expected_instances > 0:
        trace_task_params["expected_num_instances"] = expected_instances

    trace_recorder = TraceRecorder(
        system_name=system_name,
        task_name=task_name,
        system_params=system_params,
        task_params=trace_task_params,
        run_group_id=run_group_id,
        run_index=instance_index,
        task_brief=task_brief,
        phase="baseline",
        live_trace_path=None,  # main process merges live snapshots on completion
    )

    try:
        result = run_task(
            task=task,
            system=system,
            trace_recorder=trace_recorder,
            show_progress=False,
            verbose_logging=verbose,
            reset_between_instances=False,
            phase="baseline",
            initial_query=initial_query,
        )
    except ProviderRefusalError as exc:
        raise exc.with_instance(
            instance_id=initial_query.instance_id,
            instance_index=initial_query.instance_index,
        ) from exc

    trace_data = trace_recorder.finalize(result)
    return instance_index, result, trace_data


def _merge_baseline_instance_results(
    instance_results: list[BaselineSuccess],
    run_group_id: str,
    system_name: str,
    task_name: str,
    task_params: dict[str, Any],
    start_time: str,
    end_time: str,
    system_params: Optional[dict[str, Any]] = None,
    blocked_instances: Optional[list[dict[str, Any]]] = None,
    expected_num_instances: Optional[int] = None,
) -> tuple[TaskResult, dict[str, Any], dict[str, Any]]:
    """Merge N single-instance TaskResults into a combined baseline result.

    Score is the mean of per-instance scores.  Loss curves are concatenated in
    instance order.  Instance outcomes are sorted by instance_index.
    """
    sorted_results = sorted(instance_results, key=lambda x: x[0])
    task_results = [result for _, result, _ in sorted_results]
    trace_payloads = [trace for _, _, trace in sorted_results]
    sorted_blocked_instances = sorted(
        blocked_instances or [],
        key=lambda payload: (
            payload.get("instance_index")
            if isinstance(payload.get("instance_index"), int)
            else float("inf")
        ),
    )

    if task_results:
        merged_loss_curve = [v for r in task_results for v in r.eval_metrics.loss_curve]
        merged_optimal = statistics.mean(
            r.eval_metrics.optimal_performance for r in task_results
        )
        merged_actual = statistics.mean(
            r.eval_metrics.actual_performance for r in task_results
        )
    else:
        merged_loss_curve = []
        merged_optimal = 0.0
        merged_actual = 0.0
    merged_outcomes: list[InstanceOutcome] = sorted(
        (o for r in task_results for o in r.instance_outcomes),
        key=lambda o: o.instance_index,
    )
    serialized_outcomes = [serialize_instance_outcome(o) for o in merged_outcomes]

    # Merge extra eval_metrics fields if present.
    extra: Optional[dict[str, float]] = None
    extra_keys: set[str] = set()
    for r in task_results:
        if r.eval_metrics.extra:
            extra_keys.update(r.eval_metrics.extra.keys())
    if extra_keys:
        extra = {
            k: statistics.mean(
                r.eval_metrics.extra[k]
                for r in task_results
                if r.eval_metrics.extra and k in r.eval_metrics.extra
            )
            for k in extra_keys
        }

    total_instances = expected_num_instances
    if total_instances is None:
        total_instances = len(serialized_outcomes) + len(sorted_blocked_instances)
    instances_completed = len(serialized_outcomes)
    instances_blocked = len(sorted_blocked_instances)
    coverage_ratio = (
        round(instances_completed / total_instances, 6) if total_instances else 0.0
    )
    status = "completed_with_gaps" if sorted_blocked_instances else "completed"
    blocked_clause = f", {instances_blocked} blocked" if instances_blocked else ""

    mean_reward = (
        statistics.mean(o.reward for o in merged_outcomes) if merged_outcomes else 0.0
    )
    merged_result = TaskResult(
        metrics={},
        summary=(
            f"Baseline: {instances_completed}/{total_instances} completed"
            f"{blocked_clause}, mean reward {mean_reward:.4f}"
        ),
        eval_metrics=EvalMetrics(
            loss_curve=merged_loss_curve,
            optimal_performance=merged_optimal,
            actual_performance=merged_actual,
            extra=extra,
        ),
        instance_outcomes=merged_outcomes,
    )

    merged_interactions = [
        interaction
        for trace in trace_payloads
        for interaction in trace.get("interactions", [])
    ]
    response_latencies: list[float] = []
    respond_usage_events: list[dict[str, Any]] = []
    observe_usage_events: list[dict[str, Any]] = []
    for interaction in merged_interactions:
        timing = interaction.get("timing", {})
        if isinstance(timing, dict):
            latency = timing.get("response_latency_seconds")
            if isinstance(latency, (int, float)):
                response_latencies.append(float(latency))
        usage = interaction.get("usage", {})
        if not isinstance(usage, dict):
            continue
        respond_usage_events.extend(usage.get("respond", {}).get("events", []) or [])
        observe_usage_events.extend(usage.get("observe", {}).get("events", []) or [])
    attach_interaction_metrics_to_outcomes(serialized_outcomes, merged_interactions)
    task_brief = next(
        (
            trace.get("task_brief")
            for trace in trace_payloads
            if trace.get("task_brief")
        ),
        None,
    )
    system_payload = (
        trace_payloads[0].get("system")
        if trace_payloads
        else {"name": system_name, "params": system_params or {}}
    )
    merged_artifacts = {
        "instances": [trace.get("artifacts", {}) for trace in trace_payloads]
    }
    baseline_system_artifacts = build_baseline_system_artifacts(trace_payloads)
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    wall_duration_seconds = (end_dt - start_dt).total_seconds()
    total_response_seconds = sum(response_latencies)
    avg_response_seconds = (
        total_response_seconds / len(response_latencies) if response_latencies else 0.0
    )
    max_response_seconds = max(response_latencies) if response_latencies else 0.0
    interaction_usage_summary = summarize_usage_events(
        [*respond_usage_events, *observe_usage_events]
    )

    execution_summary: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "total_interactions": len(merged_interactions),
        "wall_duration_seconds": round(wall_duration_seconds, 6),
        "total_response_seconds": round(total_response_seconds, 6),
        "avg_response_seconds": round(avg_response_seconds, 6),
        "max_response_seconds": round(max_response_seconds, 6),
        "run_group_id": run_group_id,
        "run_index": None,
        "status": status,
        "phase": "baseline",
        "task_brief": task_brief,
        "instances_total": total_instances,
        "instances_completed": instances_completed,
        "instances_blocked": instances_blocked,
        "coverage_ratio": coverage_ratio,
        "blocked_instances": sorted_blocked_instances,
        "usage": {
            "respond": summarize_usage_events(respond_usage_events),
            "observe": summarize_usage_events(observe_usage_events),
            "interaction": interaction_usage_summary,
        },
        "instance_outcomes": serialized_outcomes,
    }

    trace_data: dict[str, Any] = {
        "status": status,
        "phase": "baseline",
        "schedule": task_params.get("schedule", task_params.get("rollout_schedule", ""))
        or "",
        "task_brief": task_brief,
        "system": system_payload,
        "task": {"name": task_name, "params": task_params},
        "execution": execution_summary,
        "artifacts": merged_artifacts,
        "system_artifacts": baseline_system_artifacts,
        "interactions": merged_interactions,
        "instance_outcomes": serialized_outcomes,
        "blocked_instances": sorted_blocked_instances,
        "instance_traces": trace_payloads,
        "result": {
            "score": merged_result.score,
            "metrics": {},
            "summary": merged_result.summary,
            "eval_metrics": {
                "loss_curve": merged_loss_curve,
                "optimal_performance": merged_optimal,
                "actual_performance": merged_actual,
                "extra": extra,
            },
            "instance_outcomes": serialized_outcomes,
        },
    }

    return merged_result, trace_data, execution_summary


def run_baseline(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    max_workers: int,
    run_group_id: str,
    system_name: str,
    task_name: str,
    baseline_task_params: Optional[dict[str, Any]] = None,
    live_trace_path: Optional[Path] = None,
    verbose: bool = False,
) -> tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]:
    """Run the stateless baseline, parallelising instances when safe.

    When both *task_class* and *system_class* are ``parallel_safe`` the baseline
    instances are dispatched into a ``ProcessPoolExecutor`` — one worker per
    instance, each with its own fresh task and system.  Otherwise, the same
    per-instance baseline path runs sequentially in the main process.

    Returns the same 4-tuple as ``run_single``.
    """
    merged_params = {**task_params, **(baseline_task_params or {})}
    can_parallelize = task_accepts_parallel_execution(task_class, system_class)
    try:
        num_instances = get_task_num_instances(task_class, merged_params)
    except Exception:
        if can_parallelize:
            raise
        logger.debug(
            "Sequential baseline fallback: %s does not expose num_instances during __init__.",
            task_class.__name__,
        )
        return run_single(
            task_class=task_class,
            task_params=merged_params,
            system_class=system_class,
            system_params=system_params,
            run_index=None,
            run_group_id=run_group_id,
            system_name=system_name,
            task_name=task_name,
            show_progress=True,
            verbose_runs=verbose,
            reset_between_instances=True,
            phase="baseline",
            live_trace_path=live_trace_path,
        )

    effective_workers = min(max_workers, num_instances) if can_parallelize else 1
    print(
        f"Starting baseline ({num_instances} instances) with {effective_workers} "
        f"{'parallel workers' if can_parallelize else 'worker'} [group {run_group_id}]"
    )
    logger.info(
        "started",
        extra={
            "run_group_id": run_group_id,
            "task": task_name,
            "system": system_name,
            "instances": num_instances,
            "max_workers": max_workers,
            "effective_workers": effective_workers,
            "parallel_safe": can_parallelize,
        },
    )

    start_time = datetime.now().isoformat()
    instance_results: list[Optional[BaselineSuccess]] = [None] * num_instances
    blocked_instances: list[dict[str, Any]] = []

    def _write_partial_baseline_snapshot(
        completed_results: list[BaselineSuccess],
    ) -> None:
        if live_trace_path is None:
            return
        _, partial_trace, _ = _merge_baseline_instance_results(
            completed_results,
            run_group_id=run_group_id,
            system_name=system_name,
            task_name=task_name,
            task_params=merged_params,
            system_params=system_params,
            start_time=start_time,
            end_time=datetime.now().isoformat(),
            blocked_instances=blocked_instances,
            expected_num_instances=num_instances,
        )
        _atomic_write_json(live_trace_path, partial_trace)
        logger.debug(
            "snapshot.partial.written",
            extra={
                "path": str(live_trace_path),
                "completed_instances": len(completed_results),
                "blocked_instances": len(blocked_instances),
            },
        )

    # Write an empty snapshot immediately so the viewer can show the baseline
    # bar as "in progress" before any instances complete.
    _write_partial_baseline_snapshot([])

    if not can_parallelize:
        logger.debug(
            "Sequential baseline: %s or %s is not parallel_safe.",
            system_class.__name__,
            task_class.__name__,
        )
        completed_so_far: list[BaselineSuccess] = []
        for i in range(num_instances):
            try:
                logger.info(
                    "instance.started",
                    extra={"instance_index": i},
                )
                outcome = _run_baseline_instance(
                    task_class=task_class,
                    task_params=merged_params,
                    system_class=system_class,
                    system_params=system_params,
                    instance_index=i,
                    run_group_id=run_group_id,
                    system_name=system_name,
                    task_name=task_name,
                    verbose=verbose,
                )
                refusal: ProviderRefusalError | None = None
            except ProviderRefusalError as exc:
                outcome = None
                refusal = exc
            except Exception as exc:
                logger.error(
                    "Baseline instance %d/%d failed: %s",
                    i + 1,
                    num_instances,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Baseline instance {i + 1}/{num_instances} failed: {exc}"
                ) from exc
            record_baseline_outcome(
                i,
                outcome,
                refusal,
                num_instances=num_instances,
                instance_results=instance_results,
                blocked_instances=blocked_instances,
                completed_so_far=completed_so_far,
                on_progress=_write_partial_baseline_snapshot,
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _run_baseline_instance,
                    task_class=task_class,
                    task_params=merged_params,
                    system_class=system_class,
                    system_params=system_params,
                    instance_index=i,
                    run_group_id=run_group_id,
                    system_name=system_name,
                    task_name=task_name,
                    verbose=verbose,
                ): i
                for i in range(num_instances)
            }
            drain_baseline_futures(
                futures,
                num_instances=num_instances,
                instance_results=instance_results,
                blocked_instances=blocked_instances,
                on_progress=_write_partial_baseline_snapshot,
            )

    end_time = datetime.now().isoformat()
    completed = [r for r in instance_results if r is not None]
    merged_result, trace_data, execution_summary = _merge_baseline_instance_results(
        completed,
        run_group_id=run_group_id,
        system_name=system_name,
        task_name=task_name,
        task_params=merged_params,
        system_params=system_params,
        start_time=start_time,
        end_time=end_time,
        blocked_instances=blocked_instances,
        expected_num_instances=num_instances,
    )

    # Final write with correct end_time (incremental writes use approximate timestamps).
    if live_trace_path is not None:
        _atomic_write_json(live_trace_path, trace_data)
        logger.debug(
            "live_trace.written",
            extra={"path": str(live_trace_path)},
        )

    logger.info(
        "finished",
        extra={
            "completed_instances": len(completed),
            "blocked_instances": len(blocked_instances),
            "status": execution_summary.get("status"),
        },
    )
    return None, merged_result, trace_data, execution_summary
