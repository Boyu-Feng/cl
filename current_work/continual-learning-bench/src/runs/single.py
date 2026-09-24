"""Single-run orchestration.

``run_single`` constructs a fresh task and system, runs the interaction
loop via :func:`run_task`, and finalises a :class:`TraceRecorder`.  Rollout
refusals are caught here and converted into a ``status="blocked"`` partial
trace so the parent benchmark loop can record the run without aborting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import json

from ..artifacts import build_artifact_manifest, save_artifacts
from ..errors import ProviderRefusalError
from ..interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    TaskResult,
    run_task,
)
from ..logging_utils import bind_logging_context
from ..trace_storage import TraceRecorder, build_trace_output_path
from .common import build_blocked_task_result, filter_init_params

logger = logging.getLogger(__name__)


def run_single(
    task_class: type[ContinualLearningTask],
    task_params: dict[str, Any],
    system_class: type[ContinualLearningSystem],
    system_params: dict[str, Any],
    run_index: Optional[int],
    run_group_id: str,
    system_name: str,
    task_name: str,
    output_path: Optional[Path] = None,
    show_progress: bool = False,
    verbose_runs: bool = False,
    reset_between_instances: bool = False,
    phase: str = "run",
    live_trace_path: Optional[Path] = None,
    trace_kind: str = "trace",
) -> tuple[Optional[int], TaskResult, dict[str, Any], dict[str, Any]]:
    """Execute a single independent run with fresh task and system instances."""
    human_run_number = None if run_index is None else run_index + 1

    try:
        with bind_logging_context(
            run_group_id=run_group_id,
            task=task_name,
            system=system_name,
            run_index=run_index,
            phase=phase,
        ):
            logger.info(
                "started",
                extra={
                    "task_class": task_class.__name__,
                    "system_class": system_class.__name__,
                    "show_progress": show_progress,
                    "reset_between_instances": reset_between_instances,
                },
            )
            filtered_params = filter_init_params(task_class, task_params)
            task = task_class(**filtered_params)
            derived_run_index = task_params.get(
                "run_index", task_params.get("rollout_index")
            )
            if derived_run_index is not None:
                if derived_run_index == 0 and hasattr(task, "instances"):
                    _before = list(task.instances)  # type: ignore[attr-defined]
                    task.prepare_run(0)
                    _after = list(task.instances)  # type: ignore[attr-defined]
                    if _before != _after:
                        logger.warning(
                            "%s.prepare_run(0) changed self.instances — "
                            "baseline and repeated runs will see different instance orderings. "
                            "prepare_run(run_index=0) must be a no-op.",
                            type(task).__name__,
                        )
                else:
                    task.prepare_run(derived_run_index)
            system = system_class(**system_params)
            system.reset()
            task_brief = task.get_agent_brief()
            trace_task_params = dict(task_params)
            expected_instances = getattr(task, "num_instances", None)
            if isinstance(expected_instances, int) and expected_instances > 0:
                trace_task_params["expected_num_instances"] = expected_instances

            trace_recorder = TraceRecorder(
                system_name=system_name,
                task_name=task_name,
                system_params=system_params,
                task_params=trace_task_params,
                run_group_id=run_group_id,
                run_index=run_index,
                task_brief=task_brief,
                phase=phase,
                live_trace_path=live_trace_path,
            )
            logger.info(
                "components.ready",
                extra={
                    "expected_instances": expected_instances,
                    "has_task_brief": task_brief is not None,
                    "live_trace_path": None
                    if live_trace_path is None
                    else str(live_trace_path),
                },
            )

            try:
                result = run_task(
                    task,
                    system,
                    trace_recorder=trace_recorder,
                    show_progress=show_progress,
                    verbose_logging=verbose_runs,
                    rollout_label=(
                        "baseline"
                        if phase == "baseline"
                        else None
                        if run_index is None
                        else f"run {run_index + 1}"
                    ),
                    reset_between_instances=reset_between_instances,
                    phase=phase,
                )
            except ProviderRefusalError as exc:
                # Baseline runs annotate refusals at the orchestrator level (each
                # baseline instance owns one refusal).  For rollout runs we
                # finalize the partial trace here so the live viewer keeps the
                # data and the orchestrator only has to record metadata.
                if phase == "baseline":
                    raise
                partial_outcomes = list(trace_recorder.instance_outcomes)
                partial_result = build_blocked_task_result(exc, partial_outcomes)
                blocked_payload = exc.to_record()
                trace_data = trace_recorder.finalize(
                    partial_result,
                    status="blocked",
                    extra={"blocked_reason": blocked_payload},
                )
                execution_summary = {
                    **trace_data["execution"],
                    "status": "blocked",
                    "phase": trace_data.get("phase", phase),
                    "task_brief": trace_data.get("task_brief"),
                    "instance_outcomes": trace_data.get("instance_outcomes", []),
                    "blocked_reason": blocked_payload,
                }
                logger.warning(
                    "blocked",
                    extra={"blocked_reason": blocked_payload},
                )
                return (run_index, partial_result, trace_data, execution_summary)

            trace_data = trace_recorder.finalize(result)
            execution_summary = {
                **trace_data["execution"],
                "status": trace_data.get("status", "completed"),
                "phase": trace_data.get("phase", phase),
                "task_brief": trace_data.get("task_brief"),
                "instance_outcomes": trace_data.get("instance_outcomes", []),
            }
            logger.info(
                "finished",
                extra={
                    "score": result.score,
                    "status": execution_summary.get("status"),
                    "interactions": execution_summary.get("total_interactions"),
                },
            )
            return (
                run_index,
                result,
                trace_data,
                execution_summary,
            )
    except ProviderRefusalError:
        raise
    except Exception as exc:
        logger.error(
            "failed",
            extra={
                "phase": phase,
                "run_index": run_index,
                "run_group_id": run_group_id,
                "error_type": type(exc).__name__,
            },
            exc_info=True,
        )
        try:
            run_artifacts: Optional[dict[str, Any]] = None
            if "system" in locals() and hasattr(system, "get_run_artifacts"):
                try:
                    run_artifacts = system.get_run_artifacts()
                except Exception:
                    pass
            failed_trace_path = build_trace_output_path(
                task_name,
                task_params=task_params,
                run_group_id=run_group_id,
                run_index=run_index,
                kind="failed",
            )
            failed_payload: dict[str, Any] = {
                "status": "failed",
                "error": {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                "artifacts": build_artifact_manifest(run_artifacts),
                "system_artifacts": None,
            }
            failed_trace_path.write_text(
                json.dumps(failed_payload, indent=2), encoding="utf-8"
            )
            save_artifacts(run_artifacts, failed_trace_path)
        except Exception:
            pass
        raise RuntimeError(
            f"[{phase} {human_run_number}, ID {run_group_id}] {type(exc).__name__}: {exc}"
        ) from exc
