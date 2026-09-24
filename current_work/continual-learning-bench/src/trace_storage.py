"""
Enhanced trace storage for detailed system-task interaction logging.

Captures complete interaction history including queries, responses, observations,
and system metadata for debugging and analysis.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Optional

from .artifacts import build_artifact_manifest
from .interface import (
    InstanceOutcome,
    Observation,
    Query,
    Response,
    TaskAgentBrief,
    TaskResult,
    instance_outcome_identity,
    observation_marks_instance_complete,
    serialize_instance_outcome,
    serialize_task_agent_brief,
)
from .run_ids import safe_run_id_component
from .system_manifest import build_system_manifest_entry
from .trace_metrics import (
    attach_baseline_to_outcomes,
    attach_interaction_metrics_to_outcomes,
    build_benchmark_aggregate,
    build_sequence_signature,
    summarize_instance_outcomes,
)
from .usage import summarize_usage_events

logger = logging.getLogger(__name__)

__all__ = [
    "BaselineSummaryEntry",
    "InteractionStep",
    "RunSummary",
    "RunSummaryRunEntry",
    "TraceRecorder",
    "attach_baseline_to_outcomes",
    "attach_interaction_metrics_to_outcomes",
    "build_benchmark_aggregate",
    "build_sequence_signature",
    "load_run_summaries",
    "load_run_summary",
    "save_run_summary",
    "save_trace",
    "save_viewer_artifact",
    "summarize_instance_outcomes",
]


@dataclass
class InteractionStep:
    """A single step in the interaction loop."""

    step_number: int
    timestamp: str
    query: dict[str, Any]  # Query as dict
    response: dict[str, Any]  # Response as dict
    observation: dict[str, Any]  # Observation as dict
    timing: dict[str, Any]
    usage: dict[str, Any]
    done: bool


def _copy_string_keyed_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _build_system_memory_metadata(
    interactions: list[InteractionStep],
    system_artifacts: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Collect memory snapshots from interaction metadata and final artifacts."""
    backend: Any = None
    memory_dir: Any = None
    files: dict[str, Any] = {}
    native_files: dict[str, Any] = {}
    other_files: dict[str, Any] = {}
    file_sources: dict[str, Any] = {}
    other_file_mapping: dict[str, Any] = {}
    other_file_patterns: list[Any] = []
    history: list[dict[str, Any]] = []

    for interaction in interactions:
        response = interaction.response or {}
        metadata = response.get("metadata") if isinstance(response, dict) else None
        if not isinstance(metadata, dict):
            continue
        backend = metadata.get("memory_backend", backend)
        memory_dir = metadata.get("memory_dir", memory_dir)
        memory_files = _copy_string_keyed_dict(metadata.get("memory_files"))
        if "memory_files" in metadata:
            files = memory_files
            history.append({"step": interaction.step_number, "files": memory_files})
        if "native_memory_files" in metadata:
            native_files = _copy_string_keyed_dict(metadata.get("native_memory_files"))
        if "other_memory_files" in metadata:
            other_files = _copy_string_keyed_dict(metadata.get("other_memory_files"))
        if "memory_file_sources" in metadata:
            file_sources = _copy_string_keyed_dict(metadata.get("memory_file_sources"))
        if "other_memory_file_mapping" in metadata:
            other_file_mapping = _copy_string_keyed_dict(
                metadata.get("other_memory_file_mapping")
            )
        patterns = metadata.get("other_memory_file_patterns")
        if isinstance(patterns, list):
            other_file_patterns = list(patterns)

    if isinstance(system_artifacts, dict):
        backend = system_artifacts.get("memory_backend", backend)
        memory_dir = system_artifacts.get("memory_dir", memory_dir)
        artifact_files = _copy_string_keyed_dict(system_artifacts.get("memory_files"))
        if artifact_files:
            files = artifact_files
        if "native_memory_files" in system_artifacts:
            native_files = _copy_string_keyed_dict(
                system_artifacts.get("native_memory_files")
            )
        if "other_memory_files" in system_artifacts:
            other_files = _copy_string_keyed_dict(
                system_artifacts.get("other_memory_files")
            )
        if "memory_file_sources" in system_artifacts:
            file_sources = _copy_string_keyed_dict(
                system_artifacts.get("memory_file_sources")
            )
        if "other_memory_file_mapping" in system_artifacts:
            other_file_mapping = _copy_string_keyed_dict(
                system_artifacts.get("other_memory_file_mapping")
            )
        artifact_patterns = system_artifacts.get("other_memory_file_patterns")
        if isinstance(artifact_patterns, list):
            other_file_patterns = list(artifact_patterns)
        artifact_history = system_artifacts.get("memory_history")
        if isinstance(artifact_history, list):
            history = artifact_history

    if not any([backend, memory_dir, files, history]):
        return None
    payload = {
        "files": files,
        "history": history,
    }
    if backend is not None:
        payload["backend"] = backend
    if memory_dir is not None:
        payload["memory_dir"] = memory_dir
    if any(
        [
            native_files,
            other_files,
            file_sources,
            other_file_mapping,
            other_file_patterns,
        ]
    ):
        payload.update(
            {
                "native_files": native_files,
                "other_files": other_files,
                "file_sources": file_sources,
                "other_file_mapping": other_file_mapping,
                "other_file_patterns": other_file_patterns,
            }
        )
    return payload


def _is_gzip_path(path: Path) -> bool:
    return path.suffix == ".gz"


def _atomic_write_json(path: Path | str | PathLike[str], payload: Any) -> None:
    """Write JSON atomically so the live dashboard never reads partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if _is_gzip_path(path):
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        tmp_path.write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
    else:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _read_json(path: Path | str | PathLike[str]) -> Any:
    path = Path(path)
    if _is_gzip_path(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


class TraceRecorder:
    """Records full interaction traces during task execution."""

    def __init__(
        self,
        system_name: str,
        task_name: str,
        system_params: dict,
        task_params: dict,
        trace_path: Optional[Path] = None,
        run_group_id: Optional[str] = None,
        run_index: Optional[int] = None,
        task_brief: Optional[TaskAgentBrief] = None,
        phase: str = "rollout",
        live_trace_path: Optional[Path] = None,
    ):
        self.system_name = system_name
        self.task_name = task_name
        self.system_params = system_params
        self.task_params = task_params
        self.trace_path = trace_path or (
            Path("results") / task_name / "trace_pending.json"
        )
        self.run_group_id = run_group_id
        self.run_index = run_index
        self.task_brief = task_brief
        self.phase = phase
        self.interactions: list[InteractionStep] = []
        self.instance_outcomes: list[InstanceOutcome] = []
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.system_artifacts: Optional[dict[str, Any]] = None
        self.status: str = "running"
        self.live_trace_path = live_trace_path

    def record_interaction(
        self,
        step_number: int,
        query: Query,
        response: Response,
        observation: Observation,
        done: bool,
        timing: Optional[dict[str, Any]] = None,
        usage: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a single interaction step.

        Args:
            step_number: The interaction step number
            query: Query from task to system
            response: Response from system
            observation: Observation/feedback from task
            done: Whether the task is complete
            timing: Optional timing metadata for this interaction
            usage: Optional cost/usage metadata for this interaction
        """
        query_payload: dict[str, Any] = {
            "prompt": query.prompt,
            "instance_id": query.instance_id,
            "instance_index": query.instance_index,
            "metadata": query.metadata or {},
            "response_schema": query.response_schema.__name__,
            "feedback": (
                {
                    "content": query.feedback.content,
                    "instance_complete": observation_marks_instance_complete(
                        query.feedback
                    ),
                    "metadata": query.feedback.metadata or {},
                }
                if query.feedback is not None
                else None
            ),
        }
        if hasattr(response.action, "model_dump"):
            response_action = response.action.model_dump()
            response_action_type = "structured"
        else:
            response_action = str(response.action)
            response_action_type = "unknown"

        interaction = InteractionStep(
            step_number=step_number,
            timestamp=datetime.now().isoformat(),
            query=query_payload,
            response={
                "action": response_action,
                "action_type": response_action_type,
                "metadata": response.metadata or {},
            },
            observation={
                "content": observation.content,
                "instance_complete": observation_marks_instance_complete(observation),
                "metadata": observation.metadata or {},
            },
            timing=timing or {},
            usage=usage or {},
            done=done,
        )
        self.interactions.append(interaction)
        logger.debug(
            "interaction.recorded",
            extra={
                "step": step_number,
                "done": done,
                "instance_id": query.instance_id,
                "instance_index": query.instance_index,
                "action_type": response_action_type,
                "response_latency_seconds": (timing or {}).get(
                    "response_latency_seconds"
                ),
            },
        )
        self.write_live_snapshot()

    def record_system_artifacts(self, artifacts: Optional[dict[str, Any]]) -> None:
        """Record optional system artifacts for post-run export."""
        self.system_artifacts = artifacts
        logger.info(
            "artifacts.recorded",
            extra={
                "artifact_type": None
                if artifacts is None
                else artifacts.get("artifact_type"),
                "has_artifacts": artifacts is not None,
            },
        )
        self.write_live_snapshot()

    def sync_instance_outcomes(
        self, outcomes: list[InstanceOutcome]
    ) -> list[InstanceOutcome]:
        """Upsert a batch of normalized per-instance outcomes."""
        changed: list[InstanceOutcome] = []
        index_by_key = {
            instance_outcome_identity(existing): idx
            for idx, existing in enumerate(self.instance_outcomes)
        }
        for outcome in outcomes:
            outcome_key = instance_outcome_identity(outcome)
            existing_idx = index_by_key.get(outcome_key)
            if existing_idx is None:
                self.instance_outcomes.append(outcome)
                index_by_key[outcome_key] = len(self.instance_outcomes) - 1
                changed.append(outcome)
                continue
            if self.instance_outcomes[existing_idx] != outcome:
                self.instance_outcomes[existing_idx] = outcome
                changed.append(outcome)
        if changed:
            logger.debug(
                "outcomes.synced",
                extra={
                    "changed_outcomes": len(changed),
                    "total_outcomes": len(self.instance_outcomes),
                },
            )
            self.write_live_snapshot()
        return changed

    def record_instance_outcome(self, outcome: InstanceOutcome) -> None:
        """Record a normalized per-instance outcome."""
        self.sync_instance_outcomes([outcome])

    def set_status(self, status: str) -> None:
        """Update the current trace status and refresh any live snapshot."""
        self.status = status
        self.write_live_snapshot()

    def _build_trace_payload(
        self,
        *,
        result: Optional[TaskResult],
        include_artifacts: bool = True,
    ) -> dict[str, Any]:
        """Build a trace payload for final saves and live snapshots."""
        end_time = self.end_time
        if end_time is None:
            end_time = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.start_time)
        end_dt = datetime.fromisoformat(end_time)
        wall_duration_seconds = (end_dt - start_dt).total_seconds()

        response_latencies = []
        respond_usage_events: list[dict[str, Any]] = []
        observe_usage_events: list[dict[str, Any]] = []
        for interaction in self.interactions:
            latency_seconds = interaction.timing.get("response_latency_seconds")
            if isinstance(latency_seconds, (int, float)):
                response_latencies.append(float(latency_seconds))
            usage = interaction.usage or {}
            respond_usage_events.extend(
                usage.get("respond", {}).get("events", []) or []
            )
            observe_usage_events.extend(
                usage.get("observe", {}).get("events", []) or []
            )

        total_response_seconds = sum(response_latencies)
        avg_response_seconds = (
            total_response_seconds / len(response_latencies)
            if response_latencies
            else 0.0
        )
        max_response_seconds = max(response_latencies) if response_latencies else 0.0
        interaction_usage_summary = summarize_usage_events(
            [*respond_usage_events, *observe_usage_events]
        )

        if result is not None and result.instance_outcomes:
            serialized_outcomes = [
                serialize_instance_outcome(outcome)
                for outcome in result.instance_outcomes
            ]
        else:
            serialized_outcomes = [
                serialize_instance_outcome(outcome)
                for outcome in self.instance_outcomes
            ]

        attach_interaction_metrics_to_outcomes(serialized_outcomes, self.interactions)

        schedule_id = (
            self.task_params.get(
                "schedule", self.task_params.get("rollout_schedule", "")
            )
            or ""
        )
        payload: dict[str, Any] = {
            "status": self.status,
            "phase": self.phase,
            "schedule": schedule_id,
            "task_brief": serialize_task_agent_brief(self.task_brief),
            "system": build_system_manifest_entry(self.system_name, self.system_params),
            "task": {"name": self.task_name, "params": self.task_params},
            "execution": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_interactions": len(self.interactions),
                "wall_duration_seconds": round(wall_duration_seconds, 6),
                "total_response_seconds": round(total_response_seconds, 6),
                "avg_response_seconds": round(avg_response_seconds, 6),
                "max_response_seconds": round(max_response_seconds, 6),
                "run_group_id": self.run_group_id,
                "run_index": self.run_index,
                "usage": {
                    "respond": summarize_usage_events(respond_usage_events),
                    "observe": summarize_usage_events(observe_usage_events),
                    "interaction": interaction_usage_summary,
                },
            },
            "artifacts": (
                build_artifact_manifest(self.system_artifacts)
                if include_artifacts
                else {"files": []}
            ),
            "system_memory": _build_system_memory_metadata(
                self.interactions,
                self.system_artifacts if include_artifacts else None,
            ),
            "system_artifacts": self.system_artifacts if include_artifacts else None,
            "interactions": [asdict(interaction) for interaction in self.interactions],
            "instance_outcomes": serialized_outcomes,
            "result": (
                None
                if result is None
                else {
                    "score": result.score,
                    "metrics": result.metrics,
                    "summary": result.summary,
                    "eval_metrics": asdict(result.eval_metrics)
                    if result.eval_metrics
                    else None,
                    "instance_outcomes": serialized_outcomes,
                }
            ),
        }
        return payload

    def write_live_snapshot(
        self,
        result: Optional[TaskResult] = None,
        *,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist the current trace state for the live dashboard, if enabled."""
        if self.live_trace_path is None:
            return
        payload = self._build_trace_payload(result=result, include_artifacts=False)
        if extra:
            payload.update(extra)
        _atomic_write_json(self.live_trace_path, payload)
        logger.debug(
            "snapshot.written",
            extra={
                "path": str(self.live_trace_path),
                "status": payload.get("status"),
                "interactions": len(self.interactions),
            },
        )

    def finalize(
        self,
        result: TaskResult,
        *,
        status: str = "completed",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Finalize the trace with task results.

        Args:
            result: Final task evaluation result
            status: Trace status to record (e.g. ``completed`` or ``blocked``).
            extra: Optional top-level fields to merge into the payload (used to
                attach refusal metadata for blocked runs).

        Returns:
            Complete trace as dictionary
        """
        self.status = status
        self.end_time = datetime.now().isoformat()
        payload = self._build_trace_payload(result=result, include_artifacts=True)
        if extra:
            payload.update(extra)
        self.write_live_snapshot(result=result, extra=extra)
        logger.info(
            "finalized",
            extra={
                "status": status,
                "phase": self.phase,
                "interactions": len(self.interactions),
                "outcomes": len(self.instance_outcomes),
            },
        )
        return payload


def build_trace_output_path(
    task_name: str,
    *,
    task_params: dict[str, Any],
    run_group_id: Optional[str],
    run_index: Optional[int],
    output_path: Optional[Path] = None,
    kind: str = "trace",
) -> Path:
    """Build the canonical output path for a trace file."""
    traces_dir = Path("results") / task_name
    traces_dir.mkdir(parents=True, exist_ok=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rollout_schedule_id = task_params.get(
        "schedule", task_params.get("rollout_schedule")
    )
    variant_id = task_params.get("variant")

    parts = [kind]
    selected_id = rollout_schedule_id or variant_id
    if isinstance(selected_id, str) and selected_id:
        parts.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", selected_id))
    if run_group_id is not None:
        parts.append(safe_run_id_component(run_group_id))
        if run_index is not None:
            parts.append(f"run{run_index}")
    parts.append(timestamp)
    return traces_dir / f"{'_'.join(parts)}.json"


def save_trace(
    trace_data: dict[str, Any], task_name: str, output_path: Optional[Path] = None
) -> Path:
    """
    Save complete trace data to file.

    Args:
        trace_data: Complete trace dictionary from TraceRecorder.finalize()
        task_name: Task name for organizing files
        output_path: Optional specific output path

    Returns:
        Path where trace was saved
    """
    execution = trace_data.get("execution", {})
    output_path = build_trace_output_path(
        task_name,
        task_params=trace_data.get("task", {}).get("params", {}),
        run_group_id=execution.get("run_group_id"),
        run_index=execution.get("run_index"),
        output_path=output_path,
        kind="baseline" if trace_data.get("phase") == "baseline" else "trace",
    )

    _atomic_write_json(output_path, trace_data)
    logger.info(
        "saved",
        extra={
            "path": str(output_path),
            "task": task_name,
            "phase": trace_data.get("phase"),
        },
    )
    return output_path


# ---------------------------------------------------------------------------
# Run summary data model (multi-rollout aggregates)
# ---------------------------------------------------------------------------


@dataclass
class RunSummaryRunEntry:
    """A single rollout's score and trace path within a run summary."""

    run_index: int
    score: float
    trace_path: str
    execution: Optional[dict[str, Any]] = None
    status: str = "completed"
    instance_outcomes: list[dict[str, Any]] = field(default_factory=list)
    aggregate: Optional[dict[str, Any]] = None


@dataclass
class BaselineSummaryEntry:
    """The reset-each-instance baseline trace and outcomes for a benchmark run."""

    trace_path: str
    execution: Optional[dict[str, Any]] = None
    status: str = "completed"
    instance_outcomes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunSummary:
    """Typed model for a multi-rollout run summary.

    ``variant`` is promoted to a top-level field for direct filtering.
    ``run_mode`` records which sources of variance were active;
    it is None for single runs and legacy summaries.
    """

    run_group_id: str
    system: dict[str, Any]
    task: dict[str, Any]
    run_count: int
    max_workers: int
    aggregate: dict[str, Any]
    runs: list[RunSummaryRunEntry]
    baseline: Optional[BaselineSummaryEntry] = None
    variant: Optional[str] = None
    schedule: Optional[str] = None
    run_mode: Optional[str] = None
    status: str = "completed"
    baseline_mode: str = "reset_each_instance"
    task_brief: Optional[dict[str, Any]] = None
    sequence_signature: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_group_id": self.run_group_id,
            "status": self.status,
            "system": self.system,
            "task": self.task,
            "task_brief": self.task_brief,
            "variant": self.variant,
            "schedule": self.schedule,
            "rollout_schedule": self.schedule,
            "run_count": self.run_count,
            "rollouts": self.run_count,
            "max_workers": self.max_workers,
            "run_mode": self.run_mode,
            "rollout_mode": self.run_mode,
            "baseline_mode": self.baseline_mode,
            "baseline": None if self.baseline is None else asdict(self.baseline),
            "sequence_signature": self.sequence_signature,
            "aggregate": self.aggregate,
            "runs": [asdict(r) for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSummary:
        task_info = data["task"]
        variant = data.get("variant", task_info.get("params", {}).get("variant"))
        schedule = data.get(
            "schedule",
            data.get(
                "rollout_schedule",
                task_info.get("params", {}).get(
                    "schedule", task_info.get("params", {}).get("rollout_schedule")
                ),
            ),
        )
        runs = [RunSummaryRunEntry(**r) for r in data["runs"]]
        baseline_raw = data.get("baseline")
        return cls(
            run_group_id=data["run_group_id"],
            system=data["system"],
            task=task_info,
            variant=variant,
            schedule=schedule,
            run_count=data.get("run_count", data.get("rollouts", len(runs))),
            max_workers=data["max_workers"],
            aggregate=data["aggregate"],
            runs=runs,
            run_mode=data.get("run_mode", data.get("rollout_mode")),
            baseline=(
                None if baseline_raw is None else BaselineSummaryEntry(**baseline_raw)
            ),
            status=data.get("status", "completed"),
            baseline_mode=data.get("baseline_mode", "reset_each_instance"),
            task_brief=data.get("task_brief"),
            sequence_signature=data.get("sequence_signature", []),
        )


def save_run_summary(summary: RunSummary, task_name: str) -> Path:
    """Save a RunSummary to disk."""
    traces_dir = Path("results") / task_name
    traces_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_component = safe_run_id_component(summary.run_group_id)
    output_path = traces_dir / f"run_summary_{run_id_component}_{timestamp}.json"

    _atomic_write_json(output_path, summary.to_dict())
    return output_path


def save_viewer_artifact(
    summary: RunSummary,
    task_name: str,
    baseline_trace: Optional[dict[str, Any]] = None,
    run_traces: Optional[list[dict[str, Any]]] = None,
) -> Path:
    """Save a single-file viewer artifact with embedded summary and traces."""
    traces_dir = Path("results") / task_name
    traces_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_component = safe_run_id_component(summary.run_group_id)
    output_path = traces_dir / f"viewer_artifact_{run_id_component}_{timestamp}.json.gz"

    run_traces = run_traces or []
    payload: dict[str, Any] = {
        "kind": "viewer_artifact",
        "summary": summary.to_dict(),
        "baseline_trace": baseline_trace,
        "run_traces": [
            {
                "run_index": run.run_index,
                "trace": run_traces[i] if i < len(run_traces) else None,
            }
            for i, run in enumerate(summary.runs)
        ],
    }

    _atomic_write_json(output_path, payload)
    return output_path


def _extract_run_summary_payload(raw: Any) -> dict[str, Any]:
    """Return a RunSummary payload from either a legacy summary or viewer artifact."""
    if not isinstance(raw, dict):
        raise KeyError("Run summary payload must be a JSON object.")
    if raw.get("kind") == "viewer_artifact":
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            raise KeyError("Viewer artifact is missing a valid summary payload.")
        return summary
    return raw


def load_run_summary(path: Path | str) -> RunSummary:
    """Load a run summary or viewer artifact JSON file."""
    raw = _read_json(path)
    return RunSummary.from_dict(_extract_run_summary_payload(raw))


def load_run_summaries(
    task_name: str,
    *,
    variant: Optional[str] = None,
    schedule: Optional[str] = None,
) -> list[RunSummary]:
    """Load all run summaries for a task, optionally filtered by variant/schedule."""
    repo_root = Path(__file__).resolve().parents[1]
    traces_dir = repo_root / "results" / task_name
    if not traces_dir.is_dir():
        return []

    summaries: list[RunSummary] = []
    paths = sorted(
        {
            *traces_dir.glob("viewer_artifact_*.json"),
            *traces_dir.glob("viewer_artifact_*.json.gz"),
        }
    )
    for p in paths:
        try:
            s = load_run_summary(p)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if variant is not None and s.variant != variant:
            continue
        if schedule is not None and s.schedule != schedule:
            continue
        summaries.append(s)
    return summaries
