import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from src.artifacts import register_artifact_exporter
from src.errors import ProviderRefusalError
from src.interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    Observation,
    Query,
    EvalMetrics,
    InstanceOutcome,
    Response,
    TaskResult,
    TaskStepResult,
)
from src.runs import RunMode, derive_run_task_params, run_benchmark_runs
from src.runs import (
    _merge_baseline_instance_results,
    _run_baseline_instance,
    run_baseline,
    run_benchmark,
)
from src.usage import summarize_usage_events


class DummyAction(BaseModel):
    value: str = "ok"


def _usage_payload(
    *,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    model: str = "dummy-model",
) -> dict[str, object]:
    event = {
        "call_type": "completion",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd,
    }
    return {
        "respond": {"events": [event]},
        "interaction": summarize_usage_events([event]),
    }


class DummyTask(ContinualLearningTask):
    def __init__(self, seed: int = 42, variant: str = "v1", num_instances: int = 1):
        self._seed = seed
        self._variant = variant
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self._step_count = 0

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self._step_count = 0

    def build_current_query(self) -> Query:
        return Query(
            prompt="start",
            response_schema=DummyAction,
            instance_id="dummy-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        self._step_count += 1
        return TaskStepResult(
            observation=Observation(content="ok"),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={"m1": 1.0},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0, 0.5, 1.0],
                optimal_performance=1.0,
                actual_performance=0.5,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="stub", instance_index=0, reward=0.5)
            ],
        )

    @property
    def name(self) -> str:
        return "dummy"


class DummySystem(ContinualLearningSystem):
    def respond(self, query: Query) -> Response:
        return Response(action=DummyAction())

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "dummy_system"


class FailingTask(ContinualLearningTask):
    def __init__(
        self,
        seed: int = 42,
        variant: str = "v1",
        num_instances: int = 1,
        rollout_index: int | None = None,
        fail_rollout_index: int = 2,
    ):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self._rollout_index = rollout_index
        self._fail_rollout_index = fail_rollout_index

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))

    def build_current_query(self) -> Query:
        return Query(
            prompt="start",
            response_schema=DummyAction,
            instance_id="failing-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        if self._rollout_index == self._fail_rollout_index:
            raise ValueError("boom")
        return TaskStepResult(
            observation=Observation(content="ok"),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={"m1": 1.0},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0, 0.5, 1.0],
                optimal_performance=1.0,
                actual_performance=0.5,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="stub", instance_index=0, reward=0.5)
            ],
        )

    @property
    def name(self) -> str:
        return "failing_dummy"


class PrePopulatedBaselineTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 2):
        self.num_instances = num_instances
        self.instances = [f"instance {i}" for i in range(num_instances)]

    def build_canonical_run_state(self) -> None:
        self.instances = [f"instance {i}" for i in range(self.num_instances)]

    def build_current_query(self) -> Query:
        canonical_index = self.canonical_instance_index(0)
        return Query(
            prompt=self.instances[0],
            response_schema=DummyAction,
            instance_id=self.instances[0],
            instance_index=canonical_index,
        )

    def step(self, response: Response) -> TaskStepResult:
        return TaskStepResult(
            observation=Observation(content="ok"),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0],
                optimal_performance=1.0,
                actual_performance=1.0,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="stub", instance_index=0, reward=1.0)
            ],
        )

    @property
    def name(self) -> str:
        return "prepopulated_baseline"


class PostResetBaselineTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 2):
        self.num_instances = num_instances
        self._built_instances: list[str] = []

    def build_canonical_run_state(self) -> None:
        self._built_instances = [f"instance {i}" for i in range(self.num_instances)]

    def build_current_query(self) -> Query:
        return Query(
            prompt=self._built_instances[0],
            response_schema=DummyAction,
            instance_id=self._built_instances[0],
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        return TaskStepResult(
            observation=Observation(content="ok"),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0],
                optimal_performance=1.0,
                actual_performance=1.0,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="stub", instance_index=0, reward=1.0)
            ],
        )

    @property
    def name(self) -> str:
        return "postreset_baseline"


class ExplicitBaselineTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 2):
        self.num_instances = num_instances
        self.instances: list[int] = []
        self._active_instance = 0
        self._done = False
        self._instance_outcomes: list[InstanceOutcome] = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self._done = False
        self._instance_outcomes = []
        self._active_instance = 0

    def select_run_instances(self, instance_indices: list[int] | None) -> None:
        super().select_run_instances(instance_indices)
        self._active_instance = self.canonical_instance_index(0)
        self._done = False
        self._instance_outcomes = []

    def build_current_query(self) -> Query:
        return Query(
            prompt=f"instance {self._active_instance}",
            response_schema=DummyAction,
            instance_id=str(self._active_instance),
            instance_index=self._active_instance,
        )

    def step(self, response: Response) -> TaskStepResult:
        if self._done:
            raise AssertionError("step called twice")
        self._done = True
        outcome = InstanceOutcome(
            instance_id=str(self._active_instance),
            instance_index=self._active_instance,
            reward=1.0,
            success=True,
        )
        self._instance_outcomes.append(outcome)
        return TaskStepResult(
            observation=Observation(content="ok"),
            next_query=None,
            done=True,
            instance_outcome=outcome,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0],
                optimal_performance=1.0,
                actual_performance=1.0,
            ),
            instance_outcomes=list(self._instance_outcomes),
        )

    @property
    def name(self) -> str:
        return "explicit_baseline"


class UnsafeDummyTask(DummyTask):
    parallel_safe = False


class CountingInstancesTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 3):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self.current_index = 0
        self._instance_outcomes: list[InstanceOutcome] = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self.current_index = 0
        self._instance_outcomes = []

    def select_run_instances(self, instance_indices: list[int] | None) -> None:
        super().select_run_instances(instance_indices)
        self.current_index = 0

    def build_current_query(self) -> Query:
        canonical_index = self.canonical_instance_index(self.current_index)
        return Query(
            prompt=f"instance {canonical_index}",
            response_schema=DummyAction,
            instance_id=str(canonical_index),
            instance_index=canonical_index,
        )

    def step(self, response: Response) -> TaskStepResult:
        canonical_index = self.canonical_instance_index(self.current_index)
        reward = float(canonical_index + 1)
        outcome = InstanceOutcome(
            instance_id=str(canonical_index),
            instance_index=canonical_index,
            reward=reward,
            success=True,
        )
        self._instance_outcomes.append(outcome)
        self.current_index += 1
        done = self.current_index >= len(self.instances)
        return TaskStepResult(
            observation=Observation(content="ok", instance_complete=True),
            next_query=None if done else self.build_current_query(),
            done=done,
            instance_outcome=outcome,
        )

    def evaluate(self) -> TaskResult:
        rewards = [outcome.reward for outcome in self._instance_outcomes]
        return TaskResult(
            metrics={"rewards": rewards},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0 for _ in rewards],
                optimal_performance=float(len(rewards)),
                actual_performance=sum(rewards),
            ),
            instance_outcomes=list(self._instance_outcomes),
        )

    @property
    def name(self) -> str:
        return "counting_instances"


class RefusalOnInstanceSystem(ContinualLearningSystem):
    def __init__(self, blocked_instance_index: int = 1):
        self.blocked_instance_index = blocked_instance_index

    def respond(self, query: Query) -> Response:
        if query.instance_index == self.blocked_instance_index:
            raise ProviderRefusalError(
                kind="content_policy",
                message="prompt blocked by provider",
                provider="openai",
                model="gpt-test",
            )
        return Response(action=DummyAction())

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "refusal_on_instance"


# ---------------------------------------------------------------------------
# Pure unit tests for _derive_task_params (no I/O, no processes)
# ---------------------------------------------------------------------------


class DeriveTaskParamsTests(unittest.TestCase):
    def test_replicate_preserves_seed(self):
        params = {"seed": 42, "variant": "v1"}
        derived = derive_run_task_params(params, RunMode.REPLICATE, 3)
        self.assertEqual(derived["seed"], 42)

    def test_resample_preserves_seed(self):
        params = {"seed": 42, "variant": "v1"}
        derived = derive_run_task_params(params, RunMode.RESAMPLE, 2)
        self.assertEqual(derived["seed"], 42)

    def test_resample_injects_rollout_index(self):
        params = {"seed": 42, "variant": "v1"}
        derived = derive_run_task_params(params, RunMode.RESAMPLE, 2)
        self.assertEqual(derived["rollout_index"], 2)

    def test_permute_preserves_seed(self):
        params = {"seed": 42, "variant": "v1"}
        derived = derive_run_task_params(params, RunMode.PERMUTE, 5)
        self.assertEqual(derived["seed"], 42)

    def test_permute_injects_rollout_index(self):
        params = {"seed": 42, "variant": "v1"}
        derived = derive_run_task_params(params, RunMode.PERMUTE, 5)
        self.assertEqual(derived["rollout_index"], 5)

    def test_resample_no_seed_key(self):
        params = {"variant": "v1"}
        derived = derive_run_task_params(params, RunMode.RESAMPLE, 1)
        self.assertNotIn("seed", derived)

    def test_resample_seed_none(self):
        params = {"seed": None, "variant": "v1"}
        derived = derive_run_task_params(params, RunMode.RESAMPLE, 1)
        self.assertIsNone(derived["seed"])

    def test_does_not_mutate_original(self):
        params = {"seed": 42}
        derive_run_task_params(params, RunMode.RESAMPLE, 3)
        self.assertEqual(params["seed"], 42)


class BaselineMergeTests(unittest.TestCase):
    def test_parallel_baseline_merge_attaches_aggregated_instance_metrics(self):
        instance_results = [
            (
                0,
                TaskResult(
                    metrics={},
                    summary="inst-0",
                    eval_metrics=EvalMetrics(
                        loss_curve=[0.0],
                        optimal_performance=1.0,
                        actual_performance=1.0,
                    ),
                    instance_outcomes=[
                        InstanceOutcome(
                            instance_id="inst-0",
                            instance_index=0,
                            reward=1.0,
                        )
                    ],
                ),
                {
                    "interactions": [
                        {
                            "query": {"instance_id": "inst-0", "instance_index": 0},
                            "timing": {"response_latency_seconds": 0.4},
                            "usage": _usage_payload(
                                input_tokens=10,
                                output_tokens=5,
                                cost_usd=0.1,
                            ),
                        },
                        {
                            "query": {"instance_id": "inst-0", "instance_index": 0},
                            "timing": {"response_latency_seconds": 0.6},
                            "usage": _usage_payload(
                                input_tokens=12,
                                output_tokens=6,
                                cost_usd=0.2,
                            ),
                        },
                    ],
                    "task_brief": None,
                    "system": {"name": "dummy_system", "params": {}},
                    "artifacts": {},
                },
            ),
            (
                1,
                TaskResult(
                    metrics={},
                    summary="inst-1",
                    eval_metrics=EvalMetrics(
                        loss_curve=[0.0],
                        optimal_performance=1.0,
                        actual_performance=0.5,
                    ),
                    instance_outcomes=[
                        InstanceOutcome(
                            instance_id="inst-1",
                            instance_index=1,
                            reward=0.5,
                        )
                    ],
                ),
                {
                    "interactions": [
                        {
                            "query": {"instance_id": "inst-1", "instance_index": 1},
                            "timing": {"response_latency_seconds": 1.5},
                            "usage": _usage_payload(
                                input_tokens=20,
                                output_tokens=10,
                                cost_usd=0.4,
                            ),
                        }
                    ],
                    "task_brief": None,
                    "system": {"name": "dummy_system", "params": {}},
                    "artifacts": {},
                },
            ),
        ]

        _, trace_data, execution_summary = _merge_baseline_instance_results(
            instance_results=instance_results,
            run_group_id="group123",
            system_name="dummy_system",
            task_name="dummy_task",
            task_params={},
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:10",
        )

        outcomes = trace_data["instance_outcomes"]
        self.assertAlmostEqual(outcomes[0]["cost_usd"], 0.3)
        self.assertAlmostEqual(outcomes[0]["latency_seconds"], 1.0)
        self.assertAlmostEqual(outcomes[1]["cost_usd"], 0.4)
        self.assertAlmostEqual(outcomes[1]["latency_seconds"], 1.5)
        self.assertAlmostEqual(execution_summary["total_response_seconds"], 2.5)
        self.assertAlmostEqual(execution_summary["avg_response_seconds"], 0.833333)
        self.assertAlmostEqual(execution_summary["max_response_seconds"], 1.5)
        self.assertEqual(execution_summary["usage"]["interaction"]["call_count"], 3)
        self.assertAlmostEqual(
            execution_summary["usage"]["interaction"]["cost_usd"], 0.7
        )
        self.assertEqual(execution_summary["usage"]["interaction"]["total_tokens"], 63)
        self.assertEqual(execution_summary["instance_outcomes"], outcomes)
        self.assertEqual(trace_data["result"]["instance_outcomes"], outcomes)

    def test_parallel_baseline_merge_wraps_claude_system_artifacts(self):
        import src.systems.claude  # noqa: F401

        rel = ".claude/projects/-workspace/sess_0.jsonl"
        instance_results = [
            (
                0,
                TaskResult(
                    metrics={},
                    summary="inst-0",
                    eval_metrics=EvalMetrics(
                        loss_curve=[0.0],
                        optimal_performance=1.0,
                        actual_performance=1.0,
                    ),
                    instance_outcomes=[
                        InstanceOutcome(
                            instance_id="inst-0",
                            instance_index=0,
                            reward=1.0,
                        )
                    ],
                ),
                {
                    "interactions": [],
                    "instance_outcomes": [
                        {"instance_id": "inst-0", "instance_index": 0, "reward": 1.0}
                    ],
                    "execution": {"run_index": 0},
                    "task_brief": None,
                    "system": {"name": "claude", "params": {}},
                    "artifacts": {"type": "claude", "jsonl_event_count": 1},
                    "system_artifacts": {
                        "artifact_type": "claude",
                        "conversation_id": "sess_0",
                        "claude_conversation_session_ids": ["sess_0"],
                        "claude_conversation_jsonl_files": {rel: "main\n"},
                        "jsonl_events": [{"type": "result"}],
                        "memory_history": [],
                        "memory_files": {},
                    },
                },
            )
        ]

        _, trace_data, _ = _merge_baseline_instance_results(
            instance_results=instance_results,
            run_group_id="group123",
            system_name="claude",
            task_name="dummy_task",
            task_params={},
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:10",
        )

        system_artifacts = trace_data["system_artifacts"]
        self.assertEqual(system_artifacts["artifact_type"], "claude")
        self.assertEqual(system_artifacts["phase"], "baseline")
        self.assertEqual(len(system_artifacts["baseline_instances"]), 1)
        self.assertEqual(
            system_artifacts["baseline_instances"][0]["system_artifacts"][
                "claude_conversation_jsonl_files"
            ],
            {rel: "main\n"},
        )

    def test_parallel_baseline_merge_wraps_codex_system_artifacts(self):
        import src.systems.codex  # noqa: F401

        instance_results = [
            (
                0,
                TaskResult(
                    metrics={},
                    summary="inst-0",
                    eval_metrics=EvalMetrics(
                        loss_curve=[0.0],
                        optimal_performance=1.0,
                        actual_performance=1.0,
                    ),
                    instance_outcomes=[
                        InstanceOutcome(
                            instance_id="inst-0",
                            instance_index=0,
                            reward=1.0,
                        )
                    ],
                ),
                {
                    "interactions": [],
                    "instance_outcomes": [
                        {"instance_id": "inst-0", "instance_index": 0, "reward": 1.0}
                    ],
                    "execution": {"run_index": 0},
                    "task_brief": None,
                    "system": {"name": "codex", "params": {}},
                    "artifacts": {"type": "codex", "jsonl_event_count": 1},
                    "system_artifacts": {
                        "artifact_type": "codex",
                        "conversation_id": "conv_0",
                        "jsonl_events": [{"type": "turn.completed"}],
                        "memory_history": [],
                        "memory_files": {"note.md": "codex note"},
                    },
                },
            )
        ]

        _, trace_data, _ = _merge_baseline_instance_results(
            instance_results=instance_results,
            run_group_id="group123",
            system_name="codex",
            task_name="dummy_task",
            task_params={},
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:10",
        )

        system_artifacts = trace_data["system_artifacts"]
        self.assertEqual(system_artifacts["artifact_type"], "codex")
        self.assertEqual(system_artifacts["phase"], "baseline")
        self.assertEqual(len(system_artifacts["baseline_instances"]), 1)
        self.assertEqual(
            system_artifacts["baseline_instances"][0]["system_artifacts"][
                "memory_files"
            ],
            {"note.md": "codex note"},
        )

    def test_parallel_baseline_merge_uses_registered_artifact_baseline_builder(self):
        def build_baseline(trace_payloads):
            wrapped = []
            for trace in trace_payloads:
                artifacts = trace.get("system_artifacts")
                if isinstance(artifacts, dict):
                    wrapped.append({"system_artifacts": artifacts})
            return {
                "artifact_type": "custom_baseline",
                "phase": "baseline",
                "baseline_instances": wrapped,
            }

        register_artifact_exporter(
            "custom_baseline",
            build_manifest=lambda artifacts: {"type": "custom_baseline"},
            save=lambda artifacts, trace_path: None,
            build_baseline=build_baseline,
        )
        instance_results = [
            (
                0,
                TaskResult(
                    metrics={},
                    summary="inst-0",
                    eval_metrics=EvalMetrics(
                        loss_curve=[0.0],
                        optimal_performance=1.0,
                        actual_performance=1.0,
                    ),
                    instance_outcomes=[
                        InstanceOutcome(
                            instance_id="inst-0",
                            instance_index=0,
                            reward=1.0,
                        )
                    ],
                ),
                {
                    "interactions": [],
                    "instance_outcomes": [
                        {"instance_id": "inst-0", "instance_index": 0, "reward": 1.0}
                    ],
                    "execution": {"run_index": 0},
                    "task_brief": None,
                    "system": {"name": "custom_baseline", "params": {}},
                    "artifacts": {},
                    "system_artifacts": {
                        "artifact_type": "custom_baseline",
                        "debug": "wrapped by registry",
                    },
                },
            )
        ]

        _, trace_data, _ = _merge_baseline_instance_results(
            instance_results=instance_results,
            run_group_id="group123",
            system_name="custom_baseline",
            task_name="dummy_task",
            task_params={},
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:00:10",
        )

        self.assertEqual(
            trace_data["system_artifacts"],
            {
                "artifact_type": "custom_baseline",
                "phase": "baseline",
                "baseline_instances": [
                    {
                        "system_artifacts": {
                            "artifact_type": "custom_baseline",
                            "debug": "wrapped by registry",
                        }
                    }
                ],
            },
        )


class BaselineContractTests(unittest.TestCase):
    def test_default_reset_baseline_instance_uses_prepopulated_instances(self):
        task = PrePopulatedBaselineTask(num_instances=2)
        query = task.reset_baseline_instance(1)
        self.assertEqual(query.prompt, "instance 1")
        self.assertEqual(query.instance_id, "instance 1")
        self.assertEqual(task.instances, ["instance 1"])

    def test_tasks_without_instance_list_must_override_selection(self):
        task = PostResetBaselineTask(num_instances=2)
        with self.assertRaises(NotImplementedError):
            task.reset_baseline_instance(1)

    def test_baseline_worker_uses_lifecycle_selection(self):
        _, _, trace_data = _run_baseline_instance(
            task_class=ExplicitBaselineTask,
            task_params={"num_instances": 2},
            system_class=DummySystem,
            system_params={},
            instance_index=1,
            run_group_id="group1234",
            system_name="dummy_system",
            task_name="explicit_baseline",
        )
        self.assertEqual(trace_data["interactions"][0]["query"]["prompt"], "instance 1")
        self.assertEqual(trace_data["instance_outcomes"][0]["instance_id"], "1")


class ParallelSafetyTests(unittest.TestCase):
    def test_parallel_unsafe_runs_stay_in_main_process(self):
        with patch(
            "src.runs.ProcessPoolExecutor", side_effect=AssertionError("no pool")
        ):
            results, trace_data = run_benchmark_runs(
                task_class=UnsafeDummyTask,
                task_params={},
                system_class=DummySystem,
                system_params={},
                runs=2,
                max_workers=2,
                system_name="dummy_system",
                task_name="dummy",
            )
        self.assertEqual(len(results.results), 2)
        self.assertEqual(len(trace_data), 2)


# ---------------------------------------------------------------------------
# Helpers for integration tests that let real I/O happen in a temp directory
# ---------------------------------------------------------------------------


class _TmpCwdTestCase(unittest.TestCase):
    """Mixin that switches CWD to a temp directory for each test."""

    def setUp(self):
        self._orig_cwd = os.getcwd()
        self._tmpdir = tempfile.mkdtemp()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class ParallelBenchmarkIntegrationTests(_TmpCwdTestCase):
    def test_parallel_baseline_merges_instance_traces_and_live_snapshot(self):
        live_path = os.path.join(self._tmpdir, "baseline_live.json")

        run_index, result, trace_data, execution_summary = run_baseline(
            task_class=CountingInstancesTask,
            task_params={"num_instances": 3},
            system_class=DummySystem,
            system_params={},
            max_workers=2,
            run_group_id="group1234",
            system_name="dummy_system",
            task_name="counting_instances",
            live_trace_path=live_path,
        )

        self.assertIsNone(run_index)
        self.assertAlmostEqual(result.score, 2.0)
        self.assertEqual(len(trace_data["instance_traces"]), 3)
        self.assertEqual(len(trace_data["instance_outcomes"]), 3)
        self.assertEqual(trace_data["status"], "completed")
        self.assertEqual(
            [outcome["instance_id"] for outcome in trace_data["instance_outcomes"]],
            ["0", "1", "2"],
        )
        self.assertEqual(len(execution_summary["instance_outcomes"]), 3)

        with open(live_path) as f:
            live_payload = json.load(f)
        self.assertEqual(live_payload["status"], "completed")
        self.assertEqual(len(live_payload["instance_traces"]), 3)

    def test_baseline_records_blocked_provider_refusal_and_continues(self):
        live_path = os.path.join(self._tmpdir, "baseline_live.json")

        run_index, result, trace_data, execution_summary = run_baseline(
            task_class=CountingInstancesTask,
            task_params={"num_instances": 3},
            system_class=RefusalOnInstanceSystem,
            system_params={"blocked_instance_index": 1},
            max_workers=2,
            run_group_id="group1234",
            system_name="refusal_on_instance",
            task_name="counting_instances",
            live_trace_path=live_path,
        )

        self.assertIsNone(run_index)
        self.assertAlmostEqual(result.score, 2.0)
        # Runner now creates a fallback response on content policy violations so
        # all instances complete; status is "completed" rather than "completed_with_gaps".
        self.assertEqual(trace_data["status"], "completed")
        self.assertEqual(execution_summary["status"], "completed")
        self.assertEqual(execution_summary["instances_total"], 3)
        self.assertEqual(execution_summary["instances_completed"], 3)
        self.assertEqual(execution_summary["instances_blocked"], 0)
        self.assertAlmostEqual(execution_summary["coverage_ratio"], 1.0, places=6)
        self.assertEqual(
            [outcome["instance_id"] for outcome in trace_data["instance_outcomes"]],
            ["0", "1", "2"],
        )
        self.assertEqual(len(trace_data["instance_traces"]), 3)
        self.assertEqual(trace_data["blocked_instances"], [])

        with open(live_path) as f:
            live_payload = json.load(f)
        self.assertEqual(live_payload["status"], "completed")
        self.assertEqual(live_payload["execution"]["instances_blocked"], 0)

    def test_rollout_records_blocked_provider_refusal_and_continues(self):
        baseline_live_path = os.path.join(self._tmpdir, "baseline_live_block.json")
        run_live_paths = {
            0: os.path.join(self._tmpdir, "run_0_live_block.json"),
            1: os.path.join(self._tmpdir, "run_1_live_block.json"),
        }

        baseline_info, task_results, run_trace_data_list = run_benchmark(
            task_class=CountingInstancesTask,
            task_params={"num_instances": 3},
            system_class=RefusalOnInstanceSystem,
            system_params={"blocked_instance_index": 1},
            runs=2,
            max_workers=4,
            system_name="refusal_on_instance",
            task_name="counting_instances",
            baseline_live_path=baseline_live_path,
            live_trace_paths=run_live_paths,
        )

        self.assertIsNotNone(baseline_info)
        self.assertEqual(len(task_results.results), 2)
        self.assertEqual(len(run_trace_data_list), 2)

        # Runner now creates a fallback response on content policy violations so
        # runs complete normally rather than being marked blocked.
        statuses = [trace["status"] for trace in run_trace_data_list]
        self.assertEqual(statuses, ["completed", "completed"])
        for trace in run_trace_data_list:
            self.assertNotIn("blocked_reason", trace)

        # All instances (including the refused one) appear in each run's outcomes.
        for trace in run_trace_data_list:
            outcomes = trace.get("instance_outcomes", [])
            self.assertEqual(len(outcomes), 3)
            instance_ids = [o["instance_id"] for o in outcomes]
            self.assertIn("1", instance_ids)

        # Live snapshots reflect completed status.
        for path in run_live_paths.values():
            with open(path) as f:
                payload = json.load(f)
            self.assertEqual(payload["status"], "completed")

    def test_run_benchmark_parallel_path_writes_baseline_and_run_snapshots(self):
        baseline_live_path = os.path.join(self._tmpdir, "baseline_live.json")
        run_live_paths = {
            0: os.path.join(self._tmpdir, "run_0_live.json"),
            1: os.path.join(self._tmpdir, "run_1_live.json"),
        }

        baseline_info, task_results, run_trace_data_list = run_benchmark(
            task_class=CountingInstancesTask,
            task_params={"num_instances": 2},
            system_class=DummySystem,
            system_params={},
            runs=2,
            max_workers=3,
            system_name="dummy_system",
            task_name="counting_instances",
            baseline_live_path=baseline_live_path,
            live_trace_paths=run_live_paths,
        )

        self.assertIsNotNone(baseline_info)
        assert baseline_info is not None
        _, baseline_result, baseline_trace_data, baseline_execution = baseline_info

        self.assertAlmostEqual(baseline_result.score, 1.5)
        self.assertEqual(len(baseline_trace_data["instance_traces"]), 2)
        self.assertEqual(len(baseline_execution["instance_outcomes"]), 2)
        self.assertEqual(len(task_results.results), 2)
        self.assertEqual(len(run_trace_data_list), 2)
        self.assertEqual(
            [trace["status"] for trace in run_trace_data_list],
            ["completed", "completed"],
        )

        with open(baseline_live_path) as f:
            baseline_live_payload = json.load(f)
        self.assertEqual(baseline_live_payload["status"], "completed")

        for path in run_live_paths.values():
            with open(path) as f:
                run_live_payload = json.load(f)
            self.assertEqual(run_live_payload["status"], "completed")


def _read_seeds_from_trace_data(trace_data_list) -> list[int]:
    """Extract the seed param recorded in each trace data dict."""
    return [trace["task"]["params"]["seed"] for trace in trace_data_list]


# ---------------------------------------------------------------------------
# Integration tests: rollout modes (verify seeds via trace data)
# ---------------------------------------------------------------------------


class RolloutModeIntegrationTests(_TmpCwdTestCase):
    def test_replicate_all_same_seed(self):
        _, trace_data_list = run_benchmark_runs(
            task_class=DummyTask,
            task_params={"seed": 10, "variant": "v1"},
            system_class=DummySystem,
            system_params={},
            runs=3,
            max_workers=1,
            system_name="dummy_system",
            task_name="dummy",
            run_mode=RunMode.REPLICATE,
        )
        seeds = _read_seeds_from_trace_data(trace_data_list)
        self.assertEqual(sorted(seeds), [10, 10, 10])

    def test_resample_preserves_seed_across_rollouts(self):
        _, trace_data_list = run_benchmark_runs(
            task_class=DummyTask,
            task_params={"seed": 10, "variant": "v1"},
            system_class=DummySystem,
            system_params={},
            runs=3,
            max_workers=1,
            system_name="dummy_system",
            task_name="dummy",
            run_mode=RunMode.RESAMPLE,
        )
        seeds = _read_seeds_from_trace_data(trace_data_list)
        self.assertEqual(sorted(seeds), [10, 10, 10])


# ---------------------------------------------------------------------------
# Basic rollout tests
# ---------------------------------------------------------------------------


class RunRolloutsBasicTests(_TmpCwdTestCase):
    def test_single_rollout(self):
        task_results, trace_data_list = run_benchmark_runs(
            task_class=DummyTask,
            task_params={"seed": 42, "variant": "v1"},
            system_class=DummySystem,
            system_params={},
            runs=1,
            max_workers=1,
            system_name="dummy_system",
            task_name="dummy",
        )
        self.assertEqual(len(task_results.results), 1)
        self.assertAlmostEqual(task_results.results[0].score, 0.5)
        self.assertEqual(len(trace_data_list), 1)

    def test_multi_rollout(self):
        task_results, trace_data_list = run_benchmark_runs(
            task_class=DummyTask,
            task_params={"seed": 42, "variant": "v1"},
            system_class=DummySystem,
            system_params={},
            runs=3,
            max_workers=2,
            system_name="dummy_system",
            task_name="dummy",
        )
        self.assertEqual(len(task_results.results), 3)
        self.assertEqual(len(trace_data_list), 3)
        agg = task_results.aggregate()
        self.assertIn("score", agg)
        self.assertAlmostEqual(agg["score"]["mean"], 0.5)

    def test_run_group_id_assigned(self):
        task_results, _ = run_benchmark_runs(
            task_class=DummyTask,
            task_params={"seed": 42, "variant": "v1"},
            system_class=DummySystem,
            system_params={},
            runs=2,
            max_workers=1,
            system_name="dummy_system",
            task_name="dummy",
        )
        self.assertIsNotNone(task_results.run_group_id)
        self.assertRegex(
            task_results.run_group_id,
            r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}Z",
        )

    def test_rollout_failure_message_uses_one_based_numbering(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_benchmark_runs(
                task_class=FailingTask,
                task_params={"seed": 42, "variant": "v1"},
                system_class=DummySystem,
                system_params={},
                runs=3,
                max_workers=1,
                system_name="dummy_system",
                task_name="dummy",
                run_mode=RunMode.RESAMPLE,
            )

        message = str(ctx.exception)
        self.assertIn("Run 3/3 failed", message)
        self.assertNotIn("Run 2 failed", message)


if __name__ == "__main__":
    unittest.main()
