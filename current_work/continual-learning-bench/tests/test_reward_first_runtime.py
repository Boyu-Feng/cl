import unittest

from pydantic import BaseModel

from src.interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    InstanceOutcome,
    Observation,
    Query,
    EvalMetrics,
    Response,
    TaskResult,
    TaskStepResult,
    observation_marks_instance_complete,
    run_task,
)
from src.trace_storage import TraceRecorder


class DummyAction(BaseModel):
    value: str = "ok"


class TwoInstanceTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 2):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self.current_index = 0
        self.completed = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self.current_index = 0
        self.completed = []

    def build_current_query(self) -> Query:
        return self._query_for_index(0)

    def step(self, response: Response) -> TaskStepResult:
        outcome = InstanceOutcome(
            instance_id=f"instance-{self.current_index}",
            instance_index=self.current_index,
            reward=float(self.current_index + 1),
            success=True,
        )
        self.completed.append(outcome)
        self.current_index += 1
        done = self.current_index >= 2
        next_query = None if done else self._query_for_index(self.current_index)
        return TaskStepResult(
            observation=Observation(
                content=f"completed {outcome.instance_id}",
                instance_complete=True,
            ),
            next_query=next_query,
            done=done,
            instance_outcome=outcome,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[],
                optimal_performance=2.0,
                actual_performance=2.0,
            ),
            instance_outcomes=[],
        )

    @property
    def name(self) -> str:
        return "two_instance"

    def _query_for_index(self, index: int) -> Query:
        return Query(
            prompt=f"instance {index}",
            response_schema=DummyAction,
            instance_id=f"instance-{index}",
            instance_index=index,
        )


class AutoAdvanceTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 3):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self.current_index = 0
        self._instance_outcomes: list[InstanceOutcome] = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self.current_index = 0
        self._instance_outcomes = []

    def build_current_query(self) -> Query:
        return self._query_for_index(0)

    def step(self, response: Response) -> TaskStepResult:
        reported_outcome = self._append_outcome(self.current_index)
        self.current_index += 1

        if self.current_index == 1:
            reported_outcome = self._append_outcome(self.current_index)
            self.current_index += 1

        done = self.current_index >= self.num_instances
        next_query = None if done else self._query_for_index(self.current_index)
        return TaskStepResult(
            observation=Observation(
                content=f"completed {reported_outcome.instance_id}",
                instance_complete=True,
            ),
            next_query=next_query,
            done=done,
            instance_outcome=reported_outcome,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[],
                optimal_performance=float(self.num_instances),
                actual_performance=float(self.num_instances),
            ),
            instance_outcomes=list(self._instance_outcomes),
        )

    @property
    def name(self) -> str:
        return "auto_advance"

    def _append_outcome(self, index: int) -> InstanceOutcome:
        outcome = InstanceOutcome(
            instance_id=f"instance-{index}",
            instance_index=index,
            reward=float(index + 1),
            success=True,
        )
        self._instance_outcomes.append(outcome)
        return outcome

    def _query_for_index(self, index: int) -> Query:
        return Query(
            prompt=f"instance {index}",
            response_schema=DummyAction,
            instance_id=f"instance-{index}",
            instance_index=index,
        )


class TerminalOnResetTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self._instance_outcomes: list[InstanceOutcome] = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self._instance_outcomes = []

    def select_run_instances(self, instance_indices: list[int] | None) -> None:
        super().select_run_instances(instance_indices)
        canonical_index = self.canonical_instance_index(0)
        self._instance_outcomes = [
            InstanceOutcome(
                instance_id=f"instance-{canonical_index}",
                instance_index=canonical_index,
                reward=3.0,
                success=True,
            )
        ]

    def build_current_query(self) -> Query:
        outcome = self._instance_outcomes[-1]
        return Query(
            prompt="",
            response_schema=DummyAction,
            metadata={"done": True},
            instance_id=outcome.instance_id,
            instance_index=outcome.instance_index,
        )

    def step(self, response: Response) -> TaskStepResult:
        raise AssertionError("TerminalOnResetTask.step() should never be called.")

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[],
                optimal_performance=3.0,
                actual_performance=3.0,
            ),
            instance_outcomes=list(self._instance_outcomes),
        )

    @property
    def name(self) -> str:
        return "terminal_on_reset"


class CountingSystem(ContinualLearningSystem):
    def __init__(self):
        self.reset_calls = 0
        self.respond_calls = 0

    def respond(self, query: Query) -> Response:
        self.respond_calls += 1
        return Response(action=DummyAction())

    def observe(self, observation: Observation, next_query: Query | None) -> None:
        return None

    def reset(self) -> None:
        self.reset_calls += 1

    @property
    def name(self) -> str:
        return "counting"


class FailingOnceSystem(CountingSystem):
    def __init__(self):
        super().__init__()
        self.failed = False
        self.observations: list[str] = []

    def respond(self, query: Query) -> Response:
        self.respond_calls += 1
        if not self.failed:
            self.failed = True
            raise RuntimeError("LLM call failed: invalid structured output")
        return Response(action=DummyAction())

    def observe(self, observation: Observation, next_query: Query | None) -> None:
        self.observations.append(observation.content)


class ErrorRecoveringTask(TwoInstanceTask):
    @property
    def name(self) -> str:
        return "error_recovering"

    def handle_system_error(
        self,
        query: Query,
        error: Exception,
    ) -> TaskStepResult:
        outcome = InstanceOutcome(
            instance_id=str(query.instance_id),
            instance_index=int(query.instance_index),
            reward=0.0,
            success=False,
            metadata={
                "system_error": {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "zero_credit": True,
                }
            },
        )
        self.completed.append(outcome)
        self.current_index += 1
        done = self.current_index >= 2
        next_query = None if done else self._query_for_index(self.current_index)
        return TaskStepResult(
            observation=Observation(
                content="zero-credit system error",
                instance_complete=True,
            ),
            next_query=next_query,
            done=done,
            instance_outcome=outcome,
        )


class AnonymousQueryTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))

    def build_current_query(self) -> Query:
        return Query(prompt="missing identity", response_schema=DummyAction)

    def step(self, response: Response) -> TaskStepResult:
        return TaskStepResult(
            observation=Observation(content="done"),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="done",
            eval_metrics=EvalMetrics(
                loss_curve=[],
                optimal_performance=0.0,
                actual_performance=0.0,
            ),
        )

    @property
    def name(self) -> str:
        return "anonymous_query"


class RewardFirstRuntimeTests(unittest.TestCase):
    def test_observation_instance_complete_uses_top_level_and_legacy_fallback(self):
        top_level_false = Observation(
            content="x",
            instance_complete=False,
            metadata={"instance_complete": True},
        )
        legacy_false = Observation(
            content="x",
            instance_complete=True,
            metadata={"instance_complete": False},
        )

        self.assertFalse(observation_marks_instance_complete(top_level_false))
        self.assertFalse(observation_marks_instance_complete(legacy_false))

    def test_run_task_resets_between_instances_for_baseline_mode(self):
        task = TwoInstanceTask()
        system = CountingSystem()

        run_task(task, system, reset_between_instances=True)

        self.assertEqual(system.reset_calls, 2)

    def test_trace_recorder_preserves_streamed_instance_outcomes(self):
        task = TwoInstanceTask()
        system = CountingSystem()
        recorder = TraceRecorder(
            system_name="counting",
            task_name="two_instance",
            system_params={},
            task_params={},
        )

        result = run_task(task, system, trace_recorder=recorder, show_progress=False)
        payload = recorder.finalize(result)

        self.assertEqual(len(result.instance_outcomes), 2)
        self.assertEqual(result.instance_outcomes[0].instance_id, "instance-0")
        self.assertEqual(result.instance_outcomes[1].instance_index, 1)
        self.assertEqual(len(payload["instance_outcomes"]), 2)
        self.assertEqual(payload["instance_outcomes"][0]["instance_id"], "instance-0")
        self.assertEqual(payload["instance_outcomes"][1]["instance_index"], 1)

    def test_run_task_syncs_multiple_task_outcomes_per_step(self):
        task = AutoAdvanceTask()
        system = CountingSystem()
        recorder = TraceRecorder(
            system_name="counting",
            task_name="auto_advance",
            system_params={},
            task_params={},
        )

        run_task(task, system, trace_recorder=recorder, show_progress=False)

        self.assertEqual(
            [outcome.instance_id for outcome in recorder.instance_outcomes],
            ["instance-0", "instance-1", "instance-2"],
        )

    def test_run_task_skips_system_when_initial_query_is_terminal(self):
        task = TerminalOnResetTask()
        system = CountingSystem()
        recorder = TraceRecorder(
            system_name="counting",
            task_name="terminal_on_reset",
            system_params={},
            task_params={},
        )

        result = run_task(task, system, trace_recorder=recorder, show_progress=False)
        payload = recorder.finalize(result)

        self.assertEqual(system.reset_calls, 1)
        self.assertEqual(system.respond_calls, 0)
        self.assertEqual(len(result.instance_outcomes), 1)
        self.assertEqual(result.instance_outcomes[0].instance_id, "instance-0")
        self.assertEqual(payload["instance_outcomes"][0]["instance_id"], "instance-0")

    def test_run_task_requires_query_instance_identity(self):
        task = AnonymousQueryTask()
        system = CountingSystem()

        with self.assertRaisesRegex(
            ValueError, "AnonymousQueryTask.reset\\(\\) must set Query.instance_id"
        ):
            run_task(task, system, show_progress=False)

    def test_run_task_scores_recoverable_llm_error_as_zero_credit(self):
        task = ErrorRecoveringTask()
        system = FailingOnceSystem()
        recorder = TraceRecorder(
            system_name="failing_once",
            task_name="error_recovering",
            system_params={},
            task_params={},
        )

        result = run_task(task, system, trace_recorder=recorder, show_progress=False)
        payload = recorder.finalize(result)

        self.assertEqual([o.reward for o in result.instance_outcomes], [0.0, 2.0])
        self.assertFalse(result.instance_outcomes[0].success)
        self.assertEqual(
            result.instance_outcomes[0].metadata["system_error"]["error_type"],
            "RuntimeError",
        )
        self.assertIn("zero-credit system error", system.observations)
        self.assertEqual(
            payload["interactions"][0]["response"]["metadata"]["llm_error"][
                "zero_credit"
            ],
            True,
        )


if __name__ == "__main__":
    unittest.main()
