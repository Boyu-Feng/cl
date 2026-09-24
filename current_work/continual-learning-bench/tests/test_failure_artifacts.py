import json
import os
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from src.interface import (
    ContinualLearningSystem,
    ContinualLearningTask,
    EvalMetrics,
    Observation,
    Query,
    Response,
    TaskResult,
    TaskStepResult,
)
from src.runs.single import run_single


class FailureAction(BaseModel):
    value: str = "ok"


class FailureTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self.current_index = 0

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self.current_index = 0

    def build_current_query(self) -> Query:
        return Query(
            prompt="fail please",
            response_schema=FailureAction,
            instance_id="inst-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        return TaskStepResult(
            observation=Observation(content="never"),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={},
            summary="never",
            eval_metrics=EvalMetrics(
                loss_curve=[], optimal_performance=0.0, actual_performance=0.0
            ),
        )

    @property
    def name(self) -> str:
        return "failure_task"


class FailingArtifactSystem(ContinualLearningSystem):
    def __init__(self):
        self.reset_called = False

    def respond(self, query: Query) -> Response:
        raise RuntimeError("system exploded")

    def reset(self) -> None:
        self.reset_called = True

    def get_run_artifacts(self) -> dict:
        return {"artifact_type": "custom", "debug": "saved on failure"}

    @property
    def name(self) -> str:
        return "failing_artifact_system"


class FailureArtifactTests(unittest.TestCase):
    def test_run_single_saves_failed_trace_and_artifacts_before_reraising(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    run_single(
                        task_class=FailureTask,
                        task_params={},
                        system_class=FailingArtifactSystem,
                        system_params={},
                        run_index=0,
                        run_group_id="failure-group",
                        system_name="failing_artifact_system",
                        task_name="failure_task",
                    )
            finally:
                os.chdir(old_cwd)

            self.assertIn("failure-group", str(ctx.exception))
            self.assertIn("RuntimeError", str(ctx.exception))
            traces = list(
                Path(tmpdir).glob(
                    "results/failure_task/failed_failure-group_run0_*.json"
                )
            )
            self.assertEqual(len(traces), 1)
            payload = json.loads(traces[0].read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error"]["error_type"], "RuntimeError")
            self.assertEqual(payload["artifacts"], {"type": "custom"})
            self.assertIsNone(payload["system_artifacts"])

            artifact_payload = (
                traces[0].parent / "artifacts" / traces[0].stem / "artifacts.json"
            )
            self.assertEqual(
                json.loads(artifact_payload.read_text())["debug"],
                "saved on failure",
            )


if __name__ == "__main__":
    unittest.main()
