import json
import unittest

from pydantic import BaseModel

from src.interface import (
    ContinualLearningTask,
    InstanceOutcome,
    Observation,
    Query,
    EvalMetrics,
    Response,
    TaskStepResult,
    TaskResult,
    run_task,
)
from src.systems.ace import ACESystem
from src.trace_storage import TraceRecorder
from src.vendors.ace.playbook_utils import (
    apply_curator_operations,
    get_next_global_id,
    normalize_section_name,
)


SEEDED_PLAYBOOK = """## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Check the required output schema before answering.

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


class ToyAction(BaseModel):
    value: int


class FakeGenerator:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(
        self,
        question,
        playbook,
        context="",
        reflection="(empty)",
        use_json_mode=False,
        call_id="gen",
        log_dir=None,
    ):
        self.calls.append(
            {
                "question": question,
                "context": context,
                "reflection": reflection,
                "use_json_mode": use_json_mode,
                "call_id": call_id,
            }
        )
        raw = self._responses.pop(0)
        bullet_ids = []
        try:
            payload = json.loads(raw)
            bullet_ids = payload.get("bullet_ids", [])
        except json.JSONDecodeError:
            pass
        return (
            raw,
            bullet_ids,
            {
                "role": "generator",
                "call_id": call_id,
                "model": "gpt-5",
                "provider": "openai",
                "prompt_num_tokens": 120,
                "response_num_tokens": 40,
                "total_num_tokens": 160,
            },
        )


class FakeReflector:
    def __init__(self, bullet_tags=None):
        self.calls = []
        self.bullet_tags = bullet_tags or [{"id": "str-00001", "tag": "helpful"}]

    def reflect(self, **kwargs):
        self.calls.append(kwargs)
        response = json.dumps(
            {
                "reasoning": "Reflection summary",
                "error_identification": "none",
                "root_cause_analysis": "none",
                "correct_approach": "follow schema",
                "key_insight": "Return valid JSON actions.",
                "bullet_tags": self.bullet_tags,
            }
        )
        return (
            response,
            list(self.bullet_tags),
            {
                "role": "reflector",
                "call_id": kwargs.get("call_id", "reflect"),
                "model": "gpt-5",
                "provider": "openai",
                "prompt_num_tokens": 80,
                "response_num_tokens": 30,
                "total_num_tokens": 110,
            },
        )


class FakeCurator:
    def __init__(self, operations=None):
        self.calls = []
        self.operations = operations or [
            {
                "type": "ADD",
                "section": "others",
                "content": "If the task provides a schema, keep the final answer as a JSON object.",
            }
        ]

    def curate(self, **kwargs):
        self.calls.append(kwargs)
        current_playbook = kwargs["current_playbook"]
        next_global_id = kwargs["next_global_id"]
        updated_playbook, next_global_id = apply_curator_operations(
            current_playbook, self.operations, next_global_id
        )
        return (
            updated_playbook,
            next_global_id,
            list(self.operations),
            {
                "role": "curator",
                "call_id": kwargs.get("call_id", "curate"),
                "model": "gpt-5",
                "provider": "openai",
                "prompt_num_tokens": 90,
                "response_num_tokens": 35,
                "total_num_tokens": 125,
            },
        )


class StubACESystem(ACESystem):
    def __init__(
        self,
        generator_responses,
        reflector_tags=None,
        curator_operations=None,
        curate_every_n_updates=5,
    ):
        super().__init__(
            model="stub-model",
            generator_model="stub-generator",
            reflector_model="stub-reflector",
            curator_model="stub-curator",
            api_provider="openai",
            curate_every_n_updates=curate_every_n_updates,
            name="ace_stub",
        )
        self._seeded_playbook = SEEDED_PLAYBOOK
        self._initial_playbook = self._seeded_playbook
        self.playbook = self._seeded_playbook
        self.next_global_id = get_next_global_id(self.playbook)
        self._generator_responses = list(generator_responses)
        self._reflector_tags = reflector_tags
        self._curator_operations = curator_operations

    def _ensure_initialized(self):
        if self._generator and self._reflector and self._curator:
            return
        self._generator = FakeGenerator(self._generator_responses)
        self._reflector = FakeReflector(self._reflector_tags)
        self._curator = FakeCurator(self._curator_operations)

    def _repair_generator_response(
        self,
        query: Query,
        schema_json: str,
        base_context: str,
        raw_response: str,
        validation_error: str,
    ) -> str:
        self._generator.calls.append(
            {
                "question": query.prompt,
                "context": base_context,
                "validation_error": validation_error,
                "raw_response": raw_response,
                "call_id": f"bench_respond_{self.interaction_count}_repair",
                "repair": True,
            }
        )
        return self._generator._responses.pop(0)

    def reset(self) -> None:
        super().reset()
        self.playbook = self._seeded_playbook
        self.next_global_id = get_next_global_id(self.playbook)


class OneStepTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self.value = None

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self.value = None

    def build_current_query(self) -> Query:
        return Query(
            prompt="Return a value.",
            response_schema=ToyAction,
            instance_id="episode-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        self.value = response.action.value
        return TaskStepResult(
            observation=Observation(
                content="Accepted.", metadata={"instance_complete": True}
            ),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        return TaskResult(
            metrics={"value": self.value},
            summary="Finished.",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0],
                optimal_performance=float(self.value or 0),
                actual_performance=float(self.value or 0),
            ),
            instance_outcomes=[
                InstanceOutcome(
                    instance_id="episode-0",
                    instance_index=0,
                    reward=float(self.value or 0),
                )
            ],
        )

    @property
    def name(self) -> str:
        return "one_step_task"


class TwoStepTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self._step = 0
        self.values: list[int] = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self._step = 0
        self.values = []

    def build_current_query(self) -> Query:
        return Query(
            prompt="First turn.",
            response_schema=ToyAction,
            instance_id="episode-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        self.values.append(response.action.value)
        if self._step == 0:
            self._step = 1
            return TaskStepResult(
                observation=Observation(
                    content="Intermediate environment feedback.",
                    metadata={"instance_complete": False},
                ),
                next_query=Query(
                    prompt="Second turn.",
                    response_schema=ToyAction,
                    instance_id="episode-0",
                    instance_index=0,
                ),
                done=False,
            )
        return TaskStepResult(
            observation=Observation(
                content="Episode complete.", metadata={"instance_complete": True}
            ),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        score = float(sum(self.values))
        return TaskResult(
            metrics={"values": list(self.values)},
            summary="Finished.",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0 for _ in self.values],
                optimal_performance=score,
                actual_performance=score,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="episode-0", instance_index=0, reward=score)
            ],
        )

    @property
    def name(self) -> str:
        return "two_step_task"


class TwoEpisodeTask(ContinualLearningTask):
    def __init__(self, num_instances: int = 1):
        self.num_instances = num_instances
        self.instances = list(range(num_instances))
        self._step = 0
        self.values: list[int] = []

    def build_canonical_run_state(self) -> None:
        self.instances = list(range(self.num_instances))
        self._step = 0
        self.values = []

    def build_current_query(self) -> Query:
        return Query(
            prompt="Episode one.",
            response_schema=ToyAction,
            instance_id="episode-0",
            instance_index=0,
        )

    def step(self, response: Response) -> TaskStepResult:
        self.values.append(response.action.value)
        if self._step == 0:
            self._step = 1
            return TaskStepResult(
                observation=Observation(
                    content="Episode one complete.",
                    metadata={"instance_complete": True},
                ),
                next_query=Query(
                    prompt="Episode two.",
                    response_schema=ToyAction,
                    instance_id="episode-1",
                    instance_index=1,
                ),
                done=False,
            )
        return TaskStepResult(
            observation=Observation(
                content="Episode two complete.", metadata={"instance_complete": True}
            ),
            next_query=None,
            done=True,
        )

    def evaluate(self) -> TaskResult:
        score = float(sum(self.values))
        return TaskResult(
            metrics={"values": list(self.values)},
            summary="Finished.",
            eval_metrics=EvalMetrics(
                loss_curve=[0.0 for _ in self.values],
                optimal_performance=score,
                actual_performance=score,
            ),
            instance_outcomes=[
                InstanceOutcome(instance_id="episode-0", instance_index=0, reward=score)
            ],
        )

    @property
    def name(self) -> str:
        return "two_episode_task"


class ACESystemTests(unittest.TestCase):
    def test_respond_returns_schema_valid_action_without_mutating_playbook(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "Use the schema.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 7},
                    }
                )
            ]
        )
        system.reset()
        playbook_before = system.playbook

        response = system.respond(
            Query(prompt="Return seven.", response_schema=ToyAction)
        )

        self.assertEqual(response.action.value, 7)
        self.assertEqual(system.playbook, playbook_before)
        self.assertEqual(len(system.episode_transcript), 1)
        self.assertEqual(response.metadata["playbook_bullet_count"], 1)
        self.assertEqual(response.metadata["repair_attempted"], False)

    def test_terminal_feedback_on_next_query_updates_playbook(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "Use the stored strategy.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 3},
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "Use the updated playbook.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 5},
                    }
                ),
            ],
            curate_every_n_updates=1,
        )
        system.reset()
        playbook_before = system.playbook

        system.respond(Query(prompt="Return three.", response_schema=ToyAction))
        self.assertEqual(system.playbook, playbook_before)

        result = system.respond(
            Query(
                prompt="Return five.",
                response_schema=ToyAction,
                feedback=Observation(
                    content="That worked.", metadata={"instance_complete": True}
                ),
            )
        )
        self.assertEqual(result.action.value, 5)

        self.assertNotEqual(system.playbook, playbook_before)
        self.assertIn("[str-00001] helpful=1 harmful=0", system.playbook)
        self.assertIn("[misc-00002] helpful=0 harmful=0", system.playbook)
        self.assertEqual(system.update_count, 1)
        self.assertTrue(system._last_update_reflection_ran)
        self.assertTrue(system._last_update_curation_ran)
        artifacts = system.get_run_artifacts()
        self.assertEqual(artifacts["artifact_type"], "ace")
        self.assertEqual(len(artifacts["playbook_snapshots"]), 1)
        self.assertEqual(artifacts["playbook_snapshots"][0]["update_count"], 1)
        self.assertTrue(artifacts["playbook_snapshots"][0]["curation_ran"])

    def test_multi_turn_episode_reflects_once_at_boundary(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "First answer.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 1},
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "Second answer.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 2},
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "New episode answer.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 7},
                    }
                ),
            ],
            curate_every_n_updates=1,
        )
        system.reset()

        system.respond(Query(prompt="First turn.", response_schema=ToyAction))
        system.respond(
            Query(
                prompt="Second turn.",
                response_schema=ToyAction,
                feedback=Observation(
                    content="Intermediate environment feedback.",
                    metadata={"instance_complete": False},
                ),
            )
        )
        self.assertEqual(len(system.episode_transcript), 2)
        self.assertIn(
            "Turn 1 observation after action:\nIntermediate environment feedback.",
            system._generator.calls[1]["context"],
        )

        result = system.respond(
            Query(
                prompt="New episode start.",
                response_schema=ToyAction,
                feedback=Observation(
                    content="Episode complete.", metadata={"instance_complete": True}
                ),
            )
        )
        self.assertEqual(result.action.value, 7)

        self.assertEqual(len(system.episode_transcript), 1)
        self.assertEqual(len(system._reflector.calls), 1)
        reflector_call = system._reflector.calls[0]
        reasoning_trace = reflector_call["reasoning_trace"]
        self.assertIn("First turn.", reasoning_trace)
        self.assertIn("Second turn.", reasoning_trace)
        self.assertIn(
            "Turn 1 intermediate observation:\nIntermediate environment feedback.",
            reasoning_trace,
        )
        self.assertIn("Turn 1 next decision prompt:\nSecond turn.", reasoning_trace)
        self.assertIn(
            "Turn 2 terminal observation:\nEpisode complete.", reasoning_trace
        )
        self.assertIn("Benchmark episode to analyze.", reflector_call["question"])
        self.assertIn(
            "Final decision prompt:\nSecond turn.", reflector_call["question"]
        )
        predicted_answer = json.loads(reflector_call["predicted_answer"])
        self.assertEqual(predicted_answer["episode_type"], "trajectory")
        self.assertEqual(len(predicted_answer["actions"]), 2)
        self.assertEqual(predicted_answer["actions"][0]["action"]["value"], 1)
        self.assertEqual(predicted_answer["actions"][1]["action"]["value"], 2)
        curator_call = system._curator.calls[0]
        self.assertIn("Trajectory:", curator_call["question_context"])
        self.assertIn(
            "Intermediate observation:\nIntermediate environment feedback.",
            curator_call["question_context"],
        )
        self.assertIn(
            "Final feedback:\nEpisode complete.", curator_call["question_context"]
        )

    def test_invalid_generator_output_retries_once_and_recovers(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "Broken output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": "not valid json",
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "Fixed output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 9},
                    }
                ),
            ]
        )
        system.reset()

        response = system.respond(
            Query(prompt="Return nine.", response_schema=ToyAction)
        )

        self.assertEqual(response.action.value, 9)
        self.assertTrue(response.metadata["repair_attempted"])
        self.assertEqual(len(system._generator.calls), 2)
        self.assertTrue(system.episode_transcript[0].repair_attempted)

    def test_non_json_generator_output_retries_once_and_recovers(self):
        system = StubACESystem(
            [
                "I think the answer should probably be value=5.",
                json.dumps(
                    {
                        "reasoning": "Fixed output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 5},
                    }
                ),
            ]
        )
        system.reset()

        response = system.respond(
            Query(prompt="Return five.", response_schema=ToyAction)
        )

        self.assertEqual(response.action.value, 5)
        self.assertTrue(response.metadata["repair_attempted"])
        self.assertEqual(len(system._generator.calls), 2)
        self.assertTrue(system._generator.calls[1]["repair"])

    def test_generator_response_accepts_embedded_json_text(self):
        system = StubACESystem(
            [
                """Here is the structured answer:
```json
{
  "reasoning": "Wrapped output.",
  "bullet_ids": ["str-00001"],
  "final_answer": {"value": 8}
}
```"""
            ]
        )
        system.reset()

        response = system.respond(
            Query(prompt="Return eight.", response_schema=ToyAction)
        )

        self.assertEqual(response.action.value, 8)
        self.assertFalse(response.metadata["repair_attempted"])
        self.assertEqual(len(system._generator.calls), 1)

    def test_runner_supports_ace_system(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "Worker-safe output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 4},
                    }
                )
            ]
        )
        task = OneStepTask()

        result = run_task(
            task,
            system,
            show_progress=False,
        )

        self.assertEqual(result.score, 4.0)
        self.assertEqual(task.value, 4)
        self.assertEqual(system.update_count, 1)

    def test_trace_recorder_records_terminal_update_artifacts(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "Worker-safe output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 4},
                    }
                )
            ],
            curate_every_n_updates=1,
        )
        task = OneStepTask()
        trace_recorder = TraceRecorder(
            system_name="ace",
            task_name=task.name,
            system_params={},
            task_params={},
        )

        result = run_task(
            task,
            system,
            trace_recorder=trace_recorder,
            show_progress=False,
        )

        self.assertEqual(result.score, 4.0)
        artifacts = trace_recorder.system_artifacts
        self.assertIsNotNone(artifacts)
        self.assertEqual(artifacts["artifact_type"], "ace")
        self.assertEqual(len(artifacts["playbook_snapshots"]), 1)
        self.assertIn("[misc-00002]", artifacts["final_playbook"])
        trace = trace_recorder.finalize(result)
        self.assertGreater(trace["execution"]["usage"]["interaction"]["call_count"], 0)
        self.assertGreater(trace["execution"]["usage"]["interaction"]["cost_usd"], 0.0)

    def test_runner_records_boundary_updates_before_next_query(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "Episode one answer.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 4},
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "Episode two answer.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 6},
                    }
                ),
            ],
            curate_every_n_updates=1,
        )
        task = TwoEpisodeTask()
        trace_recorder = TraceRecorder(
            system_name="ace",
            task_name=task.name,
            system_params={},
            task_params={},
        )

        result = run_task(
            task,
            system,
            trace_recorder=trace_recorder,
            show_progress=False,
        )

        self.assertEqual(result.score, 10.0)
        self.assertEqual(task.values, [4, 6])
        self.assertEqual(system.update_count, 2)
        artifacts = trace_recorder.system_artifacts
        self.assertIsNotNone(artifacts)
        self.assertEqual(len(artifacts["playbook_snapshots"]), 2)
        self.assertIn("[misc-00003]", artifacts["final_playbook"])

    def test_runner_preserves_intermediate_observations(self):
        system = StubACESystem(
            [
                json.dumps(
                    {
                        "reasoning": "First worker-safe output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 4},
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "Second worker-safe output.",
                        "bullet_ids": ["str-00001"],
                        "final_answer": {"value": 6},
                    }
                ),
            ],
            curate_every_n_updates=1,
        )
        task = TwoStepTask()

        result = run_task(
            task,
            system,
            show_progress=False,
        )

        self.assertEqual(result.score, 10.0)
        self.assertEqual(task.values, [4, 6])

    def test_section_normalization_handles_hyphen_and_underscore_variants(self):
        self.assertEqual(
            normalize_section_name("PROBLEM-SOLVING HEURISTICS"),
            "problem_solving_heuristics",
        )
        self.assertEqual(
            normalize_section_name("problem_solving_heuristics"),
            "problem_solving_heuristics",
        )


if __name__ == "__main__":
    unittest.main()
