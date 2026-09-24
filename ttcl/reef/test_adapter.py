"""Verify real Reef search accepts/rejects using official-runner scores."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from ttcl.reef.run_gepa import (
    Archive,
    EpisodeResult,
    EpisodeRunner,
    GEPAProposer,
    LocalBinding,
    ModelBindings,
    OfficialScorer,
    OfficialSelector,
    ScoreFeedback,
    UpdateCandidate,
    get_adapter,
    trajectory_item,
)


class FakeModel:
    def generate(self, messages, seed, **kwargs):
        return {"raw_response": "```\nCheck table schemas before querying.\n```"}


class FakeEpisodes(EpisodeRunner):
    def __init__(self, score):
        self.score = score
        self.contexts = []

    def run(self, descriptor, files, prompt, **kwargs):
        self.contexts.append(files["native/RULES.md"])
        return EpisodeResult(
            0,
            json.dumps({"reward": self.score}),
            "",
            ({"role": "assistant", "content": "public output"},),
            (),
        )

    def execute(self, index, context, label):
        self.contexts.append(context)
        return {"reward": self.score}, {}


class AdapterTests(unittest.TestCase):
    def test_rules_reach_first_official_actor_call(self):
        from ttcl.structured_memory.online_bank import BankSystem
        from pydantic import BaseModel

        class Action(BaseModel):
            action: str

        class CaptureModel:
            def generate(self, messages, seed):
                self.messages = messages
                raise StopIteration("captured before inference")

        with tempfile.TemporaryDirectory() as directory:
            model = CaptureModel()
            args = SimpleNamespace(
                task="database_exploration",
                memory_chars=16000,
                num_instances=1,
                seed=42,
                max_turns_per_instance=64,
                allow_initial_experience=True,
            )
            system = BankSystem(args, model, Path(directory), "", "check the schema")
            query = SimpleNamespace(
                prompt="Use permitted tools",
                instance_id="public-id",
                instance_index=0,
                response_schema=Action,
            )
            with self.assertRaises(StopIteration):
                system.respond(query)
            self.assertTrue(
                any("check the schema" in m["content"] for m in model.messages)
            )
            logged = json.loads((Path(directory) / "memory_contexts.jsonl").read_text())
            self.assertEqual(logged["memory_context"], "check the schema")

    def test_native_proposer_and_validation_selector(self):
        for score, expected in [(0.8, "select"), (0.2, "reject")]:
            with self.subTest(score=score), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = Archive(root / "archive.json")
                seed = archive.seed({"rules": ""})
                archive.record_validation(seed, [0.5])
                binding = LocalBinding(
                    base_url="http://unused",
                    model="fake",
                    backend=FakeModel(),
                    log_path=root / "calls.jsonl",
                )
                runner = FakeEpisodes(score)
                proposer = GEPAProposer(
                    archive=archive,
                    descriptor=get_adapter("native"),
                    binary=None,
                    score_episode=OfficialScorer(),
                    feedback=ScoreFeedback(),
                    minibatch_size=1,
                    rng_seed=42,
                    skip_perfect_score=True,
                    perfect_score=1.0,
                    max_metric_calls=None,
                    kinds=("rules",),
                    valset_size=1,
                    episode_runner=runner,
                )
                sample = trajectory_item(
                    "public question",
                    {"reward": 0.0},
                    {"steps": [{"public_feedback": "table exists"}]},
                )
                mutations = proposer(
                    (("rules", {"text": ""}),), (sample,), ModelBindings(binding)
                )
                self.assertIsNotNone(mutations)
                self.assertIn("Check table schemas", runner.contexts[0])
                candidate = UpdateCandidate(str(archive.pending))
                selector = OfficialSelector(archive, runner, [3])
                decision = selector.decide(candidate, selector.evaluate(candidate))
                self.assertEqual(decision.outcome, expected)
                self.assertEqual(archive.served, 1 if expected == "select" else 0)
                self.assertIsNone(archive.pending)
                self.assertIn("table exists", (root / "calls.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
