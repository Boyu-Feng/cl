"""Replay old public actions through a mocked HTTP service and real task loops.

This tests plumbing only; replay rewards are NEVER research results.
"""

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ttcl.common.openrouter import APIStop, BankTokenizer, OpenRouter, write
from ttcl.openrouter_memory.run import execute, read
from ttcl.openrouter_memory.test_api import INFO

WORKSPACE = Path(__file__).resolve().parents[2]
OLD = WORKSPACE / "ttcl/results/structured_memory/llm_online_bank_20260920_v2"


class ReplayAPI(OpenRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, opener=self.response)
        self.turns = {}

    def response(self, request, **kwargs):
        data = json.loads(request.data)
        purpose = self.purpose
        if purpose["purpose"] == "writer":
            payload = json.loads(data["messages"][1]["content"])
            self_test_episode = payload["trajectory"]["episode"]
            assert self_test_episode == purpose["episode"]
            assert all(
                e["episode"] < self_test_episode
                for entry in payload["existing_bank"]
                for e in entry["evidence"]
            )
            entry = {
                "type": "hypothesis",
                "title": "Test memory",
                "scope": "Mock only",
                "lesson": "Mock lesson",
                "application": "Mock only",
                "limitations": "Not research",
                "evidence": [{"episode": self_test_episode, "steps": [1]}],
            }
            answer = {
                "trajectory_summary": "Mock test",
                "reward_interpretation": "Mock test",
                "decision": "UPDATE",
                "operations": [
                    {
                        "op": "ADD",
                        "id": f"E{self_test_episode}",
                        "reason": "Mock only",
                        "entry": entry,
                    }
                ],
            }
        else:
            identity = (purpose["task"], purpose["arm"], purpose["episode"])
            turn = self.turns.get(identity, 0)
            trajectory = read(
                OLD
                / purpose["task"]
                / "independent"
                / f"episode_{purpose['episode']:03d}"
                / "trajectory.json"
            )
            answer = trajectory["steps"][turn]["action"]
            self.turns[identity] = turn + 1
        return io.BytesIO(
            json.dumps(
                {
                    "id": "mock-response",
                    "model": data["model"],
                    "provider": "mock",
                    "choices": [
                        {
                            "message": {"content": json.dumps(answer)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "cost": 0.0,
                    },
                }
            ).encode()
        )


class FlowTests(unittest.TestCase):
    def test_two_tasks_two_episodes_real_loops_no_gpu_and_memory_only_after_first(self):
        if not OLD.exists():
            self.skipTest("Local public-trajectory fixtures unavailable")
        with tempfile.TemporaryDirectory(prefix="openrouter_mock_") as temp:
            root = Path(temp)
            info = copy.deepcopy(INFO)
            weak = dict(info, id="test/alternate-writer")
            plan = {
                "actor_model": info["id"],
                "writer_models": {
                    "online_bank": info["id"],
                    "alternate_writer": weak["id"],
                },
                "models": {info["id"]: info, weak["id"]: weak},
                "arms": ["independent", "online_bank", "alternate_writer"],
                "tasks": ["database_exploration", "cohort_studies"],
                "num_instances": 2,
                "seed": 42,
                "bank": {"max_entries": 8, "max_chars": 9000, "max_tokens": 2048},
                "bank_tokenizer": str(
                    WORKSPACE / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
                ),
                "budget": {"max_cost_usd": 50, "max_requests": 500},
                "api": {"output_tokens": 8192, "reasoning": "medium", "retries": 0},
                "writer_output_tokens": 8192,
                "actor": {
                    "memory_chars": 16000,
                    "max_turns_per_instance": 64,
                    "action_retries": 2,
                    "normalize_action": True,
                    "allow_initial_experience": True,
                },
            }
            write(root / "plan.json", plan)
            cwd = Path.cwd()
            try:
                with (
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
                    patch(
                        "ttcl.common.local_qwen.LocalQwen.__init__",
                        side_effect=AssertionError("GPU backend must never be loaded"),
                    ),
                ):
                    execute(
                        root,
                        "MOCK_SECRET",
                        models=[info, weak],
                        backend_class=ReplayAPI,
                    )
            finally:
                os.chdir(cwd)
            self.assertEqual(read(root / "status.json")["status"], "finished")
            for task in plan["tasks"]:
                self.assertEqual(read(root / task / "audit.json")["errors"], [])
                rows = read(root / task / "results.json")
                self.assertEqual(len(rows), 6)
                self.assertTrue(all(r["status"] == "complete" for r in rows))
                self.assertTrue(
                    all(r["bank_context_chars"] == 0 for r in rows if r["episode"] == 1)
                )
                self.assertTrue(
                    all(
                        r["bank_context_chars"] > 0
                        for r in rows
                        if r["episode"] == 2 and r["arm"] != "independent"
                    )
                )
                self.assertTrue(read(root / task / "comparison.json")["complete"])
            self.assertNotIn(
                "MOCK_SECRET",
                "".join(f.read_text() for f in root.rglob("*") if f.is_file()),
            )
            self.assertGreater(read(root / "api_usage.json")["http_requests"], 0)
            stopped = root / "budget_stop"
            stopped_plan = copy.deepcopy(plan)
            stopped_plan["budget"]["max_cost_usd"] = 0.00001
            write(stopped / "plan.json", stopped_plan)
            try:
                with (
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
                    self.assertRaises(APIStop),
                ):
                    execute(
                        stopped,
                        "MOCK_SECRET",
                        models=[info, weak],
                        backend_class=ReplayAPI,
                    )
            finally:
                os.chdir(cwd)
            self.assertEqual(
                read(stopped / "status.json")["status"], "stopped_api_or_budget"
            )
            failed = read(stopped / plan["tasks"][0] / "results.json")[0]
            self.assertIsNone(failed["reward"])
            self.assertEqual(failed["status"], "failed")

    def test_cpu_bank_tokenizer_matches_original_hf(self):
        from transformers import AutoTokenizer

        path = WORKSPACE / "current_work/delta-Mem/model/Qwen3-4B-Instruct-2507"
        cpu = BankTokenizer(path)
        original = AutoTokenizer.from_pretrained(path, local_files_only=True)
        for text in [
            "An experience bank with a hypothesis.",
            "经验与反馈",
            '{"step":1,"scope":"database"}',
        ]:
            self.assertEqual(
                cpu.encode(text), original.encode(text, add_special_tokens=False)
            )


if __name__ == "__main__":
    unittest.main()
