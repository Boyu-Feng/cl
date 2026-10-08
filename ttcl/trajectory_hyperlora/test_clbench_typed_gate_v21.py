"""Regression: do not re-generate the frozen actor when no LoRA is active."""

import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import Actor
from ttcl.trajectory_hyperlora.clbench_online_typed_gate_v21 import TypedActor
from ttcl.trajectory_hyperlora.typed_evidence_v20 import FEATURES, extract


class FrozenCandidateReuseTest(TestCase):
    def test_one_generation_when_no_adapter_is_active(self):
        actor = object.__new__(TypedActor)
        actor.arm = "online"
        actor.active_adapter = False
        actor.args = SimpleNamespace(disable_events=False)
        actor.current_index = 2
        actor.gate_model = {"features": FEATURES,
                            "operators": {"numeric_mean": {}}}
        actor.evidence = extract({"y": 2}, source_index=1,
            trajectory_sha256="t", feedback="ok", reward=1, query="q")
        messages = [{"role": "user", "content": "qReturn only JSON. Action schema:\n" +
            json.dumps({"type": "object", "properties": {"x": {"type": "integer"}}})}]
        with patch.object(Actor, "generate",
                          return_value={"raw_response": '{"x": 1}'}) as generate:
            answer = actor.generate(messages, 42)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(json.loads(answer["raw_response"]), {"x": 1})
        self.assertEqual(answer["typed_gate"]["decision"], "lora")
