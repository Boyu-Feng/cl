"""Boundary checks for exact-evidence precedence across action schemas."""

import json
import unittest

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v10 import TrainedActor
from ttcl.trajectory_hyperlora.train_clbench_cumulative_hyperlora_v4 import merge_object_arrays


class MixedMemoryTest(unittest.TestCase):
    def test_array_evidence_applies_only_to_matching_public_schema(self):
        actor = object.__new__(TrainedActor)
        actor.historical_actions = [{"transmitters": [{"center_freq": 53.7}]}]

        def messages(properties):
            return [{"role": "user", "content":
                "current state\n\nReturn only JSON. Action schema:\n" +
                json.dumps({"type": "object", "properties": properties})}]

        self.assertTrue(actor._has_exact_array_evidence(messages({
            "transmitters": {"type": "array", "items": {"type": "object"}}})))
        self.assertFalse(actor._has_exact_array_evidence(messages({
            "action": {"type": "string"}})))
        self.assertFalse(actor._has_exact_array_evidence(messages({
            "transmitters": {"type": "string"}})))

    def test_array_merge_preserves_current_scalars_and_deduplicates_history(self):
        current = {"action": "REPORT", "transmitters": [{"center_freq": 53.7}]}
        history = [{"action": "OLD", "transmitters": [
            {"center_freq": 53.7}, {"center_freq": 101.2}]}]
        result = merge_object_arrays(current, history)
        self.assertEqual(result, {"action": "REPORT", "transmitters": [
            {"center_freq": 53.7}, {"center_freq": 101.2}]})
        self.assertEqual(len(current["transmitters"]), 1)


if __name__ == "__main__":
    unittest.main()
