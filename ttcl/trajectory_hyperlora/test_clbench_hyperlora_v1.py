"""Protocol checks for CLBench trajectory-to-LoRA training and online writes."""

import json
import unittest

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v3 import reward_gate
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v1 import clean_records
from ttcl.trajectory_hyperlora.train_clbench_hyperlora_v3 import executed_action


class CLBenchHyperLoRATest(unittest.TestCase):
    def test_positive_officially_unsuccessful_reward_can_seed_memory(self):
        self.assertTrue(reward_gate(.22, []))
        self.assertTrue(reward_gate(.23, [.22]))
        self.assertFalse(reward_gate(.20, [.22]))
        self.assertFalse(reward_gate(-1.0, []))
        self.assertFalse(reward_gate(0.0, []))

    def test_public_action_retains_executed_fields_without_thinking(self):
        episode = {"steps": [{"query": "state", "action": {
            "thinking": "discard this unexecuted chain", "action": "RAISE",
            "amount": 30}, "public_feedback": "rewarded"}]}
        records = clean_records(episode)
        self.assertEqual(json.loads(records[0]["action"]),
                         {"action": "RAISE", "amount": 30})
        self.assertEqual(records[0]["feedback"], "rewarded")

    def test_divergence_uses_executed_move_not_call_amount(self):
        self.assertEqual(executed_action({"action": "CALL", "amount": 5}),
                         executed_action({"action": "CALL", "amount": None,
                                          "thinking": "different text"}))
        self.assertNotEqual(executed_action({"action": "RAISE", "amount": 20}),
                            executed_action({"action": "RAISE", "amount": 40}))


if __name__ == "__main__":
    unittest.main()
