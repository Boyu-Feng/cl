"""Protocol checks for CLBench trajectory-to-parameter transfer."""

import unittest

from ttcl.trajectory_hyperlora.clbench_online_parameter_memory_v2 import (
    compact_actor_messages, public_records,
)


class CLBenchTransferTest(unittest.TestCase):
    def test_history_window_retains_initial_task_and_current_turn(self):
        messages = [{"role": "system", "content": "policy"},
            {"role": "user", "content": "initial task"}]
        for turn in range(5):
            messages.extend([{"role": "assistant", "content": f"action {turn}"},
                             {"role": "user", "content": f"feedback {turn}"}])
        compact = compact_actor_messages(messages, 2)
        self.assertEqual(compact[0], messages[0])
        self.assertEqual(compact[1], messages[1])
        self.assertEqual(compact[-1], messages[-1])
        self.assertNotIn("feedback 0", [row["content"] for row in compact])

    def test_only_public_step_fields_enter_lora_source(self):
        episode = {"steps": [{"query": "visible task", "action": {"a": 1},
            "public_feedback": "visible result", "hidden_label": "SECRET"}]}
        records = public_records(episode)
        self.assertEqual(len(records), 1)
        self.assertEqual(set(records[0]), {"observation", "action", "feedback"})
        self.assertNotIn("SECRET", str(records))


if __name__ == "__main__":
    unittest.main()
