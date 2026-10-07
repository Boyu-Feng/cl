"""State-update checks for contextual online parameter memory."""

import unittest

import torch

from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import (
    update_fields, vector_hash,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return type("Tokens", (), {"input_ids": text.split()})()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(ids)


class ContextMemoryTest(unittest.TestCase):
    def test_source_vector_running_mean_is_persistent(self):
        first = {"contextual": torch.ones(1, 2),
                 "pair_contextual": torch.ones(1, 2)}
        state = update_fields(None, first, 0)
        next_state = update_fields(state, {
            "contextual": torch.full((1, 2), 3.),
            "pair_contextual": torch.full((1, 2), 5.)}, 1)
        self.assertTrue(torch.equal(next_state["contextual"],
                                    torch.full((1, 2), 2.)))
        self.assertTrue(torch.equal(next_state["pair_contextual"],
                                    torch.full((1, 2), 3.)))
        self.assertNotEqual(vector_hash(state), vector_hash(next_state))

    def test_bounded_source_keeps_head_and_tail(self):
        records = [{"observation": "first marker", "action": "go",
                    "feedback": "middle"},
                   {"observation": "last marker", "action": "take",
                    "feedback": "terminal"}]
        text, original, retained = bounded_source_text(_Tokenizer(),
                                                        records, 12)
        self.assertGreater(original, retained)
        self.assertLessEqual(retained, 12)
        self.assertIn("Completed", text)
        self.assertIn("terminal", text)


if __name__ == "__main__":
    unittest.main()
