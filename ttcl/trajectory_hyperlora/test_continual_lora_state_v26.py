import unittest

import torch

from ttcl.trajectory_hyperlora.continual_lora_state_v26 import ContinualLoRAState


class ContinualLoRAStateTests(unittest.TestCase):
    def test_one_state_persists_and_updates_only_on_write(self):
        memory = ContinualLoRAState()
        query = torch.tensor([1., 0.])
        self.assertIsNone(memory.read(query)[0])
        first = torch.full((1, 2, 1), 2.)
        second = torch.full((1, 2, 1), 4.)
        memory.write(query, [first])
        old_digest = memory.digest()
        factors, meta = memory.read(torch.tensor([0., 1.]))
        self.assertEqual(meta["updates"], 1)
        self.assertTrue(torch.equal(factors[0], first))
        factors[0].zero_()
        self.assertEqual(memory.digest(), old_digest)
        memory.write(torch.tensor([0., 1.]), [second])
        updated, meta = memory.read(query)
        self.assertEqual(len(memory.entries), 1)
        self.assertEqual(meta["updates"], 2)
        self.assertTrue(torch.equal(updated[0], torch.full((1, 2, 1), 3.)))
        self.assertNotEqual(memory.digest(), old_digest)

    def test_rejects_shape_change(self):
        memory = ContinualLoRAState()
        memory.write(torch.tensor([1., 0.]), [torch.ones(1, 2, 1)])
        with self.assertRaisesRegex(ValueError, "shape changed"):
            memory.write(torch.tensor([1., 0.]), [torch.ones(1, 3, 1)])


if __name__ == "__main__":
    unittest.main()
