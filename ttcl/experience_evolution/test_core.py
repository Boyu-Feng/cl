import unittest
import torch
from .core import writer_messages, reward_signal
from .writer import clipped_loss


class SignalTests(unittest.TestCase):
    def test_sign_and_no_centering(self):
        for a,b,want in [(1,0,1),(0,1,-1),(1,1,0),(0,0,0)]:
            self.assertEqual(reward_signal(a,b,"delta"),want)
            value = torch.tensor([-1.0], requires_grad=True)
            clipped_loss(value, value.detach(), want).backward()
            self.assertEqual(float(value.grad), -want)

    def test_writer_cannot_see_future_or_baseline(self):
        ep = dict(initial_observation="source", trajectory=[{"action":"a","observation":"o"}],
                  reward=0, steps=1, future_task="SECRET", baseline_trajectory="SECRET")
        m = writer_messages("prior lesson", ep)
        self.assertIn("prior lesson", m[1]["content"])
        self.assertIn("source", m[1]["content"])
        self.assertNotIn("SECRET", m[1]["content"])


if __name__ == "__main__":
    unittest.main()
