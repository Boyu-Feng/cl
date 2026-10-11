import unittest
import torch
from .persistent_update_v95 import PersistentUpdateRule
from .self_evolving_update_v98 import SelfEvolvingUpdater

class TestEvolution(unittest.TestCase):
    def test_commit_uses_updated_parameters_once(self):
        torch.manual_seed(98)
        rule = PersistentUpdateRule([(3, 2)], state_dim=4, width=8)
        updater = SelfEvolvingUpdater(rule, [torch.randn(2, 5)], [10.], lr=.01)
        state = updater.initial_state([torch.zeros(1, 3, 2)])
        rows = [dict(before=torch.randn(1, 4), after=torch.randn(1, 4),
                     action=torch.tensor([i]), reward=torch.tensor([float(i == 1)]),
                     done=torch.tensor([0.])) for i in range(2)]
        before = {k: v.clone() for k, v in rule.state_dict().items()}
        def loss(b, targets):
            self.assertEqual(len(targets), 1)
            return (b[0] - 1).square().mean()
        new, info = updater.observe_and_learn(state, rows, loss)
        self.assertTrue(info['updated'])
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in rule.state_dict().items()))
        expected, _, _ = updater.propose(state, rows)
        torch.testing.assert_close(new.memory, expected.memory)
        torch.testing.assert_close(new.b[0], expected.b[0])
        self.assertGreater(float(new.b[0].abs().sum()), 0)
        self.assertEqual(new.episodes_seen, 1)
        self.assertEqual(state.episodes_seen, 0)
        # Failed trajectory commits memory once without changing parameters.
        rows[1]['reward'].zero_()
        before = {k: v.clone() for k, v in rule.state_dict().items()}
        failed, info = updater.observe_and_learn(new, rows, lambda *_: self.fail('unexpected label'))
        self.assertFalse(info['updated'])
        self.assertEqual(failed.episodes_seen, 2)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in rule.state_dict().items()))
        self.assertFalse(torch.equal(new.memory, failed.memory))

if __name__ == '__main__': unittest.main()
