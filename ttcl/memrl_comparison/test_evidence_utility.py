import unittest

from .evidence_utility import PairedUtilityGate


class PairedUtilityGateTest(unittest.TestCase):
    def test_repeats_do_not_count_as_independent_transfer(self):
        gate = PairedUtilityGate()
        base = dict(source_content_sha256='a' * 64, target_domain='domain',
                    native_state_sha256='c' * 64,
                    paired_native_state_sha256='c' * 64,
                    baseline_reward=0., with_evidence_reward=1.,
                    shown_card_ids=['a' * 64])
        first = gate.observe(**base, target_input_sha256='b' * 64, actor_seed=1)
        repeated = gate.observe(**base, target_input_sha256='b' * 64, actor_seed=2)
        self.assertEqual(first['status']['state'], 'probation')
        self.assertEqual(repeated['status']['independent_win_targets'], 1)
        independent = gate.observe(**base, target_input_sha256='d' * 64, actor_seed=3)
        self.assertEqual(independent['status']['state'], 'active')
        with self.assertRaises(ValueError):
            gate.observe(**base, target_input_sha256='d' * 64, actor_seed=3)

    def test_loss_or_state_mismatch_blocks_activation(self):
        gate = PairedUtilityGate()
        base = dict(source_content_sha256='a' * 64, target_domain='domain',
                    target_input_sha256='b' * 64, actor_seed=1,
                    native_state_sha256='c' * 64,
                    paired_native_state_sha256='c' * 64,
                    baseline_reward=1., with_evidence_reward=0.,
                    shown_card_ids=['a' * 64])
        self.assertEqual(gate.observe(**base)['status']['state'], 'quarantine')
        with self.assertRaises(ValueError):
            gate.observe(**dict(base, actor_seed=2,
                                paired_native_state_sha256='d' * 64))


if __name__ == '__main__':
    unittest.main()
