"""Mechanism tests for paired memory credit and conservative decisions."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .paired_credit_rl import (
    FEATURES, _bounded, _exact_two_memory_credit, _group_key, _scale,
    features, fit, predict,
)
from .paired_credit_policy import filter_native_retrieval


class PairedCreditRLTest(unittest.TestCase):
    def test_shapley_handles_complementary_and_substitutable_memories(self):
        complement = {'none': 0., 'full': 1., 'drop_0': 0., 'drop_1': 0.}
        substitute = {'none': 0., 'full': 1., 'drop_0': 1., 'drop_1': 1.}
        for rewards in (complement, substitute):
            self.assertEqual(_exact_two_memory_credit(rewards, 0), .5)
            self.assertEqual(_exact_two_memory_credit(rewards, 1), .5)
            self.assertEqual(sum(_exact_two_memory_credit(rewards, i)
                                 for i in range(2)), rewards['full'] - rewards['none'])
        self.assertEqual(substitute['full'] - substitute['drop_0'], 0.)

    def test_features_use_prior_record_and_retrieval_only(self):
        record = dict(metadata=dict(success=False, official_reward=-1.,
                                    q_value=-.5, q_visits=3,
                                    reward_ma=-.4, last_reward=-1.,
                                    writer_token_limit_hit=False),
                      retrieval=dict(similarity=.8, q_estimate=-.3, score=.4),
                      text_characters=120)
        x = features(record, 3, 1)
        self.assertEqual(len(x), len(FEATURES))
        self.assertEqual(x[0], 1.)
        self.assertEqual(x[FEATURES.index('success')], -1.)

    def test_observed_negative_credit_can_trigger_removal(self):
        record = dict(metadata={}, retrieval={}, text_characters=100)
        x = features(record, 1, 0)
        examples = [dict(binding=['alfworld_train', f'alfworld/case_{i}',
                                  f'm{i}', 'text', f'alf_input_{i}'],
                         x=x, label=-1.) for i in range(20)]
        examples += [dict(binding=['clbench_calibration', f'clbench/case_{i}',
                                   f'n{i}', 'text', f'cl_input_{i}'],
                          x=x, label=-1.) for i in range(20)]
        policy = fit(dict(dataset_sha256='fixture', examples=examples))
        result = predict(policy, x)
        self.assertTrue(result['remove'])
        self.assertLess(result['upper_95'], 0.)
        self.assertEqual(policy['training_cases'], 40)

    def test_gate_removes_at_most_one_of_two_predicted_harmful_memories(self):
        entries = {
            'a': dict(task_description='prior A', public_abstract='lesson A'),
            'b': dict(task_description='prior B', public_abstract='lesson B'),
        }
        class Store:
            def get(self, mid):
                return SimpleNamespace(metadata=SimpleNamespace(
                    model_dump=lambda: entries[mid]))
        tokenizer = SimpleNamespace(encode=lambda text, **_kwargs: list(text))
        memory = SimpleNamespace(store=Store(),
                                 client=SimpleNamespace(tokenizer=tokenizer))
        full = ('Task: prior A\nExperience: lesson A\n\n'
                'Task: prior B\nExperience: lesson B')
        retrieval = dict(ids=['a', 'b'], context=full,
                         candidates=[dict(memory_id='a'), dict(memory_id='b')])
        with patch('ttcl.memrl_comparison.paired_credit_policy.predict',
                   side_effect=[dict(mean=-.4, upper_95=-.3, remove=True),
                                dict(mean=-.5, upper_95=-.4, remove=True)]):
            selected = filter_native_retrieval(memory, retrieval, {})
        self.assertEqual(selected['suppressed_ids'], ['b'])
        self.assertEqual(selected['ids'], ['a'])
        self.assertEqual(selected['context'], 'Task: prior A\nExperience: lesson A')

    def test_crossfit_group_uses_input_content_across_repeats(self):
        first = dict(binding=['clbench_calibration',
                              'clbench/domain/303/memrl/episode_003',
                              'memory_000001', 'text_a', 'same_public_input'])
        second = dict(binding=['clbench_calibration',
                               'clbench/domain/404/memrl/episode_003',
                               'memory_000004', 'text_b', 'same_public_input'])
        self.assertEqual(_group_key(first), _group_key(second))

    def test_reward_normalization_keeps_small_scores_and_caps_outliers(self):
        self.assertEqual(_scale([.001, .002, .02, .03]), .05)
        self.assertEqual(_scale([0., 1., 5., 100.]), 5.)
        self.assertEqual(_bounded(100., 5.), 1.)


if __name__ == '__main__':
    unittest.main()
