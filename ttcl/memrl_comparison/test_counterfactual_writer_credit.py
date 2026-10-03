"""Audit and signal checks for full-horizon paired writer credit."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest

from .counterfactual_writer_credit import (EMPTY_SHA256, score_writer_branches,
                                           score_writer_online_branches,
                                           bind_online_writer_sample)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def grid():
    rows = []
    for target, reward in [('target-a', 1.0), ('target-b', 0.0)]:
        for seed in (17, 18):
            for arm, outcome, calls in [('candidate', reward, 8),
                                        ('previous', 0.0, 10),
                                        ('empty', 0.0, 12)]:
                rows.append(dict(source_trace_sha256=digest('trace'),
                                 source_input_sha256=digest('source'),
                                 target_input_sha256=digest(target),
                                 reviewed_annotation_sha256=digest(
                                     target + ' independently reviewed annotation'),
                                 start_snapshot_sha256=digest(target + ' snapshot'),
                                 actor_seed=seed, arm=arm,
                                 memory_sha256=(EMPTY_SHA256 if arm == 'empty'
                                                else digest(arm)),
                                 official_reward=outcome, actor_calls=calls,
                                 status='complete'))
    return rows


REVIEWS = {digest(target): digest(target + ' independently reviewed annotation')
           for target in ('target-a', 'target-b')}


def online_grid():
    rows = []
    for chain_seed in (17, 18):
        for arm in ('candidate', 'previous', 'empty'):
            previous_end = None
            for position, target in enumerate(('target-a', 'target-b')):
                start = (digest(f'{chain_seed}/{arm}/injected') if position == 0
                         else previous_end)
                previous_end = digest(f'{chain_seed}/{arm}/after/{position}')
                rows.append(dict(source_trace_sha256=digest('trace'),
                                 source_input_sha256=digest('source'),
                                 target_input_sha256=digest(target),
                                 reviewed_annotation_sha256=REVIEWS[digest(target)],
                                 target_position=position, chain_seed=chain_seed,
                                 actor_seed=chain_seed * 100 + position,
                                 root_snapshot_sha256=digest('shared root'),
                                 start_snapshot_sha256=start,
                                 end_snapshot_sha256=previous_end,
                                 writer_freeze_sha256=digest('frozen writer output'),
                                 arm=arm,
                                 memory_sha256=(EMPTY_SHA256 if arm == 'empty'
                                                else digest(arm)),
                                 official_reward=(1.0 if arm == 'candidate' and
                                                  position == 1 else 0.0),
                                 actor_calls=10, status='complete'))
    return rows


class WriterCreditTests(unittest.TestCase):
    def test_full_horizon_credit_and_empty_diagnostic(self):
        result = score_writer_branches(grid(), reviewed_targets=REVIEWS, reward_scale=1.0,
                                       call_budget=20, cost_weight=.1)
        self.assertEqual(result['paired_count'], 4)
        self.assertAlmostEqual(result['advantage'], .51)
        self.assertAlmostEqual(result['mean_vs_empty'], .52)
        self.assertEqual(result['nonzero_pairs'], 4)
        self.assertFalse(result['seed_sign_conflict'])

    def test_missing_or_unmatched_branches_cannot_be_labels(self):
        rows = grid()
        with self.assertRaisesRegex(ValueError, 'Missing paired'):
            score_writer_branches(rows[:-1], reviewed_targets=REVIEWS,
                                  reward_scale=1, call_budget=20)
        changed = copy.deepcopy(rows)
        changed[0]['start_snapshot_sha256'] = digest('different')
        with self.assertRaisesRegex(ValueError, 'frozen snapshot'):
            score_writer_branches(changed, reviewed_targets=REVIEWS,
                                  reward_scale=1, call_budget=20)
        changed = copy.deepcopy(rows)
        changed[0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'Incomplete branch'):
            score_writer_branches(changed, reviewed_targets=REVIEWS,
                                  reward_scale=1, call_budget=20)
        with self.assertRaisesRegex(ValueError, 'reviewed annotation'):
            score_writer_branches(rows, reviewed_targets={digest('target-a'): digest('review')},
                                  reward_scale=1, call_budget=20)
        changed = copy.deepcopy(rows)
        changed[0]['reviewed_annotation_sha256'] = digest('stale annotation')
        with self.assertRaisesRegex(ValueError, 'reviewed annotation binding'):
            score_writer_branches(changed, reviewed_targets=REVIEWS,
                                  reward_scale=1, call_budget=20)

    def test_identical_documents_get_no_writer_credit(self):
        rows = grid()
        for row in rows:
            if row['arm'] == 'candidate':
                row['memory_sha256'] = digest('previous')
        result = score_writer_branches(rows, reviewed_targets=REVIEWS,
                                       reward_scale=1, call_budget=20)
        self.assertTrue(result['identical_candidate_and_previous'])
        self.assertEqual(result['advantage'], 0)
        self.assertNotEqual(result['mean_unclipped_advantage'], 0)

    def test_online_chain_credits_delayed_gain_with_divergent_states(self):
        result = score_writer_online_branches(
            online_grid(), reviewed_targets=list(REVIEWS.items()),
            reward_scale=1, call_budget=20)
        self.assertEqual(result['paired_cells'], 12)
        self.assertAlmostEqual(result['advantage'], .5)
        self.assertAlmostEqual(result['immediate_mean_vs_previous'], 0.)
        self.assertAlmostEqual(result['delayed_mean_vs_previous'], .5)
        self.assertEqual([row['raw_vs_previous'] for row in
                          result['per_target_effects'] if row['chain_seed'] == 17],
                         [0, 1])
        self.assertFalse(result['seed_sign_conflict'])

    def test_online_chain_requires_continuity_and_matched_arms(self):
        reviewed = list(REVIEWS.items())
        broken = online_grid()
        broken[-1]['start_snapshot_sha256'] = digest('unrelated state')
        with self.assertRaisesRegex(ValueError, 'transition is broken'):
            score_writer_online_branches(broken, reviewed_targets=reviewed,
                                         reward_scale=1, call_budget=20)
        with self.assertRaisesRegex(ValueError, 'Incomplete online chain grid'):
            score_writer_online_branches(online_grid()[:-1],
                                         reviewed_targets=reviewed,
                                         reward_scale=1, call_budget=20)
        broken = online_grid()
        broken[0]['root_snapshot_sha256'] = digest('other root')
        with self.assertRaisesRegex(ValueError, 'root snapshot'):
            score_writer_online_branches(broken, reviewed_targets=reviewed,
                                         reward_scale=1, call_budget=20)
        broken = online_grid()
        broken[0]['reviewed_annotation_sha256'] = digest('stale annotation')
        with self.assertRaisesRegex(ValueError, 'reviewed annotation binding'):
            score_writer_online_branches(broken, reviewed_targets=reviewed,
                                         reward_scale=1, call_budget=20)

    def test_writer_gradient_needs_bound_replicated_chain_effect(self):
        messages = [{'role': 'user', 'content': 'completed public source'}]
        source_input_sha256 = hashlib.sha256(json.dumps(
            messages, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        rows = online_grid()
        for row in rows:
            row['source_input_sha256'] = source_input_sha256
        sample = dict(text='candidate', messages=messages,
                      source_input_sha256=source_input_sha256,
                      source_trace_sha256=digest('trace'),
                      writer_freeze_sha256=digest('frozen writer output'))
        report = score_writer_online_branches(
            rows, reviewed_targets=list(REVIEWS.items()),
            reward_scale=1, call_budget=20)
        labeled = bind_online_writer_sample(sample, report)
        self.assertEqual(labeled['paired_credit_status'], 'replicated_effect')
        self.assertAlmostEqual(labeled['advantage'], .5)
        with self.assertRaisesRegex(ValueError, 'differs from frozen'):
            bind_online_writer_sample(dict(sample, text='different'), report)
        opposed = copy.deepcopy(rows)
        for row in opposed:
            if (row['chain_seed'] == 18 and row['arm'] == 'candidate' and
                    row['target_position'] == 1):
                row['official_reward'] = -1.0
        uncertain = score_writer_online_branches(
            opposed, reviewed_targets=list(REVIEWS.items()),
            reward_scale=1, call_budget=20)
        labeled = bind_online_writer_sample(sample, uncertain)
        self.assertEqual(labeled['paired_credit_status'], 'needs_more_evidence')
        self.assertEqual(labeled['advantage'], 0)


if __name__ == '__main__':
    unittest.main()
