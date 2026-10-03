"""Behavioral checks for repeated whole-chain writer credit."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest

from .counterfactual_writer_credit import EMPTY_SHA256
from .noise_aware_writer_credit import (bind_noise_aware_writer_sample,
                                        score_noise_aware_online_writer)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


MESSAGES = [{'role': 'user', 'content': 'completed public source'}]
SOURCE_INPUT = hashlib.sha256(json.dumps(
    MESSAGES, ensure_ascii=False, sort_keys=True,
    separators=(',', ':')).encode()).hexdigest()
REVIEWS = [(digest(target), digest(target + ' independently reviewed'))
           for target in ('target-a', 'target-b')]


def grid() -> list[dict]:
    rows = []
    arm_order = {'candidate':0, 'previous':1, 'previous_repeat':2,
                 'candidate_repeat':3, 'empty':4}
    for chain_seed in (17, 18):
        for arm in ('candidate', 'candidate_repeat', 'previous',
                    'previous_repeat', 'empty'):
            previous_end = None
            for position, (target, reviewed) in enumerate(REVIEWS):
                start = (digest(f'{chain_seed}/{arm}/injected') if position == 0
                         else previous_end)
                previous_end = digest(f'{chain_seed}/{arm}/after/{position}')
                rows.append(dict(
                    source_trace_sha256=digest('trace'),
                    source_input_sha256=SOURCE_INPUT,
                    target_input_sha256=target,
                    reviewed_annotation_sha256=reviewed,
                    target_position=position, chain_seed=chain_seed,
                    actor_seed=chain_seed * 100 + position,
                    root_snapshot_sha256=digest('shared root'),
                    start_snapshot_sha256=start,
                    end_snapshot_sha256=previous_end,
                    writer_freeze_sha256=digest('frozen writer'),
                    execution_plan_sha256=digest('frozen arm schedule'),
                    execution_ordinal=(
                        ((chain_seed - 17) * len(REVIEWS) + position) * 5 +
                        arm_order[arm]),
                    chain_run_sha256=digest(f'{chain_seed}/{arm}/run'),
                    arm=arm,
                    memory_sha256=(EMPTY_SHA256 if arm == 'empty' else
                                   digest('candidate' if arm.startswith(
                                       'candidate') else 'previous')),
                    official_reward=(1.0 if arm.startswith('candidate') and
                                     position == 1 else 0.0),
                    actor_calls=10, status='complete'))
    return rows


def score(rows: list[dict]) -> dict:
    return score_noise_aware_online_writer(
        rows, reviewed_targets=REVIEWS, reward_scale=1., call_budget=20)


class NoiseAwareWriterCreditTests(unittest.TestCase):
    def test_delayed_effect_survives_repeated_control(self):
        report = score(grid())
        self.assertEqual(report['status'], 'replicated_above_control_noise')
        self.assertAlmostEqual(report['advantage'], .5)
        self.assertEqual(report['seed_count'], 2)
        self.assertEqual(report['branch_cells'], 20)
        sample = dict(text='candidate', messages=MESSAGES,
                      source_input_sha256=SOURCE_INPUT,
                      source_trace_sha256=digest('trace'),
                      writer_freeze_sha256=digest('frozen writer'))
        bound = bind_noise_aware_writer_sample(sample, report)
        self.assertAlmostEqual(bound['advantage'], .5)
        tampered = copy.deepcopy(report)
        tampered['advantage'] = 1.
        with self.assertRaisesRegex(ValueError, 'differs from repeated controls'):
            bind_noise_aware_writer_sample(sample, tampered)

    def test_same_document_control_gain_blocks_false_writer_label(self):
        rows = grid()
        for row in rows:
            if (row['arm'] == 'previous_repeat' and
                    row['target_position'] == 1):
                row['official_reward'] = 1.
        report = score(rows)
        self.assertEqual(report['status'], 'needs_more_evidence')
        self.assertEqual(report['advantage'], 0.)
        self.assertAlmostEqual(report['mean_corrected_effect'], .25)
        self.assertTrue(all(value['repeat_noise_radius'] == .25
                            for value in report['seed_evidence'].values()))

    def test_candidate_repeat_disagreement_blocks_label(self):
        rows = grid()
        for row in rows:
            if (row['arm'] == 'candidate_repeat' and
                    row['target_position'] == 1):
                row['official_reward'] = 0.
        report = score(rows)
        self.assertEqual(report['status'], 'needs_more_evidence')
        self.assertEqual(report['advantage'], 0.)

    def test_identical_documents_never_gain_writer_credit(self):
        rows = grid()
        for row in rows:
            if row['arm'].startswith('candidate'):
                row['memory_sha256'] = digest('previous')
        report = score(rows)
        self.assertEqual(report['status'], 'unchanged_document')
        self.assertEqual(report['advantage'], 0.)

    def test_replicated_harm_produces_negative_credit(self):
        rows = grid()
        for row in rows:
            if row['arm'].startswith('candidate') and row['target_position'] == 1:
                row['official_reward'] = -1.
        report = score(rows)
        self.assertEqual(report['status'], 'replicated_below_control_noise')
        self.assertAlmostEqual(report['advantage'], -.5)

    def test_binding_failures_are_not_training_examples(self):
        rows = grid()
        broken = copy.deepcopy(rows)
        for row in broken:
            if row['arm'] == 'candidate_repeat':
                row['memory_sha256'] = digest('different writer text')
        with self.assertRaisesRegex(ValueError, 'document'):
            score(broken)
        broken = copy.deepcopy(rows)
        broken[0]['chain_run_sha256'] = broken[2]['chain_run_sha256']
        with self.assertRaisesRegex(ValueError, 'Chain run changed'):
            score(broken)
        broken = copy.deepcopy(rows)
        for row in broken:
            if row['chain_seed'] == 18 and row['arm'] == 'candidate':
                row['chain_run_sha256'] = digest('17/candidate/run')
        with self.assertRaisesRegex(ValueError, 'reused across'):
            score(broken)
        broken = copy.deepcopy(rows)
        first = next(row for row in broken if row['chain_seed'] == 17 and
                     row['target_position'] == 0 and
                     row['arm'] == 'previous_repeat')
        second = next(row for row in broken if row['chain_seed'] == 17 and
                      row['target_position'] == 0 and
                      row['arm'] == 'candidate_repeat')
        first['execution_ordinal'], second['execution_ordinal'] = (
            second['execution_ordinal'], first['execution_ordinal'])
        with self.assertRaisesRegex(ValueError, 'not interleaved'):
            score(broken)
        broken = copy.deepcopy(rows)
        broken[0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'Incomplete branch'):
            score(broken)
        broken = copy.deepcopy(rows)
        broken[0]['reviewed_annotation_sha256'] = digest('stale review')
        with self.assertRaisesRegex(ValueError, 'reviewed annotation'):
            score(broken)


if __name__ == '__main__':
    unittest.main()
