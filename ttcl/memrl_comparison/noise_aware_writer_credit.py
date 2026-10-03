"""Noise-gated full-chain credit for a frozen MemRL experience writer.

The original whole-chain scorer supplies the official, content-bound return.
This module requires a second independent rollout of both candidate and
previous documents on every chain seed. A writer gradient is released only
when candidate and previous return ranges do not overlap on any seed.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .counterfactual_writer_credit import (bind_online_writer_sample,
                                            score_writer_online_branches)


ARMS = frozenset({'candidate', 'candidate_repeat', 'previous',
                  'previous_repeat', 'empty'})
BALANCED_ORDERS = (
    ('candidate', 'previous', 'previous_repeat', 'candidate_repeat'),
    ('previous', 'candidate', 'candidate_repeat', 'previous_repeat'),
)
EPS = 1e-12


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value))


def _score_view(rows: list[dict], candidate_arm: str,
                *, reviewed_targets: list[tuple[str, str]],
                reward_scale: float, call_budget: int, cost_weight: float,
                discount: float) -> dict:
    view = []
    for row in rows:
        if row['arm'] in {'previous', 'empty'}:
            view.append(row)
        elif row['arm'] == candidate_arm:
            view.append(dict(row, arm='candidate'))
    return score_writer_online_branches(
        view, reviewed_targets=reviewed_targets, reward_scale=reward_scale,
        call_budget=call_budget, cost_weight=cost_weight, discount=discount)


def _decision(seed_evidence: dict, same_document: bool) -> tuple[str, float, float]:
    if not isinstance(seed_evidence, dict):
        raise ValueError('Missing repeat-control evidence')
    evidence = list(seed_evidence.values())
    if not evidence or any(not isinstance(item, dict) or
                           not all(name in item for name in
                                   ('candidate_returns_vs_previous',
                                    'previous_returns_vs_previous',
                                    'corrected_effect', 'repeat_noise_radius',
                                    'decisive_positive', 'decisive_negative'))
                           for item in evidence):
        raise ValueError('Missing repeat-control evidence')
    for item in evidence:
        effect, radius = item['corrected_effect'], item['repeat_noise_radius']
        candidate = item['candidate_returns_vs_previous']
        baseline = item['previous_returns_vs_previous']
        if (not isinstance(candidate, list) or len(candidate) != 2 or
                not isinstance(baseline, list) or len(baseline) != 2 or
                any(isinstance(value, bool) or
                    not isinstance(value, (int, float)) or
                    not math.isfinite(value) for value in candidate + baseline) or
                baseline[0] != 0.):
            raise ValueError('Invalid repeated candidate or baseline returns')
        expected_effect = statistics.fmean(candidate) - statistics.fmean(baseline)
        expected_radius = (abs(candidate[0] - candidate[1]) +
                           abs(baseline[0] - baseline[1])) / 2
        if (not isinstance(effect, (int, float)) or not math.isfinite(effect) or
                not isinstance(radius, (int, float)) or
                not math.isfinite(radius) or radius < 0 or
                not math.isclose(effect, expected_effect, abs_tol=1e-12) or
                not math.isclose(radius, expected_radius, abs_tol=1e-12) or
                item['decisive_positive'] != (effect > radius + EPS) or
                item['decisive_negative'] != (effect < -radius - EPS)):
            raise ValueError('Repeat-control decision does not match evidence')
    if same_document:
        status = 'unchanged_document'
    elif all(item['decisive_positive'] for item in evidence):
        status = 'replicated_above_control_noise'
    elif all(item['decisive_negative'] for item in evidence):
        status = 'replicated_below_control_noise'
    else:
        status = 'needs_more_evidence'
    mean_effect = statistics.fmean(item['corrected_effect'] for item in evidence)
    advantage = (max(-1.0, min(1.0, mean_effect)) if status.startswith(
                 'replicated_') else 0.0)
    return status, mean_effect, advantage


def score_noise_aware_online_writer(
        rows: list[dict], *, reviewed_targets: list[tuple[str, str]],
        reward_scale: float, call_budget: int, cost_weight: float = 0.0,
        discount: float = 1.0) -> dict:
    """Score five complete arms while keeping uncertain credit out of PPO.

    The collector must freeze and audit `execution_plan_sha256` before exposing
    targets. Each arm/seed is a distinct chain run; its `chain_run_sha256` is
    constant across targets and unique among arms of that seed. The scorer
    validates these bindings, source lineage, official rewards and snapshots.
    This strict range gate is an evidence rule, not a calibrated confidence
    interval or a claim that two repeats capture all model variability.
    """
    if not rows or any(not isinstance(row, dict) or row.get('arm') not in ARMS
                       for row in rows):
        raise ValueError('Expected candidate, previous, repeat and empty arms')
    plans, chain_runs = set(), {}
    ordinals, by_target, by_chain = set(), defaultdict(list), defaultdict(list)
    for row in rows:
        plan, run = row.get('execution_plan_sha256'), row.get('chain_run_sha256')
        if not _digest(plan) or not _digest(run):
            raise ValueError('Missing frozen execution plan or chain run binding')
        ordinal = row.get('execution_ordinal')
        if (isinstance(ordinal, bool) or not isinstance(ordinal, int) or
                ordinal < 0 or ordinal in ordinals):
            raise ValueError('Missing or reused execution ordinal')
        position = row.get('target_position')
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError('Invalid ordered target position')
        ordinals.add(ordinal)
        plans.add(plan)
        key = (row.get('chain_seed'), row['arm'])
        if key in chain_runs and chain_runs[key] != run:
            raise ValueError('Chain run changed within an arm')
        chain_runs[key] = run
        by_chain[key].append((position, ordinal))
        by_target[(row.get('chain_seed'), position)].append((ordinal, row['arm']))
    if len(plans) != 1:
        raise ValueError('Branch execution plans differ')
    if len(set(chain_runs.values())) != len(chain_runs):
        raise ValueError('Same chain run reused across arms or seeds')
    for chain_rows in by_chain.values():
        positions = sorted(chain_rows)
        if (any(left[1] >= right[1] for left,right in
                    zip(positions, positions[1:]))):
            raise ValueError('Execution order breaks within-arm task chain')
    for target_rows in by_target.values():
        order = tuple(arm for _,arm in sorted(target_rows) if arm != 'empty')
        if order not in BALANCED_ORDERS:
            raise ValueError('Candidate and previous repeats were not interleaved')
    for chain_seed in {key[0] for key in chain_runs}:
        selected = {arm:run for (seed,arm),run in chain_runs.items()
                    if seed == chain_seed}
        if set(selected) != ARMS or len(set(selected.values())) != len(ARMS):
            raise ValueError('Missing or reused independent chain run')
    args = dict(reviewed_targets=reviewed_targets,
                reward_scale=reward_scale, call_budget=call_budget,
                cost_weight=cost_weight, discount=discount)
    factual = _score_view(rows, 'candidate', **args)
    candidate_repeat = _score_view(rows, 'candidate_repeat', **args)
    previous_repeat = _score_view(rows, 'previous_repeat', **args)
    keys = ('source_trace_sha256', 'source_input_sha256',
            'root_snapshot_sha256', 'writer_freeze_sha256',
            'reviewed_target_order_sha256', 'target_count', 'seed_count')
    if any(candidate_repeat[key] != factual[key] or
           previous_repeat[key] != factual[key] for key in keys):
        raise ValueError('Repeat arms differ from the factual chain binding')
    if (candidate_repeat['memory_sha256']['candidate'] !=
            factual['memory_sha256']['candidate'] or
            previous_repeat['memory_sha256']['candidate'] !=
            factual['memory_sha256']['previous']):
        raise ValueError('Repeated arm changed its candidate or previous document')
    if (set(candidate_repeat['seed_effects']) != set(factual['seed_effects']) or
            set(previous_repeat['seed_effects']) !=
            set(factual['seed_effects'])):
        raise ValueError('Repeat arms changed the paired chain seeds')
    seed_evidence = {}
    for chain_seed, primary in factual['seed_effects'].items():
        candidate_one = primary['vs_previous']
        candidate_two = candidate_repeat['seed_effects'][chain_seed]['vs_previous']
        previous_one = 0.0
        previous_two = previous_repeat['seed_effects'][chain_seed]['vs_previous']
        candidate_mean = (candidate_one + candidate_two) / 2
        baseline_mean = (previous_one + previous_two) / 2
        effect = candidate_mean - baseline_mean
        radius = (abs(candidate_one - candidate_two) / 2 +
                  abs(previous_one - previous_two) / 2)
        if not all(math.isfinite(value) for value in
                   (candidate_mean, baseline_mean, effect, radius)):
            raise ValueError('Nonfinite repeated-chain credit')
        seed_evidence[chain_seed] = dict(
            candidate_returns_vs_previous=[candidate_one, candidate_two],
            previous_returns_vs_previous=[previous_one, previous_two],
            corrected_effect=effect, repeat_noise_radius=radius,
            decisive_positive=effect > radius + EPS,
            decisive_negative=effect < -radius - EPS)
    same_document = factual['identical_candidate_and_previous']
    status, mean_effect, advantage = _decision(seed_evidence, same_document)
    return dict(schema='noise_aware_online_writer_credit_v1',
                execution_plan_sha256=next(iter(plans)),
                source_trace_sha256=factual['source_trace_sha256'],
                source_input_sha256=factual['source_input_sha256'],
                root_snapshot_sha256=factual['root_snapshot_sha256'],
                writer_freeze_sha256=factual['writer_freeze_sha256'],
                reviewed_target_order_sha256=
                factual['reviewed_target_order_sha256'],
                memory_sha256=factual['memory_sha256'],
                target_count=factual['target_count'],
                seed_count=factual['seed_count'],
                branch_cells=len(rows),
                advantage=advantage, mean_corrected_effect=mean_effect,
                status=status, identical_candidate_and_previous=same_document,
                seed_evidence=seed_evidence,
                factual_report=factual,
                caveat='Range gate from two repeats per document and seed; not calibrated statistical confidence')


def bind_noise_aware_writer_sample(sample: dict, report: dict) -> dict:
    """Bind a sample to audited evidence; uncertain credit gives no policy term."""
    if report.get('schema') != 'noise_aware_online_writer_credit_v1':
        raise ValueError('Missing noise-aware chain report')
    status, mean_effect, advantage = _decision(
        report.get('seed_evidence', {}),
        report.get('identical_candidate_and_previous'))
    if (report.get('seed_count') != len(report['seed_evidence']) or
            report.get('status') != status or
            report.get('mean_corrected_effect') != mean_effect or
            report.get('advantage') != advantage):
        raise ValueError('Noise-aware label differs from repeated controls')
    bound = bind_online_writer_sample(sample, report['factual_report'])
    if (report['source_input_sha256'] != bound['source_input_sha256'] or
            report['writer_freeze_sha256'] != bound['writer_freeze_sha256'] or
            report['memory_sha256'] !=
            report['factual_report']['memory_sha256']):
        raise ValueError('Noise-aware report differs from writer sample')
    return dict(bound, advantage=report['advantage'],
                paired_credit_status=report['status'],
                paired_credit_unclipped=report['mean_corrected_effect'],
                paired_credit_execution_plan_sha256=
                report['execution_plan_sha256'])
