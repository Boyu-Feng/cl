"""Paired, target-cluster-aware diagnostics for equal-budget v4 versus v5 RL.

The 72 training pairs are repeated observations on fewer official games; this
module reports counts at both pair and game levels without treating pairs as
independent benchmark tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def norm(values):
    return math.sqrt(sum(x*x for x in values))


def cosine(left, right):
    denominator = norm(left)*norm(right)
    return sum(x*y for x, y in zip(left, right, strict=True))/denominator \
        if denominator else None


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.old_report.read_text())
    old_audit = json.loads(args.old_audit.read_text())
    new = json.loads(args.new_report.read_text())
    new_audit = json.loads(args.new_audit.read_text())
    if (old_audit['report_sha256'] != file_hash(args.old_report) or
            new_audit['raw_report_sha256'] != file_hash(args.new_report) or
            new_audit['reference_report_sha256'] !=
                file_hash(args.old_report) or
            old['pair_schedule'] != new['pair_schedule'] or
            len(old['train']) != 72 or len(new['train']) != 72 or
            old['max_rollouts'] != 144 or new['max_rollouts'] != 144 or
            old['failures'] or new['failures'] or
            new_audit['summary']['failed_replays'] != 0 or
            new_audit['summary']['pairs'] != 72):
        raise ValueError('Changed or unaudited paired reward-training runs')
    pair_table = {'both': 0, 'new_only': 0, 'old_only': 0, 'neither': 0}
    first_divergence = {'v4': 0, 'v5': 0}
    target = defaultdict(lambda: {'pairs': 0, 'v4_signal': 0,
                                  'v5_signal': 0})
    old_norm, new_norm = [], []
    exploration_parallel_energy = []
    score_parallel_energy = []
    signal_score_parallel_energy = []
    first_old, second_old, first_new, second_new = (
        [0.]*8 for _ in range(4))
    for index, (left, right) in enumerate(zip(old['train'], new['train'],
                                             strict=True)):
        si = new['pair_schedule'][index]['source_id']
        ti = new['pair_schedule'][index]['target_id']
        if (left['source_id'] != si or right['source_id'] != si or
                left['target_id'] != ti or right['target_id'] != ti or
                left['noise'] != right['noise']):
            raise ValueError('Changed same-source paired random seed')
        old_signal = left['positive_reward'] != left['negative_reward']
        new_signal = right['positive_reward'] != right['negative_reward']
        pair_table[('both' if old_signal and new_signal else
                    'new_only' if new_signal else
                    'old_only' if old_signal else 'neither')] += 1
        target[ti]['pairs'] += 1
        target[ti]['v4_signal'] += int(old_signal)
        target[ti]['v5_signal'] += int(new_signal)
        for key, row in [('v4', left), ('v5', right)]:
            first_divergence[key] += (
                row['positive']['trajectory'][0]['command'] !=
                row['negative']['trajectory'][0]['command'])
        old_norm.append(.5*norm(left['noise']))
        new_norm.append(norm(right['delta']))
        direction = right['direction']
        sample_energy = (sum(a*b for a, b in zip(
            right['delta'], direction, strict=True)) / norm(right['delta']))**2
        score_energy = (sum(a*b for a, b in zip(
            right['precision_delta'], direction, strict=True)) /
            norm(right['precision_delta']))**2
        exploration_parallel_energy.append(sample_energy)
        score_parallel_energy.append(score_energy)
        if new_signal:
            signal_score_parallel_energy.append(score_energy)
        old_gradient = [(left['positive_reward']-
                         left['negative_reward'])*z for z in left['noise']]
        new_gradient = [.5*(right['positive_reward']-
                             right['negative_reward'])*z
                        for z in right['precision_delta']]
        for j in range(8):
            (first_old if index < 36 else second_old)[j] += old_gradient[j]
            (first_new if index < 36 else second_new)[j] += new_gradient[j]
    rows = [{'target_id': ti, **values,
             'signal_difference': values['v5_signal']-values['v4_signal']}
            for ti, values in sorted(target.items())]
    summary = {'pairs': 72, 'distinct_targets': len(rows),
        'v4_nonzero_pairs': pair_table['both']+pair_table['old_only'],
        'v5_nonzero_pairs': pair_table['both']+pair_table['new_only'],
        'paired_signal_table': pair_table,
        'target_wins_v5': sum(x['signal_difference'] > 0 for x in rows),
        'target_wins_v4': sum(x['signal_difference'] < 0 for x in rows),
        'target_ties': sum(x['signal_difference'] == 0 for x in rows),
        'first_command_divergence': first_divergence,
        'mean_realized_code_shift_norm': {
            'v4': statistics.mean(old_norm),
            'v5': statistics.mean(new_norm)},
        'v5_parallel_energy_fraction': {
            'exploration_mean': statistics.mean(exploration_parallel_energy),
            'score_mean': statistics.mean(score_parallel_energy),
            'score_nonzero_reward_mean': statistics.mean(
                signal_score_parallel_energy)},
        'half_code_gradient_cosine': {
            'v4': cosine(first_old, second_old),
            'v5': cosine(first_new, second_new)},
        'positive_success': {'v4': old_audit['summary']['positive_success'],
            'v5': new_audit['summary']['positive_success']},
        'negative_success': {'v4': old_audit['summary']['negative_success'],
            'v5': new_audit['summary']['negative_success']},
        'last_mean_code_norm': {
            'v4': old_audit['summary']['last_mean_code_norm'],
            'v5': new_audit['summary']['last_mean_code_norm']}}
    value = {'protocol': 'Descriptive paired train-only v4/v5 reward signal and score-gradient analysis; same 72 source-target pairs and noise draws; 17 different games are units for target-level rows, not independent validation',
             'v4_report_sha256': file_hash(args.old_report),
             'v4_audit_sha256': file_hash(args.old_audit),
             'v5_report_sha256': file_hash(args.new_report),
             'v5_audit_sha256': file_hash(args.new_audit),
             'summary': summary, 'targets': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False,
                                      indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_20261007.json'))
    parser.add_argument('--old-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_feedback_residual_train18_antithetic_v4_seed42_audited_20261007.json'))
    parser.add_argument('--new-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--new-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_signal_comparison_v5_20261007.json'))
    analyze(parser.parse_args())
