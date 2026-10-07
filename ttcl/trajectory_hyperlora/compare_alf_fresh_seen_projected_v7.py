"""Paired comparison of two frozen LoRA trainers on the same online stream."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def actions(episode):
    return [step['command'] for step in episode['trajectory']]


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.old_report.read_text())
    new = json.loads(args.new_report.read_text())
    old_audit = json.loads(args.old_audit.read_text())
    new_audit = json.loads(args.new_audit.read_text())
    if (old_audit['raw_report_sha256'] != file_hash(args.old_report) or
            new_audit['raw_report_sha256'] != file_hash(args.new_report) or
            old_audit['summary']['failed_replays'] != 0 or
            new_audit['summary']['failed_replays'] != 0 or
            old['plan_sha256'] != new['plan_sha256'] or
            old['review_sha256'] != new['review_sha256'] or
            old['checkpoint_sha256'] != new['checkpoint_sha256'] or
            old['source_tokens'] != new['source_tokens'] or
            old['max_steps'] != new['max_steps'] or
            old['max_new_tokens'] != new['max_new_tokens'] or
            old['actor_history_turns'] != new['actor_history_turns'] or
            old['loop_guard_max'] != new['loop_guard_max'] or
            old['residual_sha256'] == new['residual_sha256'] or
            old['failures'] or new['failures'] or
            len(old['games']) != 36 or len(new['games']) != 36):
        raise ValueError('Changed audited matching online protocol')
    totals = {'base': 0, 'raw_text': 0, 'v5_continuous': 0,
              'v5_fixed_first': 0, 'v6_continuous': 0,
              'v6_fixed_first': 0}
    discordance = {'v6_only_continuous': 0,
                   'v5_only_continuous': 0,
                   'v5_continuous_only_vs_fixed': 0,
                   'v5_fixed_only_vs_continuous': 0,
                   'v6_continuous_only_vs_fixed': 0,
                   'v6_fixed_only_vs_continuous': 0}
    action_differences = {'v5_vs_v6_continuous': 0,
                          'v5_continuous_vs_fixed': 0,
                          'v6_continuous_vs_fixed': 0}
    families = defaultdict(lambda: {key: 0 for key in totals})
    rows = []
    for a, b in zip(old['games'], new['games'], strict=True):
        match = ('index', 'family', 'game', 'game_sha256',
                 'input_content_sha256', 'prior_success_count',
                 'prior_source_records_sha256', 'prior_source_chain_sha256',
                 'source_vector_sha256', 'source_count_after',
                 'source_vector_sha256_after', 'raw_text_sha256',
                 'raw_text_original_tokens', 'raw_text_retained_tokens')
        if (any(a[key] != b[key] for key in match) or
                a['base'] != b['base'] or
                a['raw_text'] != b['raw_text'] or
                a.get('new_source_records_sha256') !=
                    b.get('new_source_records_sha256')):
            raise ValueError('Base history or text control differs across trainers')
        rewards = {'base': a['base']['reward'],
                   'raw_text': a['raw_text']['reward'],
                   'v5_continuous': a['continuous']['reward'],
                   'v5_fixed_first': a['fixed_first']['reward'],
                   'v6_continuous': b['continuous']['reward'],
                   'v6_fixed_first': b['fixed_first']['reward']}
        for key, reward in rewards.items():
            totals[key] += reward
            families[a['family']][key] += reward
        discordance['v6_only_continuous'] += (
            rewards['v6_continuous'] > rewards['v5_continuous'])
        discordance['v5_only_continuous'] += (
            rewards['v5_continuous'] > rewards['v6_continuous'])
        for prefix in ('v5', 'v6'):
            discordance[f'{prefix}_continuous_only_vs_fixed'] += (
                rewards[f'{prefix}_continuous'] >
                rewards[f'{prefix}_fixed_first'])
            discordance[f'{prefix}_fixed_only_vs_continuous'] += (
                rewards[f'{prefix}_continuous'] <
                rewards[f'{prefix}_fixed_first'])
        action_differences['v5_vs_v6_continuous'] += (
            actions(a['continuous']) != actions(b['continuous']))
        action_differences['v5_continuous_vs_fixed'] += (
            actions(a['continuous']) != actions(a['fixed_first']))
        action_differences['v6_continuous_vs_fixed'] += (
            actions(b['continuous']) != actions(b['fixed_first']))
        rows.append({'index': a['index'], 'family': a['family'],
                     'game_sha256': a['game_sha256'],
                     'prior_success_count': a['prior_success_count'],
                     'rewards': rewards})
    summary = {'games': 36, 'same_base_and_text_history': True,
               'totals': totals, 'paired_discordance': discordance,
               'action_trace_differences': action_differences,
               'families': dict(sorted(families.items()))}
    value = {'protocol': 'Paired same-game/same-own-source comparison of independently audited v5 full-score and v6 projected-score LoRA on precommitted official valid_seen online order; result remains one shared-source sequence, not multiple independent seeds',
             'v5_report_sha256': file_hash(args.old_report),
             'v5_audit_sha256': file_hash(args.old_audit),
             'v6_report_sha256': file_hash(args.new_report),
             'v6_audit_sha256': file_hash(args.new_audit),
             'summary': summary, 'games': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_shared_online_v6_20261007.json'))
    parser.add_argument('--old-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_shared_online_v6_audited_20261007.json'))
    parser.add_argument('--new-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_projected_shared_online_v7_20261007.json'))
    parser.add_argument('--new-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_projected_shared_online_v7_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_fresh_seen36_projected_comparison_v7_20261007.json'))
    analyze(parser.parse_args())
