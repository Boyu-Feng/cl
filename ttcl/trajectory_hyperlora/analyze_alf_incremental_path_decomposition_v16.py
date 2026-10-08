"""Exact path-dependent action/continuation decomposition for v12 probes.

No new episodes: validates identical pre-action prefixes and four audited
terminal outcomes for each of the 16 selected train-domain pairs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash


def analyze(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads(args.parent.read_text())
    probes = json.loads(args.probes.read_text())
    audit = json.loads(args.audit.read_text())
    review = json.loads(args.review.read_text())
    if (audit['raw_report_sha256'] != file_hash(args.probes) or
            audit['parent_report_sha256'] != file_hash(args.parent) or
            audit['review_sha256'] != file_hash(args.review) or
            audit['summary']['failed_replays'] != 0 or
            len(probes['rows']) != 16 or len(review['targets']) != 16):
        raise ValueError('Changed independently replayed v12 intervention')
    rows = []
    classes = Counter()
    for selected, probe in zip(review['targets'], probes['rows'], strict=True):
        pair = parent['rows'][selected['pair_index']]
        split = selected['split']
        freeze = pair['freeze']
        update = pair['update']
        if (probe['pair_index'] != selected['pair_index'] or
                probe['input_content_sha256'] != selected['input_content_sha256'] or
                freeze['trajectory'][:split] != update['trajectory'][:split] or
                freeze['initial_observation'] != update['initial_observation'] or
                freeze['trajectory'][split]['command'] ==
                    update['trajectory'][split]['command']):
            raise ValueError('No exact common pre-action history')
        f = freeze['reward']
        u = update['reward']
        fs = probe['arms']['freeze']['swapped_reward']
        us = probe['arms']['update']['swapped_reward']
        if (probe['arms']['freeze']['original_reward'] != f or
                probe['arms']['update']['original_reward'] != u or
                probe['arms']['freeze']['alternate_action'] !=
                    update['trajectory'][split]['command'] or
                probe['arms']['update']['alternate_action'] !=
                    freeze['trajectory'][split]['command']):
            raise ValueError('Changed four potential outcomes')
        total = u - f
        action_under_freeze = fs - f
        action_under_update = u - us
        continuation_at_freeze_action = us - f
        continuation_at_update_action = u - fs
        interaction = action_under_update - action_under_freeze
        if (total != action_under_freeze + continuation_at_update_action or
                total != action_under_update + continuation_at_freeze_action or
                interaction != continuation_at_update_action -
                    continuation_at_freeze_action):
            raise ValueError('Invalid exact path decomposition')
        if fs != f and us != u:
            category = 'action_effect_both_policies'
        elif fs == f and us == u:
            category = 'continuation_effect_both_actions'
        else:
            category = 'action_continuation_interaction'
        classes[category] += 1
        rows.append({'pair_index': selected['pair_index'],
            'split': split, 'total_update_minus_freeze': total,
            'action_effect_under_freeze': action_under_freeze,
            'action_effect_under_update': action_under_update,
            'continuation_effect_at_freeze_action': continuation_at_freeze_action,
            'continuation_effect_at_update_action': continuation_at_update_action,
            'action_continuation_interaction': interaction,
            'category': category})
    value = {'protocol': 'Exact four-outcome path decomposition at identical first-action history for selected terminal-discordant train pairs; not population mediation or independent evaluation',
        'parent_report_sha256': file_hash(args.parent),
        'intervention_report_sha256': file_hash(args.probes),
        'intervention_audit_sha256': file_hash(args.audit),
        'review_sha256': file_hash(args.review),
        'summary': {'pairs': 16, 'categories': dict(classes),
            'total_update_minus_freeze': sum(
                row['total_update_minus_freeze'] for row in rows),
            'sum_action_effect_under_freeze': sum(
                row['action_effect_under_freeze'] for row in rows),
            'sum_action_effect_under_update': sum(
                row['action_effect_under_update'] for row in rows),
            'sum_continuation_effect_at_freeze_action': sum(
                row['continuation_effect_at_freeze_action'] for row in rows),
            'sum_continuation_effect_at_update_action': sum(
                row['continuation_effect_at_update_action'] for row in rows)},
        'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(value['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--probes', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_action16_v12_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_action16_v12_audited_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_incremental_action16_v12_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_path_decomposition_v16_20261008.json'))
    analyze(parser.parse_args())
