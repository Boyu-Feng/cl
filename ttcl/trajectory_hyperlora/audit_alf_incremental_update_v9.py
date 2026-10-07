"""Replay every train-domain update-versus-freeze trajectory in ALFWorld."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.audit_alf_action_sensitive_reward_v5 import replay
from ttcl.trajectory_hyperlora.collect_alf_incremental_update_v9 import checked


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    sources, targets, _, reviewed = checked(args)
    report = json.loads(args.output.read_text())
    if (report['review_sha256'] != file_hash(args.update_review) or
            report['parent_review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['residual_sha256'] != file_hash(args.residual) or
            report['max_pairs'] != 108 or
            report['max_steps'] != 50 or report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or report['failures'] or
            len(report['rows']) != 108):
        raise ValueError('Changed or incomplete incremental update run')
    totals = {'freeze': 0., 'update': 0.,
              'update_only': 0, 'freeze_only': 0,
              'changed_trajectories': 0}
    per_transition = []
    previous_vectors = {}
    for target, row in zip(reviewed, report['rows'], strict=True):
        if (row['transition_index'] != target['transition_index'] or
                row['first_source_id'] != target['first_source_id'] or
                row['next_source_id'] != target['next_source_id'] or
                row['target_id'] != target['target_id'] or
                row['game'] != target['game'] or
                row['input_content_sha256'] != target['input_content_sha256'] or
                file_hash(args.data_root / target['game']) !=
                target['target_game_sha256'] or
                sources[row['first_source_id']]['records_sha256'] !=
                target['first_records_sha256'] or
                sources[row['next_source_id']]['records_sha256'] !=
                target['next_records_sha256'] or
                any(not math.isfinite(row[f'{arm}_code_norm'])
                    for arm in ('freeze', 'update'))):
            raise ValueError('Changed reviewed source, target, or LoRA state')
        transition = row['transition_index']
        vectors = (row['first_vector_sha256'],
                   row['updated_vector_sha256'])
        if transition in previous_vectors and previous_vectors[transition] != vectors:
            raise ValueError('Same transition changed vectors across targets')
        previous_vectors[transition] = vectors
        initial = row['freeze']['initial_observation']
        for arm in ('freeze', 'update'):
            if (row[arm]['status'] != 'complete' or
                    row[arm]['steps'] > 50 or
                    row[arm]['invalid_commands'] != 0 or
                    row[arm]['initial_observation'] != initial):
                raise ValueError('Changed actor protocol or initial state')
            totals[arm] += replay(args.data_root / target['game'],
                                  row[arm], initial)
        totals['update_only'] += (
            row['update']['reward'] > row['freeze']['reward'])
        totals['freeze_only'] += (
            row['update']['reward'] < row['freeze']['reward'])
        totals['changed_trajectories'] += (
            row['update']['trajectory'] != row['freeze']['trajectory'])
    for transition in range(6):
        rows = [x for x in report['rows']
                if x['transition_index'] == transition]
        if len(rows) != 18:
            raise ValueError('Incomplete per-transition target matrix')
        per_transition.append({'transition_index': transition,
            'first_source_id': rows[0]['first_source_id'],
            'next_source_id': rows[0]['next_source_id'],
            'freeze': sum(x['freeze']['reward'] for x in rows),
            'update': sum(x['update']['reward'] for x in rows),
            'update_only': sum(x['update']['reward'] > x['freeze']['reward']
                               for x in rows),
            'freeze_only': sum(x['update']['reward'] < x['freeze']['reward']
                               for x in rows)})
    value = {'protocol': 'Independent original-environment replay of every paired train-domain freeze/update episode with new source/target content bindings',
        'raw_report_sha256': file_hash(args.output),
        'review_sha256': file_hash(args.update_review),
        'summary': {'pairs': 108, 'rollouts': 216,
                    'different_targets': len(targets),
                    **totals, 'failed_replays': 0},
        'per_transition': per_transition}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(value, ensure_ascii=False,
                                             indent=2) + '\n')
    print(json.dumps(value['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path(
        'ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18',), default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--residual', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.pt'))
    parser.add_argument('--training-report', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_20261007.json'))
    parser.add_argument('--training-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_action_sensitive_reward_train72_v5_audited_20261007.json'))
    parser.add_argument('--update-review', type=Path, default=Path(
        'data/annotations/alf_incremental_update108_v9_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_update108_v9_audited_20261007.json'))
    audit(parser.parse_args())
