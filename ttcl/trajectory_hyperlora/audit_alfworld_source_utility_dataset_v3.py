"""Replay all new 8x18 source-target episodes against official ALFWorld games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 import checked


def replay(game: Path, episode: dict):
    if (episode['status'] != 'complete' or
            episode['steps'] != len(episode['trajectory']) or
            episode['steps'] > 50 or episode['steps'] < 1 or
            episode['invalid_commands'] != 0):
        raise ValueError('Incomplete or invalid frozen actor episode')
    env = make_env(game)
    try:
        state = env.reset()
        if str(state['feedback']) != episode['initial_observation']:
            raise ValueError('Changed official game reset')
        for index, step in enumerate(episode['trajectory']):
            commands = list(state['admissible_commands'])
            if (step['turn'] != index or
                    step['command'] not in commands or
                    step['valid'] is not True):
                raise ValueError('Changed actor action admissibility')
            state, _, done = env.step(step['command'])
            if (str(state['feedback']) != step['observation'] or
                    bool(state['won']) != step['won'] or
                    done and index != episode['steps']-1):
                raise ValueError('Changed official game transition')
        won = float(bool(state['won']))
        if (won != episode['reward'] or
                episode['termination'] !=
                ('success' if won else 'budget_or_environment_done')):
            raise ValueError('Changed official terminal reward')
        return won
    finally:
        env.close()


def audit(args):
    if args.audit_output.exists():
        raise FileExistsError(args.audit_output)
    sources, targets, bindings = checked(args)
    report = json.loads(args.output.read_text())
    if (report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(report['base']) != 18 or
            len(report['pairs']) != 144 or
            report['max_steps'] != 50 or
            report['max_new_tokens'] != 64 or
            report['actor_history_turns'] != 2 or
            report['loop_guard_max'] != 2 or
            report['context_tokens'] != 2048):
        raise ValueError('Changed or incomplete additional cross-history run')
    base = []
    for target, row in zip(targets, report['base'], strict=True):
        if (row['target_id'] != target['target_id'] or
                row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256']):
            raise ValueError('Changed reviewed new target')
        base.append(replay(args.data_root / target['game'], row['episode']))
    rewards = [[None]*len(targets) for _ in sources]
    for binding, actual in zip(bindings, report['pairs'], strict=True):
        si, ti = binding['source_id'], binding['target_id']
        target = targets[ti]
        if (actual['input_content_sha256'] !=
                binding['input_content_sha256'] or
                actual['source_id'] != si or
                actual['target_id'] != ti or
                actual['source_tokens_retained'] > 2048 or
                actual['episode']['initial_observation'] !=
                    report['base'][ti]['episode']['initial_observation']):
            raise ValueError('Changed source-target input binding')
        rewards[si][ti] = replay(args.data_root / target['game'],
                                 actual['episode'])
    if any(value is None for row in rewards for value in row):
        raise ValueError('Missing reviewed source-target cell')
    source_totals = {str(si): sum(row) for si, row in enumerate(rewards)}
    summary = {'sources': len(sources), 'targets': len(targets),
        'pairs': len(bindings), 'base_success': sum(base),
        'pair_success': sum(sum(row) for row in rewards),
        'source_dependent_targets': sum(len({rewards[si][ti]
            for si in range(len(sources))}) > 1 for ti in range(len(targets))),
        'best_static_source': max(source_totals.values()),
        'posthoc_oracle': sum(max(rewards[si][ti]
            for si in range(len(sources))) for ti in range(len(targets))),
        'failed_replays': 0, 'source_totals': source_totals}
    result = {'protocol': 'Independent original-environment replay of all base and source-conditioned episodes in additional reviewed official-train 8x18 future-reward matrix; not final validation',
        'review_sha256': file_hash(args.review),
        'raw_report_sha256': file_hash(args.output),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': summary}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, ensure_ascii=False,
                                             indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--checkpoint-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--first-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--second-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_holdout_8x12_v3c_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_additional8x18_v3_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_20261007.json'))
    parser.add_argument('--audit-output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_additional8x18_v3_audited_20261007.json'))
    audit(parser.parse_args())
