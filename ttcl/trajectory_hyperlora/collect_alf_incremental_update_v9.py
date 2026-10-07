"""Collect future-reward credit for one additional own-success LoRA update.

Six chronological within-sequence source transitions are crossed with 18
official train targets. Each pair compares the same frozen actor with the
previous successful source alone and with its two-source running mean.
Family labels never enter the actor or update rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.experience_evolution.environment import make_env
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_context_vector_mean_v1 import update_fields, vector_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked as checked_parent, save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import CenteredEvidenceResidual
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import encode_source, target_feature
from ttcl.trajectory_hyperlora.train_alf_feedback_residual_v4_antithetic import ResidualInjection


TRANSITIONS = ((0, 1), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6))


def expected(args):
    sources, targets, _ = checked_parent(args)
    training = json.loads(args.training_report.read_text())
    audit = json.loads(args.training_audit.read_text())
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (args.target_policy != 'train18' or len(targets) != 18 or
            len(sources) != 8 or training['trained_model_sha256'] !=
            file_hash(args.residual) or training['max_pairs'] != 72 or
            training['failures'] or audit['raw_report_sha256'] !=
            file_hash(args.training_report) or
            audit['summary']['failed_replays'] != 0 or
            model['model_kind'] != 'centered_bilinear_action_sensitive_v5' or
            model['basis'].shape != (128, 8)):
        raise ValueError('Changed old source/target or v5 model lineage')
    rows = []
    for transition_index, (first_id, next_id) in enumerate(TRANSITIONS):
        first, next_source = sources[first_id], sources[next_id]
        if (first['sequence'] != next_source['sequence'] or
                first['source_index'] >= next_source['source_index']):
            raise ValueError('Source transition is not chronological')
        for target in targets:
            content = {'method': 'one_own_success_increment_v9',
                'transition_index': transition_index,
                'first_source_id': first_id, 'next_source_id': next_id,
                'first_records_sha256': first['records_sha256'],
                'next_records_sha256': next_source['records_sha256'],
                'first_source_game_sha256': first['game_sha256'],
                'next_source_game_sha256': next_source['game_sha256'],
                'target_id': target['target_id'],
                'target_game_sha256': target['game_sha256'],
                'checkpoint_sha256': file_hash(args.checkpoint),
                'residual_sha256': file_hash(args.residual),
                'parent_review_sha256': file_hash(args.review),
                'alpha': [0., .5], 'max_steps': 50}
            rows.append({**content, 'game': target['game'],
                'input_content_sha256': digest(content),
                'reviewed_target': True})
    if len(rows) != 108 or len({x['input_content_sha256'] for x in rows}) != 108:
        raise ValueError('Incomplete update counterfactual design')
    return sources, targets, model, rows


def prepare(args):
    if args.update_review.exists():
        raise FileExistsError(args.update_review)
    _, _, _, rows = expected(args)
    value = {'protocol': 'New content-bound chronological own-success update versus freeze pairs on official train targets; six transitions times 18 tasks; no outcome used in selection',
        'parent_review_sha256': file_hash(args.review),
        'training_report_sha256': file_hash(args.training_report),
        'training_audit_sha256': file_hash(args.training_audit),
        'residual_sha256': file_hash(args.residual),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'transitions': [list(pair) for pair in TRANSITIONS],
        'targets': rows}
    args.update_review.parent.mkdir(parents=True, exist_ok=True)
    args.update_review.write_text(json.dumps(value, ensure_ascii=False,
                                             indent=2) + '\n')
    print(json.dumps({'reviewed_update_pairs': len(rows),
                      'review_sha256': file_hash(args.update_review)}), flush=True)


def checked(args):
    sources, targets, model, rows = expected(args)
    value = json.loads(args.update_review.read_text())
    if (value['parent_review_sha256'] != file_hash(args.review) or
            value['training_report_sha256'] != file_hash(args.training_report) or
            value['training_audit_sha256'] != file_hash(args.training_audit) or
            value['residual_sha256'] != file_hash(args.residual) or
            value['checkpoint_sha256'] != file_hash(args.checkpoint) or
            value['transitions'] != [list(pair) for pair in TRANSITIONS] or
            value['targets'] != rows):
        raise ValueError('Changed update annotations or frozen lineage')
    return sources, targets, model, rows


def evaluate(args):
    sources, _, model, rows = checked(args)
    if not 1 <= args.max_pairs <= len(rows):
        raise ValueError('Requested rollout count is outside reviewed design')
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if (report['review_sha256'] != file_hash(args.update_review) or
                report['checkpoint_sha256'] != file_hash(args.checkpoint) or
                report['residual_sha256'] != file_hash(args.residual) or
                report['max_pairs'] != args.max_pairs or
                report['failures'] or len(report['rows']) > len(rows) or
                any(x['input_content_sha256'] != y['input_content_sha256']
                    for x, y in zip(report['rows'], rows))):
            raise ValueError('Cannot resume changed or failed update rollout')
    else:
        report = {'protocol': 'Paired official train one-increment counterfactual: same actor, task, old successful source, 50-step constrained greedy budget; alpha=0 freeze versus alpha=.5 two-source running mean; six chronological source transitions, no family slots',
            'review_sha256': file_hash(args.update_review),
            'parent_review_sha256': file_hash(args.review),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'residual_sha256': file_hash(args.residual),
            'max_steps': 50, 'max_new_tokens': 64,
            'max_pairs': args.max_pairs,
            'actor_history_turns': 2, 'loop_guard_max': 2,
            'rows': [], 'failures': []}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    correction = CenteredEvidenceResidual.from_state(
        model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device).float()
    source_cache = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    for index in range(len(report['rows']), args.max_pairs):
        row = rows[index]
        first_id, next_id = row['first_source_id'], row['next_source_id']
        entry = None
        try:
            for source_id in (first_id, next_id):
                if source_id not in source_cache:
                    source_cache[source_id] = encode_source(
                        agent, tokenizer, sources[source_id]['records'],
                        args.device)
            first_fields, first_global, first_delta, _, _ = source_cache[first_id]
            next_fields, next_global, next_delta, _, _ = source_cache[next_id]
            mean_fields = update_fields(first_fields, next_fields, 1)
            mean_global = (first_global + next_global) / 2
            mean_delta = (first_delta + next_delta) / 2
            game = args.data_root / row['game']
            if file_hash(game) != row['target_game_sha256']:
                raise ValueError('Changed official target game')
            env = make_env(game)
            try:
                initial = str(env.reset()['feedback'])
            finally:
                env.close()
            q = target_feature(agent, tokenizer, initial, args.device)
            entry = {'transition_index': row['transition_index'],
                'first_source_id': first_id, 'next_source_id': next_id,
                'target_id': row['target_id'], 'game': row['game'],
                'input_content_sha256': row['input_content_sha256'],
                'first_vector_sha256': vector_hash(first_fields),
                'updated_vector_sha256': vector_hash(mean_fields)}
            for arm, fields, global_vector, delta_vector in (
                    ('freeze', first_fields, first_global, first_delta),
                    ('update', mean_fields, mean_global, mean_delta)):
                with torch.no_grad():
                    code = correction(global_vector, delta_vector, q)
                    shift = code @ basis.T
                entry[f'{arm}_code_norm'] = float(code.norm())
                with ResidualInjection(agent, shift):
                    episode = run_episode(agent, tokenizer, game, fields,
                        adapter=True, device=args.device, max_steps=50,
                        max_new_tokens=64, constrain_actions=True,
                        actor_history_turns=2, loop_guard_max=2)
                entry[arm] = episode
                if (episode['status'] != 'complete' or
                        episode['invalid_commands'] != 0 or
                        episode['initial_observation'] != initial):
                    raise ValueError(f'Changed or incomplete {arm} episode')
            report['rows'].append(entry)
            save(args.output, report)
            print(json.dumps({'done': len(report['rows']),
                'freeze': sum(x['freeze']['reward'] for x in report['rows']),
                'update': sum(x['update']['reward'] for x in report['rows']),
                'update_only': sum(x['update']['reward'] > x['freeze']['reward']
                                   for x in report['rows']),
                'freeze_only': sum(x['update']['reward'] < x['freeze']['reward']
                                   for x in report['rows']),
                'failures': len(report['failures'])}), flush=True)
        except Exception as exc:
            report['failures'].append({'index': index, 'reason': repr(exc),
                                       'partial': entry})
            save(args.output, report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
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
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-pairs', type=int, default=108)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)
