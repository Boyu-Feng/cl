"""Collect paired future reward for one own-success LoRA update on new games.

Targets were frozen by freeze_alf_incremental_pool_v14 before any outcomes.
Three disjoint target shards may run on separate currently free GPUs.
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


def checked(args):
    sources, _, _ = checked_parent(args)
    review = json.loads(args.pool_review.read_text())
    parent_sources = [{key: source[key] for key in
        ('source_id', 'sequence', 'source_index', 'game', 'game_sha256',
         'records_sha256', 'report_sha256')}
        for source in sources]
    if (review['sources'] != parent_sources or
            review['parent_review_sha256'] != file_hash(args.review) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['residual_sha256'] != file_hash(args.residual) or
            review['selection_seed'] != 'incremental-v14' or
            len(review['targets']) != 78 or len(review['pairs']) != 468 or
            any(row['directory_previously_exposed']
                for row in review['targets'] if row['split'] == 'dev')):
        raise ValueError('Changed frozen target pool or own-success sources')
    if not 0 <= args.shard < 3:
        raise ValueError('Shard must be 0, 1 or 2')
    for binding in review['pairs']:
        target = review['targets'][binding['target_id']]
        old_id, new_id = review['transitions'][binding['transition_index']]
        content = {'method': 'incremental-v14',
            'target_id': target['target_id'], 'split': target['split'],
            'target_game_sha256': target['game_sha256'],
            'transition_index': binding['transition_index'],
            'old_records_sha256': parent_sources[old_id]['records_sha256'],
            'new_records_sha256': parent_sources[new_id]['records_sha256'],
            'checkpoint_sha256': review['checkpoint_sha256'],
            'residual_sha256': review['residual_sha256'],
            'alpha': [0., .5], 'max_steps': 50}
        if binding != {**content, 'input_content_sha256': digest(content),
                       'reviewed_target': True}:
            raise ValueError('Changed input-content binding')
    scheduled = [binding for binding in review['pairs']
                 if binding['target_id'] % 3 == args.shard]
    if len(scheduled) != 156:
        raise ValueError('Incomplete target shard')
    for row in scheduled:
        target = review['targets'][row['target_id']]
        if (row['target_game_sha256'] != target['game_sha256'] or
                row['split'] != target['split'] or
                row['checkpoint_sha256'] != review['checkpoint_sha256'] or
                row['residual_sha256'] != review['residual_sha256'] or
                file_hash(args.data_root / target['game']) !=
                    target['game_sha256']):
            raise ValueError('Changed reviewed target content')
    model = torch.load(args.residual, map_location='cpu', weights_only=True)
    if (model['model_kind'] != 'centered_bilinear_action_sensitive_v5' or
            tuple(model['basis'].shape) != (128, 8)):
        raise ValueError('Changed frozen correction model')
    return sources, review, scheduled, model


def evaluate(args):
    sources, review, schedule, model = checked(args)
    if args.validate_only:
        print(json.dumps({'shard': args.shard, 'pairs': len(schedule),
            'train_pairs': sum(row['split'] == 'train' for row in schedule),
            'dev_pairs': sum(row['split'] == 'dev' for row in schedule),
            'pool_review_sha256': file_hash(args.pool_review)}), flush=True)
        return
    output = args.output or Path(
        f'results/trajectory_hyperlora/alf_incremental_pool78_v14_shard{args.shard}_20261008.json')
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        report = json.loads(output.read_text())
        if (report['pool_review_sha256'] != file_hash(args.pool_review) or
                report['checkpoint_sha256'] != file_hash(args.checkpoint) or
                report['residual_sha256'] != file_hash(args.residual) or
                report['shard'] != args.shard or report['failures'] or
                len(report['rows']) > len(schedule) or
                any(row['input_content_sha256'] != binding['input_content_sha256']
                    for row, binding in zip(report['rows'], schedule))):
            raise ValueError('Cannot resume altered or failed shard')
    else:
        report = {'protocol': 'Frozen v14 train/dev pool, one chronological own-success update against freeze; same constrained 50-step greedy actor and original environment; train and dev games from distinct task directories',
            'pool_review_sha256': file_hash(args.pool_review),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'residual_sha256': file_hash(args.residual),
            'shard': args.shard, 'num_shards': 3,
            'max_steps': 50, 'max_new_tokens': 64,
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
    correction = CenteredEvidenceResidual.from_state(model['correction']).to(args.device)
    correction.load_state_dict(model['correction'])
    correction.eval()
    basis = model['basis'].to(args.device).float()
    source_cache = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    for position in range(len(report['rows']), len(schedule)):
        binding = schedule[position]
        target = review['targets'][binding['target_id']]
        old_id, new_id = review['transitions'][binding['transition_index']]
        entry = {'position': position, 'target_id': target['target_id'],
            'split': target['split'], 'family': target['family'],
            'transition_index': binding['transition_index'],
            'old_source_id': old_id, 'new_source_id': new_id,
            'game': target['game'],
            'input_content_sha256': binding['input_content_sha256']}
        try:
            for source_id in (old_id, new_id):
                if source_id not in source_cache:
                    source_cache[source_id] = encode_source(agent, tokenizer,
                        sources[source_id]['records'], args.device)
            old_f, old_g, old_d, _, _ = source_cache[old_id]
            new_f, new_g, new_d, _, _ = source_cache[new_id]
            mean_f = update_fields(old_f, new_f, 1)
            mean_g = (old_g + new_g) / 2
            mean_d = (old_d + new_d) / 2
            game = args.data_root / target['game']
            env = make_env(game)
            try:
                initial = str(env.reset()['feedback'])
            finally:
                env.close()
            q = target_feature(agent, tokenizer, initial, args.device)
            entry['old_vector_sha256'] = vector_hash(old_f)
            entry['updated_vector_sha256'] = vector_hash(mean_f)
            for arm, fields, global_v, delta_v in (
                    ('freeze', old_f, old_g, old_d),
                    ('update', mean_f, mean_g, mean_d)):
                with torch.no_grad():
                    code = correction(global_v, delta_v, q)
                    shift = code @ basis.T
                entry[f'{arm}_code_norm'] = float(code.norm())
                with ResidualInjection(agent, shift):
                    episode = run_episode(agent, tokenizer, game, fields,
                        adapter=True, device=args.device, max_steps=50,
                        max_new_tokens=64, constrain_actions=True,
                        actor_history_turns=2, loop_guard_max=2)
                if (episode['status'] != 'complete' or
                        episode['initial_observation'] != initial or
                        episode['invalid_commands'] != 0):
                    raise ValueError(f'Incomplete {arm} episode')
                entry[arm] = episode
            report['rows'].append(entry)
            save(output, report)
            print(json.dumps({'shard': args.shard,
                'done': len(report['rows']), 'split': target['split'],
                'freeze': int(entry['freeze']['reward']),
                'update': int(entry['update']['reward'])}), flush=True)
        except Exception as exc:
            report['failures'].append({'position': position,
                'reason': repr(exc), 'partial': entry})
            save(output, report)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    evaluate(parser.parse_args())
