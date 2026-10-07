"""Collect an additional reviewed 8x18 own-history future-reward matrix.

Target games are distinct from the original 8x18 matrix, its 8x12 disjoint
development set, checkpoint candidate pool, warmstart, and all source games.
The official train split supplies more reward evidence for learning a
trajectory-conditioned LoRA update, not an independent final evaluation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import (
    FAMILIES, SOURCE_INDICES, save, source_reports,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


def expected(args):
    reports = source_reports(args)
    plan = json.loads(args.plan.read_text())
    first = json.loads(args.first_review.read_text())
    second = json.loads(args.second_review.read_text())
    old = json.loads(args.checkpoint_candidates.read_text())
    warm = json.loads(args.warmstart_candidates.read_text())
    if (len(old['rows']) != 600 or len(first['sources']) != 8 or
            len(first['targets']) != 18 or len(second['targets']) != 12 or
            first['checkpoint_sha256'] != file_hash(args.checkpoint) or
            second['checkpoint_sha256'] != file_hash(args.checkpoint)):
        raise ValueError('Changed source bank or ancestor training lineage')
    sources = []
    for source_id, (seq, index) in enumerate(SOURCE_INDICES):
        row = reports[seq]['games'][index]
        episode = row['episode']
        records = records_from_episode(episode)
        source = first['sources'][source_id]
        if (episode['status'] != 'complete' or episode['reward'] != 1. or
                row['game'] != source['game'] or
                row['new_records_sha256'] != digest(records) or
                source['records_sha256'] != digest(records) or
                source['report_sha256'] != file_hash(
                    args.source_report_seq0 if seq == 0 else
                    args.source_report_seq6)):
            raise ValueError('Changed self-success history')
        sources.append({**source, 'records': records})
    forbidden = {x['target_game'] for x in old['rows']}
    forbidden.update(x['target_game'] for x in warm['candidates']
                     if x['split'] == 'train')
    forbidden.update(x['game'] for x in first['targets']+second['targets'])
    forbidden.update(x['game'] for x in sources)
    available = defaultdict(list)
    for index, sequence in enumerate(plan['training'][12:], start=12):
        if sequence['family'] not in FAMILIES:
            continue
        for game in sequence['games']:
            if game not in forbidden:
                available[sequence['family']].append((index, game))
    targets = []
    for family in FAMILIES:
        selected = []
        selected_sequences = set()
        for index, game in available[family]:
            if index not in selected_sequences:
                selected.append((index, game))
                selected_sequences.add(index)
            if len(selected) == 3:
                break
        if len(selected) < 3:
            used_games = {game for _, game in selected}
            for index, game in available[family]:
                if game not in used_games:
                    selected.append((index, game))
                    used_games.add(game)
                if len(selected) == 3:
                    break
        if len(selected) != 3:
            raise ValueError(f'Insufficient fresh official train games: {family}')
        for position, (sequence_index, game) in enumerate(selected):
            targets.append({'target_id': len(targets),
                'family': family, 'family_position': position,
                'sequence_index': sequence_index, 'game': game,
                'game_sha256': file_hash(args.data_root / game)})
    if len({x['game'] for x in targets}) != 18:
        raise ValueError('Duplicate new target game')
    core = {'method': 'own_success_cross_future_reward_additional_train_v3',
        'checkpoint_sha256': file_hash(args.checkpoint),
        'plan_sha256': file_hash(args.plan),
        'source_report_seq0_sha256': file_hash(args.source_report_seq0),
        'source_report_seq6_sha256': file_hash(args.source_report_seq6),
        'first_review_sha256': file_hash(args.first_review),
        'second_review_sha256': file_hash(args.second_review),
        'checkpoint_candidates_sha256': file_hash(args.checkpoint_candidates),
        'warmstart_candidates_sha256': file_hash(args.warmstart_candidates),
        'sources': [{k: v for k, v in source.items() if k != 'records'}
                    for source in sources],
        'targets': targets, 'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'context_tokens': 2048}
    pairs = []
    for source in sources:
        for target in targets:
            content = {'method': core['method'],
                'source_id': source['source_id'],
                'target_id': target['target_id'],
                'source_records_sha256': source['records_sha256'],
                'source_report_sha256': source['report_sha256'],
                'target_game_sha256': target['game_sha256'],
                'checkpoint_sha256': core['checkpoint_sha256'],
                'plan_sha256': core['plan_sha256'],
                'first_review_sha256': core['first_review_sha256'],
                'second_review_sha256': core['second_review_sha256']}
            pairs.append({**content,
                'input_content_sha256': digest(content),
                'reviewed_target': True})
    return core, sources, targets, pairs


def checked(args):
    core, sources, targets, pairs = expected(args)
    reviewed = json.loads(args.review.read_text())
    if (any(reviewed[key] != value for key, value in core.items()) or
            reviewed['pair_bindings'] != pairs):
        raise ValueError('Changed reviewed source-target inputs')
    return sources, targets, pairs


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    core, _, _, pairs = expected(args)
    result = {'protocol': 'Fresh content-bound additional 8x18 official train future-reward source-target matrix; game-disjoint from prior 8x18/8x12 and checkpoint candidates',
        **core, 'pair_bindings': pairs}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'sources': 8, 'targets': 18,
                      'pair_bindings': len(pairs)}), flush=True)


def collect(args):
    sources, targets, pairs = checked(args)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        result = json.loads(args.output.read_text())
        if (result['review_sha256'] != file_hash(args.review) or
                result['checkpoint_sha256'] != file_hash(args.checkpoint) or
                result['failures'] or len(result['base']) > 18 or
                len(result['pairs']) > 144):
            raise ValueError('Cannot resume changed or failed collection')
    else:
        result = {'protocol': 'Frozen old hypernetwork and own-success sources crossed with eighteen new official train games; paired 50-step/64-token/constrained greedy/two-turn/two-repeat actor',
            'review_sha256': file_hash(args.review),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'max_steps': 50, 'max_new_tokens': 64,
            'actor_history_turns': 2, 'loop_guard_max': 2,
            'context_tokens': 2048,
            'base': [], 'pairs': [], 'failures': []}
        save(args.output, result)
    for i, row in enumerate(result['base']):
        if (row['target_id'] != i or row['game'] != targets[i]['game'] or
                row['game_sha256'] != targets[i]['game_sha256'] or
                row['episode']['status'] != 'complete'):
            raise ValueError('Changed base result prefix')
    for i, row in enumerate(result['pairs']):
        if (row['input_content_sha256'] != pairs[i]['input_content_sha256'] or
                row['episode']['status'] != 'complete'):
            raise ValueError('Changed source result prefix')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen source-conditioned actor')
    for index in range(len(result['base']), len(targets)):
        target = targets[index]
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], {}, adapter=False,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        result['base'].append({'target_id': index, 'game': target['game'],
            'game_sha256': target['game_sha256'], 'episode': episode})
        if episode['status'] != 'complete':
            result['failures'].append({'arm': 'base', 'target_id': index})
        save(args.output, result)
        print(json.dumps({'base': len(result['base']),
            'reward': episode.get('reward')}), flush=True)
        if result['failures']:
            raise RuntimeError('Base collection failure recorded')
    source_cache = {}
    for index in range(len(result['pairs']), len(pairs)):
        binding = pairs[index]
        source = sources[binding['source_id']]
        target = targets[binding['target_id']]
        if source['source_id'] not in source_cache:
            bounded, original, retained = bounded_source_text(
                tokenizer, source['records'], 2048)
            fields = contextual_text_fields(agent, tokenizer,
                bounded, args.device, 2048, pooling='both')
            source_cache[source['source_id']] = fields, original, retained
        fields, original, retained = source_cache[source['source_id']]
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], fields, adapter=True,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
        result['pairs'].append({'source_id': source['source_id'],
            'target_id': target['target_id'],
            'input_content_sha256': binding['input_content_sha256'],
            'source_tokens_original': original,
            'source_tokens_retained': retained,
            'episode': episode})
        if (episode['status'] != 'complete' or
                episode['initial_observation'] !=
                    result['base'][target['target_id']]['episode']['initial_observation']):
            result['failures'].append({'arm': 'source',
                'pair_index': index, 'reason': episode.get('error')})
        save(args.output, result)
        print(json.dumps({'pairs': len(result['pairs']),
            'of': len(pairs), 'reward': episode.get('reward'),
            'failures': len(result['failures'])}), flush=True)
        if result['failures']:
            raise RuntimeError('Source collection failure recorded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'collect'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
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
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        collect(args)
