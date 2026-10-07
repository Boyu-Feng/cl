"""Build a content-bound, train-only cross-source future-reward dataset.

Eight own successful trajectories cross 18 disjoint official ALFWorld train
problems. Family tags balance analysis only; the actor receives none.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields


SOURCE_INDICES = ((0, 3), (0, 7), (0, 11), (0, 15),
                  (6, 3), (6, 9), (6, 13), (0, 17))
FAMILIES = ('look_at_obj_in_light', 'pick_and_place_simple',
    'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
    'pick_heat_then_place_in_recep', 'pick_two_obj_and_place')


def source_reports(args):
    reports = {}
    for seq, path in ((0, args.source_report_seq0), (6, args.source_report_seq6)):
        report = json.loads(path.read_text())
        if (report['checkpoint_sha256'] != file_hash(args.checkpoint) or
                report['failures'] or len(report['games']) != 18):
            raise ValueError(f'Changed own source run for sequence {seq}')
        reports[seq] = report
    return reports


def expected(args):
    reports = source_reports(args)
    plan = json.loads(args.plan.read_text())
    selected_by_family = defaultdict(list)
    forbidden = set()
    forbidden_sequences = set()
    if args.target_policy == 'independent12':
        old_training = json.loads(args.checkpoint_training_candidates.read_text())
        warmstart = json.loads(args.warmstart_candidates.read_text())
        prior_review = json.loads(args.prior_train_review.read_text())
        if len(old_training['rows']) != 600 or len(prior_review['targets']) != 18:
            raise ValueError('Changed old training or cross-matrix source split')
        forbidden = {row['target_game'] for row in old_training['rows']}
        forbidden.update(row['target_game'] for row in warmstart['candidates']
                         if row['split'] == 'train')
        forbidden.update(row['game'] for row in prior_review['targets'])
        forbidden_sequences = {row['sequence_index'] for row in
                               prior_review['targets']}
    target_per_family = 2 if args.target_policy == 'independent12' else 3
    for sequence_index, sequence in enumerate(plan['training'][12:], start=12):
        if sequence_index in forbidden_sequences:
            continue
        family = sequence['family']
        if family in FAMILIES and len(selected_by_family[family]) < target_per_family:
            available = next((game for game in sequence['games']
                              if game not in forbidden), None)
            if available is not None:
                selected_by_family[family].append((sequence_index, available))
    if args.target_policy == 'independent12':
        # Some families have only one unused frozen sequence. The second
        # problem may share that held-out sequence but never a training one.
        for sequence_index, sequence in enumerate(plan['training'][12:], start=12):
            family = sequence['family']
            if (sequence_index in forbidden_sequences or
                    family not in FAMILIES or
                    len(selected_by_family[family]) >= target_per_family):
                continue
            used = {game for _, game in selected_by_family[family]}
            available = next((game for game in sequence['games']
                              if game not in forbidden and game not in used), None)
            if available is not None:
                selected_by_family[family].append((sequence_index, available))
    if set(selected_by_family) != set(FAMILIES) or any(
            len(selected_by_family[family]) != target_per_family
            for family in FAMILIES):
        raise ValueError('Plan lacks disjoint target sequences per family')
    targets = []
    for family in FAMILIES:
        for family_position, (sequence_index, game) in enumerate(selected_by_family[family]):
            targets.append({'target_id': len(targets), 'family': family,
                'family_position': family_position,
                'sequence_index': sequence_index, 'game': game,
                'game_sha256': file_hash(args.data_root / game)})
    sources = []
    for seq, index in SOURCE_INDICES:
        row = reports[seq]['games'][index]
        episode = row['episode']
        if episode['status'] != 'complete' or episode['reward'] != 1:
            raise ValueError('Selected own source did not win')
        records = records_from_episode(episode)
        if row['new_records_sha256'] != digest(records):
            raise ValueError('Selected own source trajectory changed')
        sources.append({'source_id': len(sources), 'sequence': seq,
            'source_index': index, 'game': row['game'],
            'game_sha256': file_hash(args.data_root / row['game']),
            'records_sha256': digest(records),
            'report_sha256': file_hash(args.source_report_seq0 if seq == 0
                                       else args.source_report_seq6),
            'records': records})
    source_games = {row['game'] for row in sources}
    if (len(sources) != 8 or len(targets) != 6 * target_per_family or
            len({row['game'] for row in targets}) != len(targets) or
            source_games & {row['game'] for row in targets}):
        raise ValueError('Source/target overlap or missing candidates')
    core = {'method': ('own_success_cross_future_reward_holdout_v3c'
                       if args.target_policy == 'independent12' else
                       'own_success_cross_future_reward_v2'),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'plan_sha256': file_hash(args.plan),
        'source_report_seq0_sha256': file_hash(args.source_report_seq0),
        'source_report_seq6_sha256': file_hash(args.source_report_seq6),
        'sources': [{k: v for k, v in row.items() if k != 'records'}
                    for row in sources], 'targets': targets,
        'max_steps': 50, 'max_new_tokens': 64,
        'actor_history_turns': 2, 'loop_guard_max': 2,
        'context_tokens': 2048}
    if args.target_policy == 'independent12':
        core['target_policy'] = args.target_policy
        core['checkpoint_training_candidates_sha256'] = file_hash(
            args.checkpoint_training_candidates)
        core['warmstart_candidates_sha256'] = file_hash(
            args.warmstart_candidates)
        core['prior_train_review_sha256'] = file_hash(args.prior_train_review)
    pairs = []
    for source in sources:
        for target in targets:
            binding = {'method': core['method'], 'source_id': source['source_id'],
                'target_id': target['target_id'],
                'source_records_sha256': source['records_sha256'],
                'source_report_sha256': source['report_sha256'],
                'target_game_sha256': target['game_sha256'],
                'checkpoint_sha256': core['checkpoint_sha256'],
                'plan_sha256': core['plan_sha256']}
            pairs.append({**binding, 'input_content_sha256': digest(binding),
                'reviewed_target': True})
    return core, sources, targets, pairs


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    core, _, _, pairs = expected(args)
    value = {'protocol': 'Fresh content-bound own-success sources and disjoint official train targets; frozen target order and budget',
        **core, 'pair_bindings': pairs}
    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'sources': len(core['sources']),
                      'targets': len(core['targets']), 'pairs': len(pairs)}), flush=True)


def checked(args):
    core, sources, targets, pairs = expected(args)
    review = json.loads(args.review.read_text())
    if (any(review[key] != val for key, val in core.items()) or
            review['pair_bindings'] != pairs):
        raise ValueError('Unreviewed or changed future-reward source/target content')
    return sources, targets, pairs


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def evaluate(args):
    sources, targets, pairs = checked(args)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        report = json.loads(args.output.read_text())
        if (report['review_sha256'] != file_hash(args.review) or
                report['checkpoint_sha256'] != file_hash(args.checkpoint) or
                report['failures'] or len(report['base']) > len(targets) or
                len(report['pairs']) > len(pairs)):
            raise ValueError('Cannot resume changed or failed reward collection')
        for index, row in enumerate(report['base']):
            if row['target_id'] != targets[index]['target_id'] or row['game'] != targets[index]['game']:
                raise ValueError('Changed base prefix')
        for index, row in enumerate(report['pairs']):
            if (row['input_content_sha256'] != pairs[index]['input_content_sha256'] or
                    row['source_id'] != pairs[index]['source_id'] or
                    row['target_id'] != pairs[index]['target_id']):
                raise ValueError('Changed source-target prefix')
    else:
        report = {'protocol': f'Frozen actor; 8 reviewed own successes crossed with {len(targets)} disjoint official train tasks; paired base once per task; identical constrained 50-step greedy actor, two-turn history, two-repeat loop guard; source family excluded from actor',
            'review_sha256': file_hash(args.review),
            'checkpoint_sha256': file_hash(args.checkpoint),
            'max_steps': 50, 'max_new_tokens': 64,
            'actor_history_turns': 2, 'loop_guard_max': 2,
            'context_tokens': 2048,
            'base': [], 'pairs': [], 'failures': []}
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen contextual task-conditioned hypernetwork')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target in targets[len(report['base']):]:
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], {}, adapter=False,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        report['base'].append({'target_id': target['target_id'],
            'game': target['game'], 'game_sha256': target['game_sha256'],
            'episode': episode})
        if episode['status'] != 'complete':
            report['failures'].append({'phase': 'base', 'target_id': target['target_id']})
        save(args.output, report)
        print(json.dumps({'base': len(report['base']),
            'pairs': len(report['pairs']), 'failures': len(report['failures'])}), flush=True)
        if report['failures']:
            raise RuntimeError('Base reward collection failure recorded')
    source_cache = {}
    for index in range(len(report['pairs']), len(pairs)):
        binding = pairs[index]
        source = sources[binding['source_id']]
        target = targets[binding['target_id']]
        if source['source_id'] not in source_cache:
            bounded, original, retained = bounded_source_text(
                tokenizer, source['records'], 2048)
            fields = contextual_text_fields(agent, tokenizer, bounded,
                args.device, 2048, pooling='both')
            source_cache[source['source_id']] = (fields, original, retained)
        fields, original, retained = source_cache[source['source_id']]
        episode = run_episode(agent, tokenizer,
            args.data_root / target['game'], fields, adapter=True,
            device=args.device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
        report['pairs'].append({'source_id': source['source_id'],
            'target_id': target['target_id'],
            'input_content_sha256': binding['input_content_sha256'],
            'source_tokens_original': original,
            'source_tokens_retained': retained,
            'episode': episode})
        if episode['status'] != 'complete':
            report['failures'].append({'phase': 'pair', 'pair_index': index})
        save(args.output, report)
        print(json.dumps({'base': len(report['base']),
            'pairs': len(report['pairs']), 'success': sum(
                x['episode'].get('reward', 0) for x in report['pairs']),
            'failures': len(report['failures'])}), flush=True)
        if report['failures']:
            raise RuntimeError('Source reward collection failure recorded')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18', 'independent12'),
                        default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path(
        'data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)

if __name__ == '__main__':
    main()
