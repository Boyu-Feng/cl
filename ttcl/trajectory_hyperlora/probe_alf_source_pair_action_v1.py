"""Test source-pair action contrasts under one fixed winning-history actor.

The forced continuation invokes the exact 50-step/two-turn/loop-guard actor
through run_episode; only its generation at the reviewed split is replaced.
The full resulting episode is retained for independent environment replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ttcl.trajectory_hyperlora import alfworld_zero_shot_probe as actor_module
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import (
    digest, file_hash, records_from_episode,
)
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields
from ttcl.trajectory_hyperlora.probe_alf_own_only_first_divergence_v2 import save
from ttcl.trajectory_hyperlora.prepare_alf_source_pair_action_v1 import expected


def forced_episode(agent, tokenizer, game, fields, reference, row, device):
    original_generate = actor_module.generate
    calls = 0

    def substitute(*params, **kwargs):
        nonlocal calls
        turn = calls
        calls += 1
        if turn == row['turn']:
            allowed = params[5] if len(params) > 5 else kwargs.get('available')
            # generate(agent, tokenizer, messages, device, max_tokens, available)
            if row['negative_action'] not in allowed:
                raise ValueError('Forced action excluded by frozen loop guard')
            return row['negative_action']
        return original_generate(*params, **kwargs)

    with patch.object(actor_module, 'generate', substitute):
        episode = run_episode(agent, tokenizer, game, fields,
            adapter=True, device=device, max_steps=50, max_new_tokens=64,
            constrain_actions=True, actor_history_turns=2,
            loop_guard_max=2)
    if (episode['status'] != 'complete' or
            calls != episode['steps'] or
            episode['steps'] <= row['turn'] or
            episode['trajectory'][:row['turn']] !=
                reference['trajectory'][:row['turn']] or
            episode['trajectory'][row['turn']]['command'] !=
                row['negative_action']):
        raise ValueError(f'Forced source-pair episode changed: {episode}')
    return episode


def source_records(args, source):
    path = (args.source_seq0 if source['sequence'] == 0
            else args.source_seq6)
    if file_hash(path) != source['report_sha256']:
        raise ValueError('Changed own-source report')
    report = json.loads(path.read_text())
    row = report['games'][source['source_index']]
    records = records_from_episode(row['episode'])
    if (row['game'] != source['game'] or
            row['episode']['reward'] != 1. or
            digest(records) != source['records_sha256']):
        raise ValueError('Changed winning own trajectory')
    return records


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates = json.loads(args.candidates.read_text())
    candidate_audit = json.loads(args.candidate_audit.read_text())
    review = json.loads(args.review.read_text())
    matrix = json.loads(args.report.read_text())
    if (candidates['rows'] != expected(args) or
            candidate_audit['candidates_sha256'] !=
                file_hash(args.candidates) or
            candidate_audit['summary']['verified_candidates'] !=
                len(candidates['rows']) or
            matrix['review_sha256'] != file_hash(args.review) or
            matrix['checkpoint_sha256'] != file_hash(args.checkpoint) or
            matrix['failures'] or args.max_cases > len(candidates['rows'])):
        raise ValueError('Changed source-pair review or frozen rollout')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen contextual source actor')
    sources = {x['source_id']: x for x in review['sources']}
    arms = {(x['target_id'], x['source_id']): x for x in matrix['pairs']}
    result = {'protocol': 'Official train source-dependent action intervention: 8x18 matrix winner/loser histories; replay winning source exactly, force losing source first-divergence action, continue same winning-source policy with 50-step/64-token/two-turn/two-repeat protocol; local policy-conditional effect only',
        'candidates_sha256': file_hash(args.candidates),
        'candidate_audit_sha256': file_hash(args.candidate_audit),
        'review_sha256': file_hash(args.review),
        'matrix_sha256': file_hash(args.report),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'max_cases': args.max_cases, 'rows': [], 'failures': []}
    save(args.output, result)
    fields_cache = {}
    for row in candidates['rows'][:args.max_cases]:
        try:
            source_id = row['winner_source_id']
            if source_id not in fields_cache:
                records = source_records(args, sources[source_id])
                bounded, _, _ = bounded_source_text(tokenizer, records, 2048)
                with torch.no_grad():
                    fields_cache[source_id] = contextual_text_fields(
                        agent, tokenizer, bounded, args.device, 2048,
                        pooling='both')
            fields = fields_cache[source_id]
            game = args.data_root / row['target_game']
            original = arms[row['target_id'], source_id]['episode']
            with torch.no_grad():
                reference = run_episode(agent, tokenizer, game, fields,
                    adapter=True, device=args.device, max_steps=50,
                    max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
            if (reference['status'] != 'complete' or
                    digest(reference) != digest(original) or
                    reference['reward'] != 1.):
                raise ValueError('Frozen winning-history actor did not reproduce')
            with torch.no_grad():
                changed = forced_episode(agent, tokenizer, game, fields,
                                         original, row, args.device)
            result['rows'].append({'input_content_sha256':
                row['input_content_sha256'],
                'target_id': row['target_id'],
                'winner_source_id': source_id,
                'loser_source_id': row['loser_source_id'],
                'turn': row['turn'],
                'reference_episode_sha256': digest(reference),
                'reference_reward': reference['reward'],
                'loser_history_reward':
                    arms[row['target_id'], row['loser_source_id']]['episode']['reward'],
                'forced_reward': changed['reward'],
                'forced_episode': changed})
            save(args.output, result)
            print(json.dumps({'n': len(result['rows']),
                'target_id': row['target_id'],
                'turn': row['turn'],
                'forced_reward': changed['reward']}), flush=True)
        except Exception as exc:
            result['failures'].append({'input_content_sha256':
                row['input_content_sha256'], 'reason': repr(exc)})
            save(args.output, result)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_audited_20261007.json'))
    parser.add_argument('--source-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--candidates', type=Path, default=Path('data/annotations/alf_source_pair_first_action10_reviewed_20261007.json'))
    parser.add_argument('--candidate-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_first_action10_audited_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_source_pair_action_intervention10_v1_20261007.json'))
    parser.add_argument('--max-cases', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    args = parser.parse_args()
    if not 1 <= args.max_cases <= 10:
        parser.error('Use at most ten reviewed candidate targets')
    run(args)
