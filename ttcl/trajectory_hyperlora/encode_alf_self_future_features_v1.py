"""Freeze model embeddings of reviewed self trajectories and future states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import (
    bounded_source_text,
)
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import (
    contextual_text_fields, task_context_text,
)
from ttcl.trajectory_hyperlora.encode_alf_future_utility_features_v1 import (
    event_indices,
)


def encode(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text())
    bank_audit = json.loads(args.bank_audit.read_text())
    plan = json.loads(args.pairs.read_text())
    pair_audit = json.loads(args.pair_audit.read_text())
    ancestry = json.loads(args.ancestry_audit.read_text())
    if (bank_audit['review_sha256'] != file_hash(args.bank) or
            pair_audit['pairs_review_sha256'] != file_hash(args.pairs) or
            ancestry['pairs_review_sha256'] != file_hash(args.pairs) or
            ancestry['summary']['independent_of_complete_checkpoint'] or
            len(bank['rows']) != 57 or bank['failures'] or
            plan['bank_review_sha256'] != file_hash(args.bank)):
        raise ValueError('Changed or misrepresented train-domain pair lineage')
    bank_by_game = {row['game']: row for row in bank['rows']}
    targets = {}
    for pair in plan['pairs']:
        row = bank_by_game[pair['target_game']]
        action = row['reviewed_targets'][pair['target_turn']]
        key = action['input_content_sha256']
        if pair['target_action_input_sha256'] != key:
            raise ValueError('Changed target action content')
        targets[key] = (row, action)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen contextual feature encoder')
    global_vectors, delta_vectors, event_counts = [], [], []
    target_vectors = {}
    with torch.no_grad():
        for index, row in enumerate(bank['rows']):
            bounded, _, _ = bounded_source_text(tokenizer,
                row['records'], 2048)
            global_vectors.append(contextual_text_fields(agent, tokenizer,
                bounded, args.device, 2048, pooling='both')
                ['pair_contextual'].float().flatten().cpu())
            deltas = []
            for event_index in event_indices(len(row['records'])):
                event = row['records'][event_index]
                before = (f"Action: {event['action']}\n"
                          f"Before: {event['observation']}")
                after = (f"Action: {event['action']}\n"
                         f"Environment reply: {event['feedback']}")
                a = contextual_text_fields(agent, tokenizer, before,
                    args.device, 2048, pooling='both')
                b = contextual_text_fields(agent, tokenizer, after,
                    args.device, 2048, pooling='both')
                deltas.append((b['pair_contextual'] - a['pair_contextual'])
                              .float().flatten().cpu())
            event_counts.append(len(deltas))
            delta_vectors.append(F.pad(torch.stack(deltas),
                (0, 0, 0, 16-len(deltas))))
            print(json.dumps({'encoded_source': index+1,
                              'of': len(bank['rows'])}), flush=True)
        for index, (key, (row, action)) in enumerate(sorted(targets.items())):
            initial = row['reviewed_targets'][0]['observation']
            query = task_context_text(initial, action['observation'])
            target_vectors[key] = (contextual_text_fields(agent, tokenizer,
                query, args.device, 2048, pooling='both')
                ['pair_contextual'].float().flatten().cpu())
            print(json.dumps({'encoded_target_state': index+1,
                              'of': len(targets)}), flush=True)
    value = {'protocol': 'Frozen existing Qwen contextual embeddings of replay-reviewed official train self trajectories and different-game action states; includes ancestry-exposed internal dev, not an independent benchmark',
        'bank_review_sha256': file_hash(args.bank),
        'bank_audit_sha256': file_hash(args.bank_audit),
        'pairs_review_sha256': file_hash(args.pairs),
        'pair_audit_sha256': file_hash(args.pair_audit),
        'ancestry_audit_sha256': file_hash(args.ancestry_audit),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_games': [row['game'] for row in bank['rows']],
        'source_episode_sha256': [row['episode_sha256']
                                  for row in bank['rows']],
        'global': torch.stack(global_vectors),
        'deltas': torch.stack(delta_vectors),
        'event_counts': torch.tensor(event_counts),
        'targets': target_vectors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.output)
    print(json.dumps({'sources': len(bank['rows']),
                      'target_states': len(targets),
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--bank', type=Path, default=Path(
        'data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--bank-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_base57_replay_reviewed_audited_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path(
        'data/annotations/alf_self_future_pairs_v2_checkpoint_disjoint_reviewed_20261007.json'))
    parser.add_argument('--pair-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_pairs_v2_checkpoint_disjoint_audited_20261007.json'))
    parser.add_argument('--ancestry-audit', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_pairs_v2_ancestry_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_self_future_base57_features_20261007.pt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    encode(parser.parse_args())
