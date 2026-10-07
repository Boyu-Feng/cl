"""Encode reviewed own-only-success action states; reuse frozen base57 source features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text


def encode(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source_bank.read_text())
    target = json.loads(args.target_bank.read_text())
    plan = json.loads(args.pairs.read_text())
    audit = json.loads(args.pair_audit.read_text())
    frozen = torch.load(args.source_features, map_location='cpu', weights_only=True)
    if (frozen['bank_review_sha256'] != file_hash(args.source_bank)
        or frozen['checkpoint_sha256'] != file_hash(args.checkpoint)
        or plan['source_bank_sha256'] != file_hash(args.source_bank)
        or plan['target_bank_sha256'] != file_hash(args.target_bank)
        or audit['pairs_review_sha256'] != file_hash(args.pairs)
        or audit['summary']['different_game_action_pairs'] != 234
        or audit['summary']['target_arm'] != 'own'
        or len(source['rows']) != 57 or len(target['rows']) != 42):
        raise ValueError('Changed or unreviewed self-trajectory inputs')
    targets = {row['game']: row for row in target['rows']}
    requested = {}
    for pair in plan['pairs']:
        row = targets[pair['target_game']]
        action = row['reviewed_targets'][pair['target_turn']]
        key = action['input_content_sha256']
        if pair['target_action_input_sha256'] != key:
            raise ValueError('Changed target action binding')
        requested[key] = (row, action)
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    encoded = {}
    with torch.no_grad():
        for index, (key, (row, action)) in enumerate(sorted(requested.items())):
            initial = row['reviewed_targets'][0]['observation']
            query = task_context_text(initial, action['observation'])
            encoded[key] = contextual_text_fields(agent, tokenizer, query,
                args.device, 2048, pooling='both')['pair_contextual'].float().flatten().cpu()
            print(json.dumps({'encoded_target': index + 1,
                              'of': len(requested)}), flush=True)
    value = {'protocol': 'Frozen Qwen contextual states for replay-verified own-only-success official train actions; sources from fixed base57 features; train-only ancestor-exposed',
        'source_bank_sha256': file_hash(args.source_bank),
        'target_bank_sha256': file_hash(args.target_bank),
        'pairs_review_sha256': file_hash(args.pairs),
        'pair_audit_sha256': file_hash(args.pair_audit),
        'source_features_sha256': file_hash(args.source_features),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_games': frozen['source_games'],
        'global': frozen['global'], 'deltas': frozen['deltas'],
        'event_counts': frozen['event_counts'], 'targets': encoded}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.output)
    print(json.dumps({'sources': len(value['source_games']),
                      'target_states': len(encoded)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-bank', type=Path, default=Path('data/annotations/alf_self_base57_replay_reviewed_20261007.json'))
    parser.add_argument('--target-bank', type=Path, default=Path('data/annotations/alf_own_only42_replay_reviewed_20261007.json'))
    parser.add_argument('--pairs', type=Path, default=Path('data/annotations/alf_own_only_future_pairs_reviewed_20261007.json'))
    parser.add_argument('--pair-audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_future_pairs_audited_20261007.json'))
    parser.add_argument('--source-features', type=Path, default=Path('results/trajectory_hyperlora/alf_self_future_base57_features_20261007.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_only_future_features_v2_20261007.pt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    encode(parser.parse_args())
