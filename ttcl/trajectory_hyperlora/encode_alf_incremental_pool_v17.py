"""Freeze trajectory and task embeddings for a generic update-utility gate.

Only runs after all three v14 shards have passed original-environment audits.
The same frozen actor encoder supplies both source trajectory feedback and
current-task features; family/action labels are not used as model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.evaluate_alf_fresh_seen_online_v6 import target_feature


def encode(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    review = json.loads(args.pool_review.read_text())
    source = torch.load(args.source_features, map_location='cpu', weights_only=True)
    if (source['review_sha256'] != review['parent_review_sha256'] or
            source['checkpoint_sha256'] != file_hash(args.checkpoint) or
            review['checkpoint_sha256'] != file_hash(args.checkpoint) or
            tuple(source['global'].shape) != (8, 2560) or
            tuple(source['deltas'].shape) != (8, 16, 2560)):
        raise ValueError('Changed frozen own-trajectory representation')
    source_delta = torch.stack([source['deltas'][i, :int(n)].mean(0)
        for i, n in enumerate(source['event_counts'])])
    observations = {}
    report_hashes = []
    audit_hashes = []
    for shard in range(3):
        report_path = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_20261008.json'
        audit_path = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_audited_20261008.json'
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (report['pool_review_sha256'] != file_hash(args.pool_review) or
                audit['pool_review_sha256'] != file_hash(args.pool_review) or
                audit['raw_report_sha256'] != file_hash(report_path) or
                audit['failed_replays'] != 0 or
                report['failures'] or len(report['rows']) != 156):
            raise ValueError('Unfinished or changed audited future-reward shard')
        report_hashes.append(file_hash(report_path))
        audit_hashes.append(file_hash(audit_path))
        for row in report['rows']:
            target = review['targets'][row['target_id']]
            if (row['game'] != target['game'] or
                    row['split'] != target['split'] or
                    row['freeze']['initial_observation'] !=
                    row['update']['initial_observation']):
                raise ValueError('Changed target task context')
            initial = row['freeze']['initial_observation']
            if target['target_id'] in observations and \
                    observations[target['target_id']] != initial:
                raise ValueError('Same target reset produced different text')
            observations[target['target_id']] = initial
    if set(observations) != set(range(78)):
        raise ValueError('Incomplete target representation set')
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_context_scope != 'current' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Changed frozen actor feature encoder')
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    target_vectors = []
    for index in range(78):
        target_vectors.append(target_feature(agent, tokenizer,
                                             observations[index], args.device).cpu())
        print(json.dumps({'encoded_target': index + 1}), flush=True)
    value = {'protocol': 'Frozen Qwen source trajectory feedback and official task initial observation features for train-only update gate; no family/action slot',
        'pool_review_sha256': file_hash(args.pool_review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_features_sha256': file_hash(args.source_features),
        'report_sha256': report_hashes,
        'audit_sha256': audit_hashes,
        'initial_observation_sha256': [hashlib.sha256(
            observations[i].encode()).hexdigest() for i in range(78)],
        'source_global': source['global'].float(),
        'source_delta': source_delta.float(),
        'target': torch.stack(target_vectors).float()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.output)
    print(json.dumps({'targets': len(target_vectors),
        'features_sha256': file_hash(args.output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(
        'current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--source-features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--results-root', type=Path,
                        default=Path('results/trajectory_hyperlora'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_pool78_v17_features_20261008.pt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.7)
    encode(parser.parse_args())
