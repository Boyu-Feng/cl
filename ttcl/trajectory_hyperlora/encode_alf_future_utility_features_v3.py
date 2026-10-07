"""Encode new targets while reusing exactly bound frozen source features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v3 import checked
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text


def encode(args):
    if args.output_features.exists():
        raise FileExistsError(args.output_features)
    sources, targets, _ = checked(args)
    report = json.loads(args.output.read_text())
    first_review = json.loads(args.first_review.read_text())
    old = torch.load(args.original_features, map_location='cpu',
                     weights_only=True)
    if (report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(report['base']) != 18 or
            old['review_sha256'] != file_hash(args.first_review) or
            old['checkpoint_sha256'] != file_hash(args.checkpoint) or
            old['target_policy'] != 'train18' or
            len(first_review['sources']) != len(sources) or
            any({k: v for k, v in source.items() if k != 'records'} !=
                first_review['sources'][i]
                for i, source in enumerate(sources))):
        raise ValueError('Changed frozen source features or new target base')
    observations = []
    for target, row in zip(targets, report['base'], strict=True):
        if (row['target_id'] != target['target_id'] or
                row['game'] != target['game'] or
                row['game_sha256'] != target['game_sha256'] or
                row['episode']['status'] != 'complete'):
            raise ValueError('Changed reviewed new target observation')
        observations.append(row['episode']['initial_observation'])
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen contextual encoder')
    targets_encoded = []
    with torch.no_grad():
        for target, observation in zip(targets, observations, strict=True):
            vector = contextual_text_fields(agent, tokenizer,
                task_context_text(observation, observation),
                args.device, 2048,
                pooling='both')['pair_contextual'].float().flatten().cpu()
            targets_encoded.append(vector)
            print(json.dumps({'encoded_target': target['target_id']}),
                  flush=True)
    result = {'protocol': 'Frozen old hypernetwork Qwen target-context features for new reviewed 8x18 train games; original eight source global/event/delta encodings reused only after exact source-content and checkpoint binding',
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'original_features_sha256': file_hash(args.original_features),
        'base_observations_sha256': digest(observations),
        'target_policy': 'additional_train18_v3',
        'global': old['global'],
        'events': old['events'],
        'deltas': old['deltas'],
        'event_counts': old['event_counts'],
        'targets': torch.stack(targets_encoded)}
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output_features)
    print(json.dumps({'sources': len(sources), 'targets': len(targets),
        'features_sha256': file_hash(args.output_features)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--original-features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--output-features', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_additional8x18_features_v3_20261007.pt'))
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    encode(parser.parse_args())
