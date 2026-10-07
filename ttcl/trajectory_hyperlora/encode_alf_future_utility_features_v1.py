"""Encode task and ordered action-feedback changes without task-specific slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_dataset_v2 import checked
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text


def event_indices(length, maximum=16):
    if length <= maximum:
        return list(range(length))
    return sorted({round(i * (length - 1) / (maximum - 1))
                   for i in range(maximum)})


def encode(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, targets, _ = checked(args)
    report = json.loads(args.matrix.read_text())
    if (report['review_sha256'] != file_hash(args.review) or
            report['checkpoint_sha256'] != file_hash(args.checkpoint) or
            report['failures'] or len(report['base']) != len(targets)):
        raise ValueError('Need completed, content-bound base target observations')
    observations = []
    for target, row in zip(targets, report['base'], strict=True):
        if (row['target_id'] != target['target_id'] or
                row['game'] != target['game'] or
                row['episode']['status'] != 'complete'):
            raise ValueError('Changed target base observation')
        observations.append(row['episode']['initial_observation'])
    agent, tokenizer = load_agent(args.model, args.checkpoint,
                                  args.device, args.gpu_fraction)
    if (agent.encoder_kind != 'contextual' or
            agent.task_pair_pooling != 'mean'):
        raise ValueError('Expected frozen contextual source encoder')
    global_vectors, event_vectors, delta_vectors, event_counts = [], [], [], []
    target_vectors = []
    with torch.no_grad():
        for source in sources:
            bounded, _, _ = bounded_source_text(tokenizer, source['records'], 2048)
            global_vector = contextual_text_fields(agent, tokenizer, bounded,
                args.device, 2048, pooling='both')['pair_contextual'].float().flatten().cpu()
            global_vectors.append(global_vector)
            events, deltas = [], []
            records = source['records']
            for index in event_indices(len(records)):
                record = records[index]
                before = f"Action: {record['action']}\nBefore: {record['observation']}"
                after = f"Action: {record['action']}\nEnvironment reply: {record['feedback']}"
                before_vec = contextual_text_fields(agent, tokenizer, before,
                    args.device, 2048, pooling='both')['pair_contextual'].float().flatten().cpu()
                after_vec = contextual_text_fields(agent, tokenizer, after,
                    args.device, 2048, pooling='both')['pair_contextual'].float().flatten().cpu()
                events.append(after_vec)
                deltas.append(after_vec - before_vec)
            if not events:
                raise ValueError('Own success has no interaction events')
            event_counts.append(len(events))
            width = global_vector.numel()
            event_vectors.append(F.pad(torch.stack(events),
                (0, 0, 0, 16 - len(events))))
            delta_vectors.append(F.pad(torch.stack(deltas),
                (0, 0, 0, 16 - len(deltas))))
            print(json.dumps({'encoded_source': source['source_id'],
                'events': len(events), 'width': width}), flush=True)
        for target, observation in zip(targets, observations, strict=True):
            context = task_context_text(observation, observation)
            vector = contextual_text_fields(agent, tokenizer, context,
                args.device, 2048, pooling='both')['pair_contextual'].float().flatten().cpu()
            target_vectors.append(vector)
            print(json.dumps({'encoded_target': target['target_id']}), flush=True)
    value = {'protocol': 'Frozen Qwen mean-pooled full source, ordered action-feedback event and before-after delta, and initial target context; no ALFWorld family slots; at most 16 uniformly sampled source events',
        'review_sha256': file_hash(args.review),
        'checkpoint_sha256': file_hash(args.checkpoint),
        'base_observations_sha256': digest(observations),
        'target_policy': args.target_policy,
        'global': torch.stack(global_vectors),
        'events': torch.stack(event_vectors),
        'deltas': torch.stack(delta_vectors),
        'event_counts': torch.tensor(event_counts),
        'targets': torch.stack(target_vectors)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.output)
    print(json.dumps({'sources': len(sources), 'targets': len(targets),
                      'output': str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report-seq0', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--source-report-seq6', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq6_6_20261007.json'))
    parser.add_argument('--plan', type=Path, default=Path('ttcl/results/experience_evolution/alfworld_delta_20260922/plan.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--target-policy', choices=('train18', 'independent12'), default='train18')
    parser.add_argument('--checkpoint-training-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train600_candidates_20261006.json'))
    parser.add_argument('--warmstart-candidates', type=Path, default=Path('results/trajectory_hyperlora/alf_next_task_candidates_20261005.json'))
    parser.add_argument('--prior-train-review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_future_reward_8x18_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_future_reward_8x18_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_future_reward_train8x18_features_20261007.pt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    encode(args)

if __name__ == '__main__':
    main()
