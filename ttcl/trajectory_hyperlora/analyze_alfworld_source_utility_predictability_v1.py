"""Leave-one-target-out diagnostic of a generic latent source utility selector.

This is training-domain analysis only; it does not alter or re-evaluate the actor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.alfworld_online_persistent_context_lora_v1 import bounded_source_text
from ttcl.trajectory_hyperlora.alfworld_source_utility_matrix_v1 import checked_pairs, SOURCE_INDICES, TARGET_INDICES
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text


def encode_pairs(agent, tokenizer, args, pairs):
    source_vectors, target_vectors = {}, {}
    target_report = json.loads(args.target_report.read_text())
    with torch.no_grad():
        for pair in pairs:
            si, ti = pair['source_index'], pair['target_index']
            if si not in source_vectors:
                source_text, _, _ = bounded_source_text(tokenizer, pair['source_records'], 2048)
                source_vectors[si] = contextual_text_fields(agent, tokenizer, source_text,
                    args.device, 2048, pooling='both')['pair_contextual'].float().cpu()
            if ti not in target_vectors:
                observation = target_report['games'][ti]['base']['initial_observation']
                target_vectors[ti] = contextual_text_fields(agent, tokenizer,
                    task_context_text(observation, observation), args.device, 2048,
                    pooling='both')['pair_contextual'].float().cpu()
        features = []
        for pair in pairs:
            source = source_vectors[pair['source_index']].to(args.device)
            target = target_vectors[pair['target_index']].to(args.device)
            source = F.layer_norm(source, (source.shape[-1],))
            target = F.layer_norm(target, (target.shape[-1],))
            raw = torch.cat((source * target, (source - target).abs()), dim=-1)
            features.append(agent.task_pair_latent(raw).float().cpu().flatten())
    return torch.stack(features).double()


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = checked_pairs(args)
    audited = json.loads(args.audit.read_text())
    if (audited['review_sha256'] != file_hash(args.review) or
            audited['raw_report_sha256'] != file_hash(args.matrix) or
            len(audited['pairs']) != len(pairs)):
        raise ValueError('Changed or unaudited source utility matrix')
    rewards = torch.tensor([r['lora'] for r in audited['pairs']], dtype=torch.float64)
    for pair, row in zip(pairs, audited['pairs'], strict=True):
        if pair['source_index'] != row['source_index'] or pair['target_index'] != row['target_index']:
            raise ValueError('Cross utility order changed')
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device, args.gpu_fraction)
    features = encode_pairs(agent, tokenizer, args, pairs)
    rows = []
    for ti in TARGET_INDICES:
        train = torch.tensor([p['target_index'] != ti for p in pairs])
        test_indices = [i for i, p in enumerate(pairs) if p['target_index'] == ti]
        x_train = features[train]
        center = x_train.mean(0)
        x_train = F.normalize(x_train - center, dim=-1)
        x_test = F.normalize(features[test_indices] - center, dim=-1)
        y_train = rewards[train]
        mean = y_train.mean()
        kernel = x_train @ x_train.T
        weights = torch.linalg.solve(kernel + torch.eye(len(kernel), dtype=kernel.dtype), y_train - mean)
        prediction = mean + (x_test @ x_train.T) @ weights
        selected_pos = int(torch.argmax(prediction).item())
        selected = test_indices[selected_pos]
        train_source_totals = {si: sum(float(rewards[i]) for i, p in enumerate(pairs)
            if train[i] and p['source_index'] == si) for si in SOURCE_INDICES}
        static_source = max(SOURCE_INDICES, key=lambda si: train_source_totals[si])
        static = next(i for i in test_indices if pairs[i]['source_index'] == static_source)
        rows.append({'target_index': ti, 'selected_source': pairs[selected]['source_index'],
            'selected_reward': float(rewards[selected]), 'static_source': static_source,
            'static_reward': float(rewards[static]), 'base_reward': pairs[selected]['target_base_reward'],
            'oracle_reward': max(float(rewards[i]) for i in test_indices),
            'mean_candidate_reward': sum(float(rewards[i]) for i in test_indices) / len(test_indices),
            'scores': {str(pairs[i]['source_index']): float(prediction[pos])
                for pos, i in enumerate(test_indices)}})
    result = {'protocol': 'Training-domain leave-one-target-out kernel ridge on frozen learned source-target pair latent; fixed unit kernel regularization; no target labels in fit; diagnostic, not an online policy test',
        'audit_sha256': file_hash(args.audit), 'checkpoint_sha256': file_hash(args.checkpoint),
        'summary': {'targets': len(rows), 'selected': sum(r['selected_reward'] for r in rows),
            'static_train_only': sum(r['static_reward'] for r in rows),
            'base': sum(r['base_reward'] for r in rows),
            'oracle': sum(r['oracle_reward'] for r in rows),
            'random_expected': sum(r['mean_candidate_reward'] for r in rows)}, 'targets': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_simple40_onpolicy600_20261006.pt'))
    parser.add_argument('--source-report', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_vector_reward_gate_train_seq0_6_20261007.json'))
    parser.add_argument('--target-report', type=Path, default=Path('results/trajectory_hyperlora/alf_online_context_persistent_train_seq6_6_20261007.json'))
    parser.add_argument('--review', type=Path, default=Path('data/annotations/alf_own_success_source_utility_matrix48_reviewed_20261007.json'))
    parser.add_argument('--matrix', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_source_utility_matrix48_20261007.json'))
    parser.add_argument('--audit', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_source_utility_matrix48_audited_20261007.json'))
    parser.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alf_own_success_source_utility_predictability_20261007.json'))
    parser.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    parser.add_argument('--context-tokens', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-fraction', type=float, default=.65)
    args = parser.parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()
