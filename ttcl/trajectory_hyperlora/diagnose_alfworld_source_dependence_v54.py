"""Read-only source sensitivity diagnostic; no environment score or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import alignment_pair
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import bounded_source_fields
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.contextual_alf_source import contextual_text_fields, task_context_text
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save


def effective_inner(left, right, a):
    gram = a @ a.T
    return torch.sum((left.T @ right) * gram)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alfworld_bootstrap_own_success_v43b_hyper_20261010.pt'))
    p.add_argument('--source-report', type=Path, default=Path('results/trajectory_hyperlora/alfworld_online_own_trajectory_v52_stream240_20261010.json'))
    p.add_argument('--families', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    p.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alfworld_source_dependence_v54_20261010.json'))
    p.add_argument('--device', default='cuda:3')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.source_report.read_text())
    family = {r['game']:r['family'] for r in json.loads(args.families.read_text())['games']}
    wins = [r for r in report['rows'] if r['online']['reward'] == 1]
    agent, tokenizer = load_agent(args.model, args.checkpoint, args.device, .7)
    result = {'protocol': 'Read-only generated effective BA sensitivity on exposed train trajectories; targets differ from source tasks; not causal performance or heldout evidence',
        'checkpoint_sha256': file_hash(args.checkpoint), 'source_report_sha256': file_hash(args.source_report),
        'families_sha256': file_hash(args.families), 'runner_sha256': file_hash(Path(__file__)), 'rows': []}
    for row in wins:
        eligible = [r for r in wins if r['game'] != row['game'] and family[r['game']] == family[row['game']]]
        if not eligible:
            continue
        target = min(eligible, key=lambda x: x['index'])
        coherent, shifted = alignment_pair(records_from_episode(row['online']), tokenizer)
        fields = [bounded_source_fields(agent, tokenizer, x, args.device) for x in (coherent, shifted)]
        initial = target['online']['initial_observation']
        query = contextual_text_fields(agent, tokenizer, task_context_text(initial, initial), args.device, 2048, pooling='both')
        factors = []
        with torch.no_grad():
            for f in fields:
                agent.set_source(f, target_fields=query)
                factors.append([layer.b[0].float().clone() for layer in agent.adapters])
            measures = []
            for layer, head, normal, corrupt in zip(agent.adapters, agent.b_heads, *factors, strict=True):
                a = layer.a.detach().float()
                norm2 = effective_inner(normal, normal, a)
                corrupt2 = effective_inner(corrupt, corrupt, a)
                delta2 = effective_inner(normal-corrupt, normal-corrupt, a)
                bias = head.bias.reshape_as(normal).detach().float()
                measures.append({'effective_relative_change':float(torch.sqrt(delta2.clamp_min(0)/norm2.clamp_min(1e-30))),
                    'effective_cosine':float(effective_inner(normal, corrupt, a)/(norm2*corrupt2).clamp_min(1e-30).sqrt()),
                    'head_bias_to_output_norm_ratio':float(torch.sqrt(effective_inner(bias,bias,a).clamp_min(0)/norm2.clamp_min(1e-30)))})
        agent.set_source(None)
        result['rows'].append({'source_index':row['index'], 'target_index':target['index'],
            'source_game':row['game'], 'target_game':target['game'],
            'coherent_sha256':digest(coherent),'shifted_sha256':digest(shifted),'layers':measures})
    values = [x for row in result['rows'] for x in row['layers']]
    result['summary'] = {'source_target_pairs':len(result['rows']),
        **{key:sum(x[key] for x in values)/len(values) for key in values[0]}}
    save(args.output, result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
