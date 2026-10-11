"""Fit bounded LoRA updates on own past actions; measure held-out later utility.

Mechanism/data-generation experiment only. No environment rewards are inferred
from action likelihood, and no H or evaluation-task labels are trained here.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ttcl.trajectory_hyperlora.train_causal_prefix_v70 import checked
from ttcl.trajectory_hyperlora.prepare_causal_prefix_v70 import prefix_records
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import effective_norm
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash


def prepare(args):
    prior = checked(SimpleNamespace(review=args.parent_review))
    parent = json.loads(args.parent.read_text())
    assert prior['parent_sha256'] == file_hash(args.parent)
    audit = json.loads(args.parent.with_suffix('.audit.json').read_text())
    assert audit['passed'] and audit['report_sha256'] == file_hash(args.parent)
    boot = json.loads((args.bootstrap.parent/'training.json').read_text())
    assert boot['complete'] and not boot['failures']
    assert boot['bootstrap_sha256'] == file_hash(args.bootstrap)
    manifest = json.loads((args.bootstrap.parent/'manifest.json').read_text())
    assert manifest['review_sha256'] == file_hash(args.parent_review)
    assert manifest['schedule'][0]['stream_index'] == prior['own_success_indices'][7]
    groups, skipped = [], []
    for index in prior['own_success_indices'][8:]:
        ids = [k for k, row in enumerate(prior['targets']) if row['stream_index'] == index]
        if len(ids) < 2:
            skipped.append({'stream_index': index, 'action_targets': len(ids),
                            'reason': 'Need at least two disjoint action targets for support/query split'})
            continue
        half = len(ids)//2
        support, query = ids[:half], ids[half:]
        cutoff = prior['targets'][support[-1]]['target_step']+1
        assert max(prior['targets'][k]['target_step'] for k in support) < min(prior['targets'][k]['target_step'] for k in query)
        row = parent['rows'][index]
        assert row['index'] == index and row['episode']['reward'] == 1
        assert all(prior['targets'][k]['episode_sha256'] == digest(row['episode']) for k in ids)
        source = prefix_records(row['episode']['initial_observation'], row['episode']['trajectory'][:cutoff])
        content = {'stream_index': index, 'game': row['game'], 'source_end_exclusive': cutoff,
                   'source_records': source, 'support_indices': support, 'query_indices': query,
                   'support_bindings': [prior['targets'][k]['input_content_sha256'] for k in support],
                   'query_bindings': [prior['targets'][k]['input_content_sha256'] for k in query],
                   'availability': 'After own episode completion; source input excludes query actions',
                   'update_steps': 20, 'update_order': [support[k % len(support)] for k in range(20)]}
        groups.append({**content, 'input_content_sha256': digest(content), 'reviewed_target': True})
    assert len(groups)+len(skipped) == 20 and groups
    deps = ['fit_support_updates_v76.py', 'train_causal_prefix_v70.py', 'prepare_causal_prefix_v70.py',
            'alfworld_zero_shot_probe.py', 'train_alf_next_task.py', 'direct_composition_pilot.py',
            'alfworld_failure_retry_residual_v31b.py']
    weights = sorted(args.model.glob('*.safetensors'))
    assert weights
    return {'parent_review_sha256': file_hash(args.parent_review), 'parent_sha256': file_hash(args.parent),
            'bootstrap_sha256': file_hash(args.bootstrap), 'loader_checkpoint_sha256': file_hash(args.checkpoint),
            'bootstrap_training_sha256': file_hash(args.bootstrap.parent/'training.json'),
            'dependencies': {name: file_hash(Path(__file__).with_name(name)) for name in deps},
            'model_weights': {str(p): file_hash(p) for p in weights},
            'model_config_sha256': file_hash(args.model/'config.json'),
            'groups': groups, 'groups_sha256': digest(groups), 'skipped': skipped, 'lr': 1e-4, 'weight_decay': 0.,
            'max_relative': .25, 'seed': 760000, 'utility_tolerance': 1e-6,
            'loss_aggregation': 'Mean of per-action mean token cross-entropies including EOS',
            'budget': {'shared_bootstrap_steps': 300, 'teacher_steps': 20*len(groups), 'query_gradient_steps': 0},
            'scope': 'Own-success training-prefix replay; independent B-only teachers from shared bootstrap, fixed A. Later-query labels screen candidates, never enter teacher optimizer. No online or environment-performance claim.'}


def run(args, review):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    targets = checked(SimpleNamespace(review=args.parent_review))['targets']
    agent, tok = load_agent(args.model, args.checkpoint, args.device, .60)
    agent.requires_grad_(False); agent.eval(); agent.set_source(None)
    bootstrap = torch.load(args.bootstrap, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for layer, a in zip(agent.adapters, bootstrap['A'], strict=True):
            layer.a.copy_(a.to(layer.a))
    shared = [x.to(args.device) for x in bootstrap['B']]
    factors = torch.nn.ParameterList([torch.nn.Parameter(x.clone()) for x in shared])
    for layer, b in zip(agent.adapters, factors, strict=True):
        layer.b = b
    a_hash = factors_hash([x.a for x in agent.adapters])
    assert a_hash == factors_hash(bootstrap['A'])
    reference = effective_norm(agent, shared)
    assert reference > 0
    ids = sorted({k for g in review['groups'] for k in g['support_indices']+g['query_indices']})
    prefixes = {}
    for k in ids:
        text = tok.apply_chat_template(targets[k]['actor_messages'], tokenize=False, add_generation_prompt=True)
        prefixes[k] = tok(text, add_special_tokens=False).input_ids
        assert len(prefixes[k]) <= 3000
    result = {'complete': False, 'review_sha256': file_hash(args.review), 'rows': [], 'failures': [],
              'actor_ids_sha256': digest(prefixes), 'A_sha256': a_hash, 'shared_B_sha256': factors_hash(shared),
              'reference_effective_norm': reference}

    def install(values):
        with torch.no_grad():
            for b, value in zip(factors, values, strict=True):
                b.copy_(value.to(b))

    def losses(indices):
        with torch.no_grad():
            values = [float(target_loss(agent, tok, targets[k], prefixes[k], args.device)) for k in indices]
        assert all(torch.isfinite(torch.tensor(values)))
        return values

    try:
        for group in review['groups']:
            index = group['stream_index']
            torch.manual_seed(review['seed']+index)
            install(shared)
            before_support = losses(group['support_indices'])
            before_query = losses(group['query_indices'])
            optimizer = torch.optim.AdamW(list(factors), lr=review['lr'], weight_decay=0.)
            train_losses = []
            for k in group['update_order']:
                loss = target_loss(agent, tok, targets[k], prefixes[k], args.device)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(factors, 1., error_if_nonfinite=True)
                optimizer.step()
                delta = [b.detach()-s for b, s in zip(factors, shared, strict=True)]
                magnitude = effective_norm(agent, delta)
                scale = min(1., review['max_relative']*reference/magnitude) if magnitude else 1.
                install([s+scale*d for s, d in zip(shared, delta, strict=True)])
                train_losses.append(float(loss.detach()))
            delta = [b.detach().clone()-s for b, s in zip(factors, shared, strict=True)]
            magnitude = effective_norm(agent, delta)
            assert magnitude <= review['max_relative']*reference*(1+1e-5)
            after_support = losses(group['support_indices'])
            after_query = losses(group['query_indices'])
            generator = torch.Generator(device='cpu').manual_seed(review['seed']+index)
            random_delta = [torch.randn(d.shape, generator=generator).to(d) for d in delta]
            random_norm = effective_norm(agent, random_delta)
            random_delta = [d*(magnitude/random_norm) for d in random_delta]
            assert abs(effective_norm(agent, random_delta)-magnitude) <= 1e-5*max(1., magnitude)
            install([s+d for s,d in zip(shared, random_delta, strict=True)])
            random_query = losses(group['query_indices'])
            mean = lambda xs: sum(xs)/len(xs)
            gain = mean(before_query)-mean(after_query)
            over_random = mean(random_query)-mean(after_query)
            artifact = args.output/f'teacher_{index:03d}.pt'
            torch.save({'delta': [d.cpu() for d in delta], 'random_delta': [d.cpu() for d in random_delta],
                        'A_sha256': a_hash, 'bootstrap_sha256': review['bootstrap_sha256'],
                        'group_binding': group['input_content_sha256']}, artifact)
            row = {'stream_index': index, 'group_binding': group['input_content_sha256'],
                   'support_before': before_support, 'support_after': after_support,
                   'query_before': before_query, 'query_after': after_query, 'query_random': random_query,
                   'query_gain': gain, 'query_gain_over_random': over_random,
                   'accepted': gain > review['utility_tolerance'] and over_random > review['utility_tolerance'],
                   'training_steps': len(train_losses), 'train_loss_mean': mean(train_losses),
                   'delta_effective_norm': magnitude, 'artifact_sha256': file_hash(artifact)}
            assert factors_hash([x.a for x in agent.adapters]) == a_hash
            result['rows'].append(row)
            result['summary'] = {'n': len(result['rows']), 'accepted': sum(r['accepted'] for r in result['rows']),
                                 'training_steps': sum(r['training_steps'] for r in result['rows'])}
            save(args.output/'report.json', result)
            print(json.dumps({k: row[k] for k in ('stream_index', 'query_gain', 'query_gain_over_random', 'accepted')}), flush=True)
        result['complete'] = True
        save(args.output/'report.json', result)
    except Exception as exc:
        result['failures'].append(repr(exc)); save(args.output/'report.json', result); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run'))
    p.add_argument('--parent-review', type=Path, default=Path('data/annotations/alfworld_causal_prefix_v70_train120_20261010.json'))
    p.add_argument('--parent', type=Path, default=Path('results/trajectory_hyperlora/alfworld_cold_joint_memory_v63_hybrid_20261010.json'))
    p.add_argument('--bootstrap', type=Path, default=Path('results/trajectory_hyperlora/alfworld_centered_experience_v75b_20261010/bootstrap.pt'))
    p.add_argument('--checkpoint', type=Path, default=Path('results/trajectory_hyperlora/alfworld_centered_experience_v75b_20261010/hyper.pt'))
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--review', type=Path, default=Path('data/annotations/alfworld_support_updates_v76_20261010.json'))
    p.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alfworld_support_updates_v76_20261010'))
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args(); review = prepare(args)
    if args.command == 'prepare':
        if args.review.exists(): raise FileExistsError(args.review)
        save(args.review, review)
        print(json.dumps({'groups': len(review['groups']), 'budget': review['budget']}))
    else:
        assert json.loads(args.review.read_text()) == review
        run(args, review)


if __name__ == '__main__':
    main()
