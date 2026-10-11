"""Fresh own-trajectory H learning plus guarded additive persistent LoRA B.

Development stream; every reward is measured before that episode can write.
No expert sources, old hypernetwork checkpoints, or same-task retries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

import torch

from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import (
    action_fields, approved_success, bounded_source_fields, fit_examples,
    trainable_digest,
)
from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import (
    examples, new_agent, ordered,
)
from ttcl.trajectory_hyperlora.evaluate_alfworld_bootstrap_own_success_v43 import source_for
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save, bounded_readout
from ttcl.trajectory_hyperlora.alfworld_failure_retry_residual_v31b import bounded_residual, effective_norm
from ttcl.trajectory_hyperlora.evaluate_alfworld_failure_retry_v31b import source_factors
from ttcl.trajectory_hyperlora.persistent_residual_guard_v62 import guarded_add
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.train_alf_next_task import target_loss


@contextmanager
def read_memory(agent, state):
    """Same per-read cap for rollout, success training, and retention scoring."""
    original = agent.set_source

    def patched(source, *args, **kwargs):
        result = original(source, *args, **kwargs)
        if source is not None and state is not None:
            applied, _ = bounded_readout(agent, state, [x.b for x in agent.adapters])
            for layer, value in zip(agent.adapters, applied, strict=True):
                layer.b = layer.b + value.to(layer.b.device, layer.b.dtype)
        return result

    agent.set_source = patched
    try:
        yield
    finally:
        agent.set_source = original
        original(None)


def anchor_items(pool):
    """At most 12 reviewed past actions from four latest eligible own wins."""
    items = examples(pool)
    eligible = sorted({i for _, i, _ in items})[-4:]
    chosen = []
    for i in eligible:
        actions = [x for x in items if x[1] == i]
        chosen.extend(actions[k] for k in sorted({0, len(actions)//2, len(actions)-1}))
    if not chosen:
        raise ValueError('No cross-task reviewed success anchors')
    return chosen


def scorer(agent, tokenizer, pool, device):
    items = anchor_items(pool)
    cached = []
    bindings = []
    for j, i, action in items:
        fields = bounded_source_fields(agent, tokenizer, pool[j]['source_records'], device)
        prefix, query = action_fields(agent, tokenizer, action,
            pool[i]['actions'][0]['current_observation'], device)
        cached.append((fields, query, prefix, action))
        bindings.append({'source_stream_index': pool[j]['stream_index'],
            'target_stream_index': pool[i]['stream_index'],
            'source_episode_sha256': pool[j]['episode_sha256'],
            'target_annotation_sha256': pool[i]['annotation_input_content_sha256'],
            'action_sha256': digest(action)})

    def score(state):
        result = []
        with torch.no_grad(), read_memory(agent, state):
            for fields, query, prefix, action in cached:
                agent.set_source(fields, target_fields=query)
                result.append(float(target_loss(agent, tokenizer, action, prefix, device)))
                agent.set_source(None)
        return result
    return score, bindings


def bindings(args):
    rows = ordered(args)[:args.limit]
    files = ['alfworld_cold_joint_memory_v63.py', 'audit_alfworld_cold_joint_memory_v63.py',
        'persistent_residual_guard_v62.py',
        'alfworld_online_own_trajectory_v52b.py', 'alfworld_bootstrap_own_success_v43.py',
        'alfworld_zero_shot_probe.py', 'contextual_alf_source.py', 'direct_composition_pilot.py',
        'train_alf_next_task.py', 'alfworld_failure_retry_residual_v31b.py',
        'alfworld_persistent_failure_delta_v36.py', 'evaluate_alfworld_failure_retry_v31b.py',
        'evaluate_alfworld_bootstrap_own_success_v43.py', 'audit_my_online_update_v28_reward.py']
    fixed = {'method': 'v63 cold own H plus guarded persistent additive residual',
        'mode': args.mode, 'seed': args.seed, 'limit': args.limit,
        'source_report_sha256': file_hash(args.source_report),
        'model_config_sha256': file_hash(args.model/'config.json'),
        'code_sha256': {f: file_hash(Path(__file__).with_name(f)) for f in files},
        'bootstrap_wins': 8, 'bootstrap_steps': 300, 'online_steps_per_success': 20,
        'bootstrap_lr': 1e-4, 'online_lr': 1e-5, 'rank': 4, 'layers': 2,
        'max_steps': 30, 'max_new_tokens': 64, 'history_turns': 2, 'loop_guard_max': 2,
        'family_source_min_count': 2, 'A_frozen_after_bootstrap': True,
        'write_reference': 'first cross-task bootstrap pair after bootstrap, frozen',
        'write_cap': .25, 'read_cap': .25, 'anchor_loss_tolerance': .01,
        'write_steps': [1., .5, .25, .125], 'anchors': 'latest4 eligible own wins, first/middle/last action',
        'failure_direction': 'own prior success source B minus own failed trajectory B',
        'failure_source_truncation': 'first record plus fitting tail, 2048 tokens',
        'initial_state': 'pretrained frozen base; fresh random H; empty pool; zero R'}
    targets = []
    for i, row in enumerate(rows):
        content = {'index': i, 'game': row['game'], 'family': row['family'],
            'game_sha256': file_hash(args.data_root/row['game']),
            'initial_observation_sha256': digest(row['arms']['base']['initial_observation']),
            'protocol_sha256': digest(fixed)}
        targets.append({**content, 'input_content_sha256': digest(content), 'reviewed_target': True})
    return rows, fixed, targets


def prepare(args):
    if args.review.exists():
        raise FileExistsError(args.review)
    _, fixed, targets = bindings(args)
    save(args.review, {**fixed, 'targets': targets, 'targets_sha256': digest(targets)})
    print(json.dumps({'review': str(args.review), 'targets': len(targets)}), flush=True)


def checked(args):
    rows, fixed, targets = bindings(args)
    review = json.loads(args.review.read_text())
    if review != {**fixed, 'targets': targets, 'targets_sha256': digest(targets)}:
        raise ValueError('Changed v63 protocol or targets')
    return rows, fixed, targets


def snapshot(agent, state, reference, path):
    torch.save({'trainable_state': {n: p.detach().cpu().clone()
        for n, p in agent.named_parameters() if p.requires_grad},
        'A': [x.a.detach().cpu().clone() for x in agent.adapters],
        'R': state, 'reference_norm': reference}, path)
    return {'path': str(path), 'sha256': file_hash(path)}


def evaluate(args):
    if any(p.exists() for p in (args.output, args.annotations, args.states)):
        raise FileExistsError('Use fresh v63 outputs')
    rows, fixed, targets = checked(args)
    args.states.mkdir(parents=True)
    save(args.annotations, {'protocol': 'v63 new own-success action targets, replay reviewed',
        'eval_review_sha256': file_hash(args.review), 'annotations': []})
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    agent, tokenizer = new_agent(args)
    agent.eval()
    rng = random.Random(args.seed)
    pool, state, reference, optimizer, frozen_A = [], None, None, None, None
    report = {**fixed, 'review_sha256': file_hash(args.review), 'rows': [], 'failures': [],
        'bootstrap': None, 'initial_parameters_sha256': trainable_digest(agent)}
    for i, (old, target) in enumerate(zip(rows, targets, strict=True)):
        before_h, before_r = trainable_digest(agent), factors_hash(state)
        count = sum(x['family'] == target['family'] for x in pool)
        j = source_for(target, pool) if count else None
        active = optimizer is not None and count >= 2
        fields = bounded_source_fields(agent, tokenizer, pool[j]['source_records'], args.device) if active else {}
        entry = {'index': i, 'game': target['game'], 'target_input_content_sha256': target['input_content_sha256'],
            'memory_before': len(pool), 'adapter_active': active,
            'source_stream_index': pool[j]['stream_index'] if j is not None else None,
            'source_episode_sha256': pool[j]['episode_sha256'] if j is not None else None,
            'H_before_sha256': before_h, 'R_before_sha256': before_r,
            'update': {'steps': 0}, 'write': None}
        try:
            with read_memory(agent, state):
                ep = run_episode(agent, tokenizer, args.data_root/target['game'], fields,
                    adapter=active, device=args.device, max_steps=30, max_new_tokens=64,
                    constrain_actions=True, actor_history_turns=2, loop_guard_max=2)
            entry['episode'] = ep
            if ep['status'] != 'complete' or digest(ep['initial_observation']) != target['initial_observation_sha256']:
                raise ValueError('Incomplete or changed v63 episode')
            replay(args.data_root/target['game'], ep)
            if ep['reward'] == 1:
                item = approved_success(args, target, ep, i, args.annotations)
                entry['annotation_input_content_sha256'] = item['annotation_input_content_sha256']
                if optimizer is None:
                    pool.append(item)
                    if len(pool) == 8:
                        items = examples(pool)
                        if not items:
                            raise ValueError('Empty bootstrap cross-task pairs')
                        optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=1e-4, weight_decay=.01)
                        entry['update'] = fit_examples(agent, tokenizer, optimizer, pool, items, 300, rng, args.device)
                        for layer in agent.adapters:
                            layer.a.requires_grad_(False)
                        frozen_A = factors_hash([x.a for x in agent.adapters])
                        optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=1e-5, weight_decay=.01)
                        sj, ti, _ = items[0]
                        sf = bounded_source_fields(agent, tokenizer, pool[sj]['source_records'], args.device)
                        good = source_factors(agent, tokenizer, sf, pool[ti]['actions'][0]['current_observation'], args.device)
                        reference = effective_norm(agent, good)
                        if not reference > 0:
                            raise ValueError('Zero bootstrap LoRA reference')
                        report['bootstrap'] = {'index': i, 'own_wins': 8, 'action_targets': len(items),
                            'frozen_A_sha256': frozen_A, 'reference_norm': reference,
                            'calibration_source_stream_index': pool[sj]['stream_index'],
                            'calibration_target_stream_index': pool[ti]['stream_index']}
                else:
                    if j is not None and args.mode in ('hybrid', 'hyper'):
                        with read_memory(agent, state):
                            entry['update'] = fit_examples(agent, tokenizer, optimizer, pool+[item],
                                [(j, len(pool), action) for action in item['actions']], 20, rng, args.device)
                        entry['update']['R_used_sha256'] = before_r
                    pool.append(item)
            elif active and args.mode in ('hybrid', 'residual'):
                # Same current H and same target query for both sides of the delta.
                good = source_factors(agent, tokenizer, fields, ep['initial_observation'], args.device)
                failed_records = records_from_episode(ep)
                failed_fields = bounded_source_fields(agent, tokenizer, failed_records, args.device)
                bad = source_factors(agent, tokenizer, failed_fields, ep['initial_observation'], args.device)
                toward, norms = bounded_residual(agent, good, bad)
                delta = [-x for x in toward]
                score, anchors = scorer(agent, tokenizer, pool, args.device)
                state, trace = guarded_add(state, delta, norm=lambda x: effective_norm(agent, x),
                    reference_norm=reference, score=score)
                delta_path = args.states/f'delta_{i:03d}.pt'
                torch.save(delta, delta_path)
                content = {'stream_index': i, 'game_sha256': target['game_sha256'],
                    'episode_sha256': digest(ep), 'failed_records_sha256': digest(failed_records),
                    'own_success_source_episode_sha256': pool[j]['episode_sha256'],
                    'H_sha256': before_h, 'anchors': anchors, 'delta_sha256': factors_hash(delta)}
                entry['write'] = {**content, 'input_content_sha256': digest(content),
                    'reviewed_target': True, 'review_note': 'Episode replay verified; residual direction is a hypothesis, not a correct-action label',
                    'delta_file': str(delta_path), 'delta_file_sha256': file_hash(delta_path),
                    'norms': norms, 'guard': trace}
            if frozen_A is not None and factors_hash([x.a for x in agent.adapters]) != frozen_A:
                raise ValueError('Persistent LoRA coordinate basis changed')
            entry.update(H_after_sha256=trainable_digest(agent), R_after_sha256=factors_hash(state))
            # Persist every state-changing step for an independent state-chain audit.
            if before_h != entry['H_after_sha256'] or before_r != entry['R_after_sha256']:
                entry['snapshot'] = snapshot(agent, state, reference, args.states/f'state_{i:03d}.pt')
            report['rows'].append(entry)
            report['summary'] = {'n': len(report['rows']),
                'wins': sum(x['episode']['reward'] for x in report['rows']), 'own_successes': len(pool),
                'training_steps': sum(x['update']['steps'] for x in report['rows']),
                'accepted_residual_writes': sum(bool(x['write'] and x['write']['guard']['changed']) for x in report['rows'])}
            save(args.output, report)
            print(json.dumps({'mode': args.mode, **report['summary']}), flush=True)
        except Exception as exc:
            report['failures'].append({'index': i, 'error': repr(exc), 'partial_entry': entry})
            save(args.output, report)
            raise
    report['final_snapshot'] = snapshot(agent, state, reference, args.states/'final.pt')
    report['complete'] = True
    save(args.output, report)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'evaluate'))
    p.add_argument('--mode', choices=('hybrid', 'hyper', 'residual'), default='hybrid')
    p.add_argument('--tag', default='20261010')
    p.add_argument('--device', default='cuda:2')
    p.add_argument('--gpu-fraction', type=float, default=.65)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, choices=range(1, 241), default=240)
    p.add_argument('--source-report', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    return p


def paths(args):
    stem = f'alfworld_cold_joint_memory_v63_{args.mode}_{args.tag}'
    args.review = Path('data/annotations')/f'{stem}_review.json'
    args.annotations = Path('data/annotations')/f'{stem}_wins.json'
    args.output = Path('results/trajectory_hyperlora')/f'{stem}.json'
    args.states = Path('results/trajectory_hyperlora')/f'{stem}_states'
    return args


if __name__ == '__main__':
    args = paths(parser().parse_args())
    (prepare if args.command == 'prepare' else evaluate)(args)
