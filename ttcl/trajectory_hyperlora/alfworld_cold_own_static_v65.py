"""Fresh static LoRA A+B trained only on its own chronological successes.

Matched eligibility mirrors v63; unrestricted eligibility measures ordinary
LoRA without the hypernetwork's family-source availability restriction.
Independent policy streams do not imply identical post-bootstrap labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from ttcl.trajectory_hyperlora.alfworld_bootstrap_own_success_v43 import new_agent, ordered, examples, action_rows
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import approved_success
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import static_train
from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash, digest, records_from_episode
from ttcl.trajectory_hyperlora.alfworld_online_accumulate_lora_v1 import factors_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import run_episode
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay


def bound(args):
    rows = ordered(args)[:args.limit]
    dependencies = ['alfworld_cold_own_static_v65.py', 'alfworld_online_shared_trajectory_v53.py',
        'alfworld_bootstrap_own_success_v43.py', 'alfworld_online_own_trajectory_v52b.py',
        'alfworld_zero_shot_probe.py', 'direct_composition_pilot.py', 'train_alf_next_task.py',
        'audit_my_online_update_v28_reward.py']
    fixed = {'method': 'v65 fresh static A+B own-success prequential stream',
        'eligibility': args.eligibility, 'seed': args.seed, 'limit': args.limit,
        'rank': 4, 'layers': 2, 'bootstrap_wins': 8, 'bootstrap_steps': 300,
        'bootstrap_lr': 1e-4, 'online_steps_per_eligible_win': 20, 'online_lr': 1e-5,
        'max_steps': 30, 'max_new_tokens': 64, 'history_turns': 2,
        'loop_guard_max': 2, 'constrain_actions': True,
        'initialization': 'same new_agent seed and fresh A as v63; static B zero; H unused and frozen',
        'bootstrap_targets': 'same-family cross-task action list from first8 own wins, same examples helper as v63',
        'post_bootstrap_trainables': 'A+B; no hypernetwork, no persistent failure residual',
        'source_report_sha256': file_hash(args.source_report),
        'model_config_sha256': file_hash(args.model/'config.json'),
        'code_sha256': {f: file_hash(Path(__file__).with_name(f)) for f in dependencies}}
    targets = []
    for i,row in enumerate(rows):
        content = {'index': i, 'game': row['game'], 'family': row['family'],
            'game_sha256': file_hash(args.data_root/row['game']),
            'initial_observation_sha256': digest(row['arms']['base']['initial_observation']),
            'protocol_sha256': digest(fixed)}
        targets.append({**content, 'input_content_sha256': digest(content), 'reviewed_target': True})
    return rows, fixed, targets


def checked(args):
    rows, fixed, targets = bound(args)
    assert json.loads(args.review.read_text()) == {**fixed, 'targets': targets, 'targets_sha256': digest(targets)}
    return rows, fixed, targets


def model_hash(agent, factors):
    return factors_hash([layer.a for layer in agent.adapters]+list(factors))


def snapshot(agent, factors, path):
    torch.save({'A': [x.a.detach().cpu().clone() for x in agent.adapters],
        'B': [x.detach().cpu().clone() for x in factors]}, path)
    return {'path': str(path), 'sha256': file_hash(path)}


def evaluate(args):
    if any(p.exists() for p in (args.output, args.annotations, args.states)):
        raise FileExistsError('Use fresh own-static outputs')
    _, fixed, targets = checked(args)
    args.states.mkdir(parents=True)
    save(args.annotations, {'protocol': 'v65 new own-win action targets after official environment replay',
        'eval_review_sha256': file_hash(args.review), 'annotations': []})
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=args.device)
    agent, tokenizer = new_agent(args)
    # Only the direct LoRA matrices can receive gradients.
    for p in agent.parameters():
        p.requires_grad_(False)
    for layer in agent.adapters:
        layer.a.requires_grad_(True)
    factors = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(
        1, layer.base.out_features, layer.rank, device=args.device)) for layer in agent.adapters])
    params = [x.a for x in agent.adapters]+list(factors)
    agent.eval()
    pool, optimizer = [], None
    rng = random.Random(args.seed)
    report = {**fixed, 'review_sha256': file_hash(args.review), 'rows': [], 'failures': [],
        'bootstrap': None, 'trainable_parameter_count': sum(x.numel() for x in params),
        'initial_A_sha256': factors_hash([x.a for x in agent.adapters]),
        'initial_parameters_sha256': model_hash(agent, factors),
        'initial_snapshot': snapshot(agent, factors, args.states/'initial.pt')}
    for i,target in enumerate(targets):
        same = sum(x['family'] == target['family'] for x in pool)
        active = optimizer is not None and (args.eligibility == 'unrestricted' or same >= 2)
        entry = {'index': i, 'game': target['game'],
            'target_input_content_sha256': target['input_content_sha256'],
            'memory_before': len(pool), 'family_wins_before': same, 'adapter_active': active,
            'parameters_before_sha256': model_hash(agent, factors), 'update': {'steps': 0}}
        try:
            episode = run_episode(agent, tokenizer, args.data_root/target['game'], {},
                adapter=False, fixed_adapter=list(factors) if active else None,
                device=args.device, max_steps=30, max_new_tokens=64, constrain_actions=True,
                actor_history_turns=2, loop_guard_max=2)
            entry['episode'] = episode
            if episode['status'] != 'complete' or digest(episode['initial_observation']) != target['initial_observation_sha256']:
                raise ValueError('Incomplete or changed static episode')
            replay(args.data_root/target['game'], episode)
            if episode['reward'] == 1:
                item = approved_success(args, target, episode, i, args.annotations)
                entry['annotation_input_content_sha256'] = item['annotation_input_content_sha256']
                if optimizer is None:
                    pool.append(item)
                    if len(pool) == 8:
                        items = [action for _,_,action in examples(pool)]
                        if not items:
                            raise ValueError('No bootstrap targets')
                        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
                        entry['update'] = static_train(agent, tokenizer, factors, optimizer,
                            items, 300, rng, args.device)
                        report['bootstrap'] = {'index': i, 'own_wins': 8, 'action_targets': len(items),
                            'targets_sha256': digest(items)}
                        optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
                else:
                    if args.eligibility == 'unrestricted' or same >= 1:
                        entry['update'] = static_train(agent, tokenizer, factors, optimizer,
                            item['actions'], 20, rng, args.device)
                    pool.append(item)
            entry['parameters_after_sha256'] = model_hash(agent, factors)
            if entry['parameters_after_sha256'] != entry['parameters_before_sha256']:
                entry['snapshot'] = snapshot(agent, factors, args.states/f'state_{i:03d}.pt')
            report['rows'].append(entry)
            report['summary'] = {'n': len(report['rows']), 'wins': sum(x['episode']['reward'] for x in report['rows']),
                'own_successes': len(pool), 'training_steps': sum(x['update']['steps'] for x in report['rows'])}
            save(args.output, report)
            print(json.dumps({'eligibility': args.eligibility, **report['summary']}), flush=True)
        except Exception as exc:
            report['failures'].append({'index': i, 'error': repr(exc), 'partial_entry': entry})
            save(args.output, report)
            raise
    report['final_snapshot'] = snapshot(agent, factors, args.states/'final.pt')
    report['complete'] = True
    save(args.output, report)


def audit(args):
    _, fixed, targets = checked(args)
    report = json.loads(args.output.read_text())
    annotations = json.loads(args.annotations.read_text())
    assert report.get('complete') and not report['failures']
    assert all(report[k] == v for k,v in fixed.items())
    assert report['review_sha256'] == annotations['eval_review_sha256'] == file_hash(args.review)
    assert len(report['rows']) == len(targets)
    def check_snapshot(snap, expected):
        assert file_hash(Path(snap['path'])) == snap['sha256']
        state = torch.load(snap['path'], map_location='cpu', weights_only=True)
        assert factors_hash(state['A']+state['B']) == expected
        return state
    initial = check_snapshot(report['initial_snapshot'], report['initial_parameters_sha256'])
    assert factors_hash(initial['A']) == report['initial_A_sha256']
    assert all(torch.count_nonzero(x) == 0 for x in initial['B'])
    pool, previous = [], report['initial_parameters_sha256']
    for i,(row,target) in enumerate(zip(report['rows'],targets,strict=True)):
        assert row['index'] == i and row['game'] == target['game']
        assert row['target_input_content_sha256'] == target['input_content_sha256']
        assert row['parameters_before_sha256'] == previous and row['memory_before'] == len(pool)
        same = sum(x['family'] == target['family'] for x in pool)
        assert same == row['family_wins_before']
        assert row['adapter_active'] == (len(pool) >= 8 and (args.eligibility == 'unrestricted' or same >= 2))
        ep = row['episode']
        assert ep['status'] == 'complete' and ep['steps'] <= 30
        assert digest(ep['initial_observation']) == target['initial_observation_sha256']
        replay(args.data_root/target['game'], ep)
        steps = 0
        if ep['reward'] == 1:
            actions = action_rows(args.data_root/target['game'], ep)
            content = {'stream_index': i, 'game': target['game'], 'family': target['family'],
                'game_sha256': target['game_sha256'], 'episode_sha256': digest(ep),
                'source_records_sha256': digest(records_from_episode(ep)), 'actions_sha256': digest(actions),
                'runner_sha256': fixed['code_sha256']['alfworld_online_own_trajectory_v52b.py']}
            annotation = annotations['annotations'][len(pool)]
            assert all(annotation[k] == v for k,v in content.items())
            assert annotation['input_content_sha256'] == digest(content) == row['annotation_input_content_sha256']
            if len(pool) == 7:
                steps = 300
            elif len(pool) >= 8 and (args.eligibility == 'unrestricted' or same >= 1):
                steps = 20
            pool.append({'game': target['game'], 'family': target['family'], 'actions': actions})
            if steps == 300:
                items = [a for _,_,a in examples(pool)]
                assert report['bootstrap'] == {'index': i, 'own_wins': 8, 'action_targets': len(items), 'targets_sha256': digest(items)}
        assert row['update']['steps'] == steps
        changed = row['parameters_after_sha256'] != previous
        assert ('snapshot' in row) == changed
        if not steps:
            assert not changed
        if changed:
            check_snapshot(row['snapshot'], row['parameters_after_sha256'])
        previous = row['parameters_after_sha256']
    check_snapshot(report['final_snapshot'], previous)
    assert len(annotations['annotations']) == len(pool)
    assert report['summary'] == {'n': len(targets), 'wins': sum(x['episode']['reward'] for x in report['rows']),
        'own_successes': len(pool), 'training_steps': sum(x['update']['steps'] for x in report['rows'])}
    result = {'passed': True, 'episodes_replayed': len(targets), 'summary': report['summary'],
        'report_sha256': file_hash(args.output), 'scope': 'environment replay, new annotations, chronology and saved parameter chain; no optimizer replay'}
    save(args.output.with_suffix('.audit.json'), result)
    print(json.dumps(result), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare','evaluate','audit'))
    p.add_argument('--eligibility', choices=('matched','unrestricted'), default='matched')
    p.add_argument('--tag', default='20261010')
    p.add_argument('--limit', type=int, choices=range(1,241), default=240)
    p.add_argument('--device', default='cuda:3')
    p.add_argument('--gpu-fraction', type=float, default=.8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source-report', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    return p


def paths(args):
    stem = f'alfworld_cold_own_static_v65_{args.eligibility}_{args.tag}'
    args.review = Path('data/annotations')/f'{stem}_review.json'
    args.annotations = Path('data/annotations')/f'{stem}_wins.json'
    args.output = Path('results/trajectory_hyperlora')/f'{stem}.json'
    args.states = Path('results/trajectory_hyperlora')/f'{stem}_states'
    return args


if __name__ == '__main__':
    args = paths(parser().parse_args())
    if args.command == 'prepare':
        if args.review.exists():
            raise FileExistsError(args.review)
        _,fixed,targets = bound(args)
        save(args.review, {**fixed,'targets':targets,'targets_sha256':digest(targets)})
        print(json.dumps({'review':str(args.review),'tasks':len(targets)}),flush=True)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        audit(args)
