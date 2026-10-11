"""Frozen v60 final H, continual A+B LoRA, and base on official unseen134."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import digest, file_hash
from ttcl.trajectory_hyperlora.alfworld_persistent_failure_delta_v36 import save
from ttcl.trajectory_hyperlora.alfworld_zero_shot_probe import load_agent, run_episode
from ttcl.trajectory_hyperlora.alfworld_online_own_trajectory_v52b import bounded_source_fields
from ttcl.trajectory_hyperlora.alfworld_online_shared_trajectory_v53 import alignment_pair
from ttcl.trajectory_hyperlora.train_alfworld_feedback_stream_v60 import checked
from ttcl.trajectory_hyperlora.audit_my_online_update_v28_reward import replay
from ttcl.trajectory_hyperlora.analyze_alfworld_online_shared_trajectory_v53 import paired

ARMS = ('hyper', 'continual_lora', 'base')


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'evaluate', 'audit'))
    p.add_argument('--model', type=Path, default=Path('current_work/delta-Mem/model/Qwen3-4B-Instruct-2507'))
    p.add_argument('--data-root', type=Path, default=Path('ttcl/data/alfworld_delta'))
    p.add_argument('--training-dir', type=Path, default=Path('results/trajectory_hyperlora/alfworld_feedback_stream_v60_20261010'))
    p.add_argument('--training-review', type=Path, default=Path('data/annotations/alfworld_feedback_stream_v60_20261010.json'))
    p.add_argument('--original-review', type=Path, default=Path('data/annotations/alfworld_source_permutation_v42_134_20261010.json'))
    p.add_argument('--review', type=Path, default=Path('data/annotations/alfworld_feedback_unseen_v68_20261010.json'))
    p.add_argument('--output', type=Path, default=Path('results/trajectory_hyperlora/alfworld_feedback_unseen_v68_20261010.json'))
    p.add_argument('--device', default='cuda:6')
    return p


def bound(args):
    root = args.training_dir
    source = checked(SimpleNamespace(review=args.training_review, data_root=args.data_root))
    audit = json.loads((root/'audit.json').read_text())
    assert audit['passed'] and audit['replayed_episodes'] == 288
    assert audit['training_sha256'] == file_hash(root/'training.json')
    assert audit['evaluation_sha256'] == file_hash(root/'evaluation.json')
    training = json.loads((root/'training.json').read_text())
    snapshot = next(x for x in training['snapshots'] if x['label'] == '120')
    for arm in ('hyper', 'static'):
        assert snapshot[arm+'_sha256'] == file_hash(root/f'{arm}_120.pt')
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['review_sha256'] == file_hash(args.training_review)
    assert manifest['model_config_sha256'] == file_hash(args.model/'config.json')
    for name, value in manifest['dependencies'].items():
        assert value == file_hash(Path(__file__).with_name(name))
    latest = {r['family']: r for r in source['stream']}
    assert all('/train/' in r['game'] for r in source['stream'])
    original = json.loads(args.original_review.read_text())['targets']
    official = {str(p.relative_to(args.data_root)) for p in (args.data_root/'json_2.1.1/valid_unseen').glob('*/trial_*/game.tw-pddl')}
    assert len(original) == len(official) == 134
    assert {r['game'] for r in original} == official
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    records = {family: alignment_pair(r['source_records'], tok)[0] for family, r in latest.items()}
    targets = []
    for i, row in enumerate(original):
        assert row['index'] == i and row['game_sha256'] == file_hash(args.data_root/row['game'])
        family = Path(row['game']).parts[2].split('-')[0]
        assert family == row['family']
        s = latest[family]
        content = {'index': i, 'game': row['game'], 'game_sha256': row['game_sha256'],
            'family': family, 'source_index': s['index'], 'source_binding': s['input_content_sha256'],
            'source_records_sha256': digest(records[family])}
        targets.append({**content, 'input_content_sha256': digest(content), 'reviewed_target': True})
    fixed = {'protocol': 'User-requested official unseen134 re-evaluation, previously exposed; frozen v60 prefix120 models and source pool; no test updates or retries',
        'checkpoint_prefix': 120, 'arms': list(ARMS), 'max_steps': 50, 'max_new_tokens': 64,
        'history_turns': 2, 'loop_guard_max': 2, 'constrain_actions': True,
        'training_review_sha256': file_hash(args.training_review), 'training_audit_sha256': file_hash(root/'audit.json'),
        'training_sha256': file_hash(root/'training.json'), 'manifest_sha256': file_hash(root/'manifest.json'),
        'hyper_sha256': snapshot['hyper_sha256'], 'static_sha256': snapshot['static_sha256'],
        'original_review_sha256': file_hash(args.original_review),
        'runner_sha256': file_hash(Path(__file__)), 'dependencies': manifest['dependencies'],
        'primary_comparisons': [['hyper', 'continual_lora'], ['hyper', 'base'], ['continual_lora', 'base']],
        'targets': targets, 'targets_sha256': digest(targets)}
    return fixed, records


def evaluate(args, review, records):
    if args.output.exists():
        raise FileExistsError(args.output)
    agent, tok = load_agent(args.model, args.training_dir/'hyper_120.pt', args.device, .65)
    hyper_a = [layer.a.detach().clone() for layer in agent.adapters]
    state = torch.load(args.training_dir/'static_120.pt', map_location='cpu', weights_only=True)
    assert state['review_sha256'] == review['training_review_sha256'] and state['prefix_length'] == 120
    factors = [x.to(args.device) for x in state['factors']]
    for p in agent.parameters():
        p.requires_grad_(False)
    fields = {k: bounded_source_fields(agent, tok, v, args.device) for k, v in records.items()}
    result = {'review_sha256': file_hash(args.review), 'protocol': review['protocol'], 'rows': [], 'failures': []}
    for target in review['targets']:
        row = {k: target[k] for k in ('index', 'game', 'family', 'input_content_sha256', 'source_records_sha256')}
        try:
            for arm in ARMS:
                with torch.no_grad():
                    for layer, a in zip(agent.adapters, state['A'] if arm == 'continual_lora' else hyper_a, strict=True):
                        layer.a.copy_(a.to(layer.a))
                ep = run_episode(agent, tok, args.data_root/target['game'], fields[target['family']] if arm == 'hyper' else {},
                    adapter=arm == 'hyper', fixed_adapter=factors if arm == 'continual_lora' else None,
                    device=args.device, max_steps=50, max_new_tokens=64, constrain_actions=True,
                    actor_history_turns=2, loop_guard_max=2)
                row[arm] = ep
                assert ep['status'] == 'complete' and ep['reward'] in (0, 1) and ep['steps'] <= 50
                assert ep['initial_observation'] == row['hyper']['initial_observation']
            result['rows'].append(row)
            result['summary'] = {'n': len(result['rows']), **{a: sum(r[a]['reward'] for r in result['rows']) for a in ARMS}}
            save(args.output, result)
            print(json.dumps(result['summary']), flush=True)
        except Exception as exc:
            result['failures'].append({'index': target['index'], 'error': repr(exc), 'partial_row': row})
            save(args.output, result)
            raise
    result['complete'] = True
    save(args.output, result)


def audit(args, review):
    report = json.loads(args.output.read_text())
    assert report.get('complete') and not report['failures'] and len(report['rows']) == 134
    assert report['review_sha256'] == file_hash(args.review)
    count = 0
    for target, row in zip(review['targets'], report['rows'], strict=True):
        assert all(row[k] == target[k] for k in ('index', 'game', 'family', 'input_content_sha256', 'source_records_sha256'))
        for arm in ARMS:
            ep = row[arm]
            assert ep['status'] == 'complete' and ep['steps'] <= 50
            assert ep['initial_observation'] == row['hyper']['initial_observation']
            replay(args.data_root/row['game'], ep)
            count += 1
        if count % 30 == 0:
            print(json.dumps({'replayed_episodes': count}), flush=True)
    assert report['summary'] == {'n': 134, **{a: sum(r[a]['reward'] for r in report['rows']) for a in ARMS}}
    comparisons = [paired(report['rows'], a, b) for a, b in review['primary_comparisons']]
    running = 0.
    for i, item in enumerate(sorted(comparisons, key=lambda x: x['mcnemar_exact_two_sided_p'])):
        running = max(running, min(1., (3-i)*item['mcnemar_exact_two_sided_p']))
        item['holm_adjusted_p'] = running
    result = {'passed': True, 'replayed_episodes': count, 'report_sha256': file_hash(args.output),
        'review_sha256': file_hash(args.review), 'summary': report['summary'], 'comparisons': comparisons}
    save(args.output.with_suffix('.audit.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    args = parser().parse_args()
    fixed, records = bound(args)
    if args.command == 'prepare':
        if args.review.exists():
            raise FileExistsError(args.review)
        save(args.review, fixed)
        print(json.dumps({'reviewed_targets': 134, 'checkpoint_prefix': 120, 'arms': list(ARMS)}), flush=True)
    else:
        assert json.loads(args.review.read_text()) == fixed
        if args.command == 'evaluate':
            evaluate(args, fixed, records)
        else:
            audit(args, fixed)
