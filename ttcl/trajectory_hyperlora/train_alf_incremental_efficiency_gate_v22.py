"""Train-only test of weak completion-efficiency credit for LoRA writes.

The official terminal reward remains the selection and development metric.
When both paired arms win, a bounded step difference supplies an additional
training target. Both-fail pairs remain neutral. No task/action hand slots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.train_alf_incremental_kernel_gate_v17 import (
    gain, inputs, kernel, predict,
)


SHAPE_WEIGHTS = (0., .1, .25)
FIXED_KERNEL = ('product', 1., 1., 10.)


def step_credit(args, review, report_hashes):
    episodes = {}
    for shard in range(3):
        path = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_20261008.json'
        if file_hash(path) != report_hashes[shard]:
            raise ValueError('Changed audited reward source')
        for row in json.loads(path.read_text())['rows']:
            key = row['input_content_sha256']
            if key in episodes:
                raise ValueError('Duplicate paired episode')
            episodes[key] = row
    credits = []
    for binding in review['pairs']:
        row = episodes[binding['input_content_sha256']]
        freeze, update = row['freeze'], row['update']
        if (row['target_id'] != binding['target_id'] or
                row['transition_index'] != binding['transition_index'] or
                not 1 <= freeze['steps'] <= 50 or
                not 1 <= update['steps'] <= 50):
            raise ValueError('Changed paired completion-time binding')
        # A failed episode's duration is not a progress measure. One win
        # against one loss is already labeled by the official terminal reward.
        credits.append((freeze['steps']-update['steps'])/50
                       if freeze['reward'] == update['reward'] == 1 else 0.)
    return np.asarray(credits, dtype='float64')


def run(args):
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError('New gate requires fresh output paths')
    (review, features, source, task, normalizers, y, freeze, update,
     task_ids, source_ids, splits, folds,
     report_hashes, audit_hashes) = inputs(args)
    credit = step_credit(args, review, report_hashes)
    if len(credit) != len(y) or np.any(np.abs(credit) > 1.):
        raise ValueError('Invalid bounded completion credit')
    train = np.flatnonzero(splits == 'train')
    dev = np.flatnonzero(splits == 'dev')
    if len(train) != 360 or len(dev) != 108:
        raise ValueError('Changed frozen train/dev split')
    candidates = [('always_update', None), ('always_freeze', None)] + [
        ('efficiency_product', weight) for weight in SHAPE_WEIGHTS]
    matrix = kernel(source, task, task_ids, source_ids, FIXED_KERNEL)
    outcomes = []
    for name, weight in candidates:
        candidate = (name, None, None, None) if name.startswith('always_') \
            else FIXED_KERNEL
        target = y if weight is None else y+weight*credit
        target_cv = np.zeros(len(train))
        source_cv = np.zeros(len(train))
        for fold in range(5):
            held = train[folds[train] == fold]
            fit = train[folds[train] != fold]
            target_cv[np.searchsorted(train, held)] = predict(
                matrix if weight is not None else None,
                target, fit, held, candidate)
        for source_id in range(6):
            held = train[source_ids[train] == source_id]
            fit = train[source_ids[train] != source_id]
            source_cv[np.searchsorted(train, held)] = predict(
                matrix if weight is not None else None,
                target, fit, held, candidate)
        outcomes.append({'candidate': name,
            'shape_weight': weight,
            'target_cv_official_gain_vs_freeze': gain(y[train], target_cv),
            'source_cv_official_gain_vs_freeze': gain(y[train], source_cv),
            'joint_official_cv_gain': gain(y[train], target_cv) +
                                      gain(y[train], source_cv)})
    selected_index = max(range(len(outcomes)), key=lambda i: (
        outcomes[i]['joint_official_cv_gain'], -i))
    selected_name, selected_weight = candidates[selected_index]
    selected = (selected_name, None, None, None) if selected_weight is None \
        else FIXED_KERNEL
    target = y if selected_weight is None else y+selected_weight*credit
    dev_prediction = predict(matrix if selected_weight is not None else None,
                             target, train, dev, selected)
    model = {'protocol': 'Frozen v22 one-to-two weak completion-efficiency credit gate, trained and selected only on v14 train official reward group CV; no family/action input',
        'pool_review_sha256': file_hash(args.pool_review),
        'features_sha256': file_hash(args.features),
        'selected': selected, 'shape_weight': selected_weight,
        'source_embeddings': torch.from_numpy(source).float(),
        'task_embeddings': torch.from_numpy(task).float(),
        **normalizers,
        'train_task_ids': torch.from_numpy(task_ids[train]),
        'train_source_ids': torch.from_numpy(source_ids[train]),
        'train_labels': torch.from_numpy(target[train]).float()}
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.model_output)
    value = {'protocol': model['protocol'],
        'pool_review_sha256': file_hash(args.pool_review),
        'features_sha256': file_hash(args.features),
        'report_sha256': report_hashes,
        'audit_sha256': audit_hashes,
        'candidates': outcomes,
        'selected_index': selected_index,
        'selected': selected,
        'selected_shape_weight': selected_weight,
        'train': {'pairs': len(train),
            'both_win_with_step_difference': int(np.sum((freeze[train] == 1) &
                (update[train] == 1) & (credit[train] != 0))),
            'official_label_nonzero': int(np.count_nonzero(y[train])),
            'always_freeze': float(freeze[train].sum()),
            'always_update': float(update[train].sum())},
        'dev': {'pairs': len(dev),
            'always_freeze': float(freeze[dev].sum()),
            'always_update': float(update[dev].sum()),
            'selected_success': float(freeze[dev].sum()+
                gain(y[dev], dev_prediction)),
            'selected_update_decisions': int((dev_prediction > 0).sum()),
            'rows': [{'target_id': int(task_ids[i]),
                'transition_index': int(source_ids[i]),
                'freeze': float(freeze[i]), 'update': float(update[i]),
                'prediction': float(dev_prediction[k]),
                'write': bool(dev_prediction[k] > 0)}
                for k, i in enumerate(dev)]},
        'model_sha256': file_hash(args.model_output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False,
                                      indent=2)+'\n')
    print(json.dumps({'selected': selected,
        'selected_shape_weight': selected_weight,
        'train': value['train'],
        'dev': {key: val for key, val in value['dev'].items()
                if key != 'rows'}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_pool78_v17_features_20261008.pt'))
    parser.add_argument('--results-root', type=Path, default=Path(
        'results/trajectory_hyperlora'))
    parser.add_argument('--model-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_efficiency_gate_v22_20261008.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_efficiency_gate_v22_20261008.json'))
    run(parser.parse_args())
