"""Train a family-free full-information gate for future LoRA writes.

Inputs are frozen encodings of own old/new trajectories and the future task.
The target is paired environment reward(update) - reward(freeze). Hyperparameters
are selected only by grouped train folds; frozen dev games are read once after
selection. Always-update and always-freeze are explicit candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.alfworld_online_from_empty import file_hash
from ttcl.trajectory_hyperlora.centered_feedback_residual_v3 import (
    normalized_source, normalized_target,
)


TASK_TEMPERATURES = (.25, 1.)
SOURCE_TEMPERATURES = (.25, 1.)
RIDGE = (.1, 1., 10., 100.)
MODES = ('product', 'task', 'source', 'sum')


def unit(values: torch.Tensor) -> np.ndarray:
    return F.normalize(values.float(), dim=-1).cpu().numpy().astype('float64')


def inputs(args):
    review = json.loads(args.pool_review.read_text())
    features = torch.load(args.features, map_location='cpu', weights_only=True)
    if (features['pool_review_sha256'] != file_hash(args.pool_review) or
            features['checkpoint_sha256'] != review['checkpoint_sha256'] or
            len(review['targets']) != 78 or len(review['pairs']) != 468 or
            tuple(features['source_global'].shape) != (8, 2560) or
            tuple(features['source_delta'].shape) != (8, 2560) or
            tuple(features['target'].shape) != (78, 2560)):
        raise ValueError('Changed source/task feature lineage')
    reports = {}
    report_hashes = []
    audit_hashes = []
    for shard in range(3):
        raw = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_20261008.json'
        audited = args.results_root / f'alf_incremental_pool78_v14_shard{shard}_audited_20261008.json'
        report = json.loads(raw.read_text())
        audit = json.loads(audited.read_text())
        if (audit['raw_report_sha256'] != file_hash(raw) or
                audit['pool_review_sha256'] != file_hash(args.pool_review) or
                audit['failed_replays'] != 0 or
                report['pool_review_sha256'] != file_hash(args.pool_review) or
                report['failures'] or len(report['rows']) != 156):
            raise ValueError('Missing original-environment audited labels')
        report_hashes.append(file_hash(raw))
        audit_hashes.append(file_hash(audited))
        for row in report['rows']:
            if row['input_content_sha256'] in reports:
                raise ValueError('Duplicate train/dev paired label')
            reports[row['input_content_sha256']] = row
    if (features['report_sha256'] != report_hashes or
            features['audit_sha256'] != audit_hashes or
            len(reports) != 468):
        raise ValueError('Feature embeddings were not made from these labels')
    global_vectors = features['source_global']
    delta_vectors = features['source_delta']
    source_rows = []
    for old_id, new_id in review['transitions']:
        old = normalized_source(global_vectors[old_id], delta_vectors[old_id])
        mean = normalized_source(
            (global_vectors[old_id] + global_vectors[new_id]) / 2,
            (delta_vectors[old_id] + delta_vectors[new_id]) / 2)
        source_rows.append(torch.cat((F.normalize(old, dim=0),
                                      F.normalize(mean - old, dim=0))))
    source_raw = torch.stack(source_rows)
    source_center = source_raw.mean(0)
    source = unit(source_raw - source_center)
    task_raw = torch.stack([normalized_target(x)
                            for x in features['target']])
    train_target_ids = [row['target_id'] for row in review['targets']
                        if row['split'] == 'train']
    task_center = task_raw[train_target_ids].mean(0)
    task = unit(task_raw - task_center)
    labels = []
    freeze = []
    update = []
    task_ids = []
    source_ids = []
    splits = []
    folds = []
    for binding in review['pairs']:
        row = reports.get(binding['input_content_sha256'])
        target = review['targets'][binding['target_id']]
        if (row is None or row['target_id'] != target['target_id'] or
                row['transition_index'] != binding['transition_index'] or
                row['split'] != target['split']):
            raise ValueError('Changed paired label binding')
        a, b = row['freeze']['reward'], row['update']['reward']
        labels.append(b-a)
        freeze.append(a)
        update.append(b)
        task_ids.append(target['target_id'])
        source_ids.append(binding['transition_index'])
        splits.append(target['split'])
        folds.append(target['family_position'] % 5)
    return (review, features, source, task,
        {'source_center': source_center, 'task_center': task_center},
        np.asarray(labels, dtype='float64'),
        np.asarray(freeze, dtype='float64'),
        np.asarray(update, dtype='float64'),
        np.asarray(task_ids), np.asarray(source_ids),
        np.asarray(splits), np.asarray(folds),
        report_hashes, audit_hashes)


def candidates():
    # Stable tie order favors the previously deployed unconditional update.
    result = [('always_update', None, None, None),
              ('always_freeze', None, None, None)]
    for mode in MODES:
        for task_temperature in TASK_TEMPERATURES:
            for source_temperature in SOURCE_TEMPERATURES:
                for ridge in reversed(RIDGE):
                    result.append((mode, task_temperature,
                                   source_temperature, ridge))
    return result


def kernel(source, task, task_ids, source_ids, candidate):
    mode, tq, ts, _ = candidate
    if mode.startswith('always_'):
        return None
    task_similarity = np.clip(task @ task.T, -1., 1.)
    source_similarity = np.clip(source @ source.T, -1., 1.)
    task_kernel = np.exp(-(1-task_similarity)/tq)[
        np.ix_(task_ids, task_ids)]
    source_kernel = np.exp(-(1-source_similarity)/ts)[
        np.ix_(source_ids, source_ids)]
    if mode == 'product':
        return task_kernel * source_kernel
    if mode == 'task':
        return task_kernel
    if mode == 'source':
        return source_kernel
    if mode == 'sum':
        return (task_kernel + source_kernel)/2
    raise ValueError('Unknown pre-registered gate kernel')


def predict(matrix, labels, fit, test, candidate):
    mode, _, _, ridge = candidate
    if mode == 'always_update':
        return np.ones(len(test))
    if mode == 'always_freeze':
        return -np.ones(len(test))
    center = labels[fit].mean()
    block = matrix[np.ix_(fit, fit)] + ridge*np.eye(len(fit))
    weights = np.linalg.solve(block, labels[fit]-center)
    return center + matrix[np.ix_(test, fit)] @ weights


def gain(labels, predictions):
    return float(np.dot(labels, predictions > 0))


def run(args):
    if args.output.exists() or args.model_output.exists():
        raise FileExistsError('Frozen gate or report already exists')
    (review, features, source, task, normalizers, y, freeze, update,
     task_ids, source_ids, splits, folds,
     report_hashes, audit_hashes) = inputs(args)
    train = np.flatnonzero(splits == 'train')
    dev = np.flatnonzero(splits == 'dev')
    if len(train) != 360 or len(dev) != 108:
        raise ValueError('Changed frozen train/dev split')
    evaluated = []
    for candidate in candidates():
        matrix = kernel(source, task, task_ids, source_ids, candidate)
        target_cv = np.zeros(len(train))
        source_cv = np.zeros(len(train))
        for fold in range(5):
            held = train[folds[train] == fold]
            fit = train[folds[train] != fold]
            target_cv[np.searchsorted(train, held)] = predict(
                matrix, y, fit, held, candidate)
        for source_id in range(6):
            held = train[source_ids[train] == source_id]
            fit = train[source_ids[train] != source_id]
            source_cv[np.searchsorted(train, held)] = predict(
                matrix, y, fit, held, candidate)
        evaluated.append({'candidate': candidate,
            'target_cv_gain_vs_freeze': gain(y[train], target_cv),
            'source_cv_gain_vs_freeze': gain(y[train], source_cv),
            'joint_cv_gain': (gain(y[train], target_cv) +
                              gain(y[train], source_cv))})
    selected_index = max(range(len(evaluated)),
        key=lambda i: (evaluated[i]['joint_cv_gain'], -i))
    selected = candidates()[selected_index]
    matrix = kernel(source, task, task_ids, source_ids, selected)
    dev_prediction = predict(matrix, y, train, dev, selected)
    train_prediction = predict(matrix, y, train, train, selected)
    result = {'protocol': 'Pre-registered family-free trajectory/task kernel gate; full-information paired terminal reward; target-group and source-transition CV jointly select mode/temperature/ridge on 60 train games; 18 directory-disjoint dev games evaluated once',
        'pool_review_sha256': file_hash(args.pool_review),
        'features_sha256': file_hash(args.features),
        'report_sha256': report_hashes, 'audit_sha256': audit_hashes,
        'candidate_count': len(evaluated),
        'candidates': evaluated,
        'selected_index': selected_index, 'selected': selected,
        'train': {'pairs': len(train),
            'always_freeze': float(freeze[train].sum()),
            'always_update': float(update[train].sum()),
            'selected_in_sample': float(freeze[train].sum()+
                gain(y[train], train_prediction)),
            'selected_target_cv_gain_vs_freeze':
                evaluated[selected_index]['target_cv_gain_vs_freeze'],
            'selected_source_cv_gain_vs_freeze':
                evaluated[selected_index]['source_cv_gain_vs_freeze']},
        'dev': {'pairs': len(dev),
            'always_freeze': float(freeze[dev].sum()),
            'always_update': float(update[dev].sum()),
            'selected_success': float(freeze[dev].sum()+
                                      gain(y[dev], dev_prediction)),
            'selected_update_decisions': int((dev_prediction > 0).sum()),
            'positive_labels': int((y[dev] > 0).sum()),
            'negative_labels': int((y[dev] < 0).sum()),
            'rows': [{'target_id': int(task_ids[i]),
                'transition_index': int(source_ids[i]),
                'freeze': float(freeze[i]), 'update': float(update[i]),
                'prediction': float(dev_prediction[k]),
                'write': bool(dev_prediction[k] > 0)}
                for k, i in enumerate(dev)]}}
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model = {'protocol': result['protocol'],
        'pool_review_sha256': result['pool_review_sha256'],
        'features_sha256': result['features_sha256'],
        'selected': selected,
        'source_embeddings': torch.from_numpy(source).float(),
        'task_embeddings': torch.from_numpy(task).float(),
        **normalizers,
        'train_task_ids': torch.from_numpy(task_ids[train]),
        'train_source_ids': torch.from_numpy(source_ids[train]),
        'train_labels': torch.from_numpy(y[train]).float()}
    torch.save(model, args.model_output)
    result['model_sha256'] = file_hash(args.model_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'selected': selected, 'train': result['train'],
                      'dev': {k:v for k,v in result['dev'].items()
                              if k != 'rows'}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-review', type=Path, default=Path(
        'data/annotations/alf_incremental_pool78_v14_reviewed_20261008.json'))
    parser.add_argument('--features', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_pool78_v17_features_20261008.pt'))
    parser.add_argument('--results-root', type=Path,
                        default=Path('results/trajectory_hyperlora'))
    parser.add_argument('--model-output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_kernel_gate_v17_20261008.pt'))
    parser.add_argument('--output', type=Path, default=Path(
        'results/trajectory_hyperlora/alf_incremental_kernel_gate_v17_20261008.json'))
    run(parser.parse_args())
