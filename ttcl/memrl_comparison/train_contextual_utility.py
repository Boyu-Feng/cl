"""Fit a frozen, conservative set-level MemRL utility gate on development runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .contextual_utility import retrieval_features, words


def collect(origin):
    runs = origin / 'runs'
    records, inputs = [], []
    for benchmark, repeat in [('alfworld', 92601), ('clbench', 303)]:
        for task_dir in sorted((runs / benchmark).iterdir()):
            if not task_dir.is_dir():
                continue
            arm = task_dir / str(repeat) / 'memrl'
            for episode in sorted(arm.glob('episode_*')):
                mem_path = episode / 'row.json'
                none_path = episode.parent.parent / 'none' / episode.name / 'row.json'
                ret_path = episode / ('retrieval_1.json' if benchmark == 'alfworld' else 'retrieval.json')
                if not all(p.exists() for p in (mem_path, none_path, ret_path)):
                    continue
                mem, none, retrieval = (read(p) for p in (mem_path, none_path, ret_path))
                if mem['status'] != 'complete' or none['status'] != 'complete':
                    continue
                prior = episode.parent / f'episode_{int(episode.name[-3:])-1:03d}' / 'memory_after.json'
                ids = retrieval['ids']
                if ids and not prior.exists():
                    raise ValueError(f'Missing prior memory state: {prior}')
                metadata = {mid: read(prior)['items'][mid]['metadata'] for mid in ids} if ids else {}
                z = retrieval_features(retrieval, metadata)
                key = 'first_attempt' if benchmark == 'alfworld' else 'reward'
                target = max(-1., min(1., float(mem[key]) - float(none[key])))
                records.append(dict(benchmark=benchmark, task=task_dir.name,
                                    repeat=repeat, episode=episode.name,
                                    query_words=sorted(words(retrieval['query'])),
                                    features=z, target=target))
                inputs.append({str(p.relative_to(origin)): sha(p) for p in
                               (mem_path, none_path, ret_path, *([prior] if ids else []))})
    return records, inputs


def fit(origin, output, threshold=-.2, penalty=4.):
    records, inputs = collect(origin)
    if not records:
        raise ValueError('No complete paired development examples')
    vocab = sorted(set().union(*(r['query_words'] for r in records)))
    lookup = {word: i+1 for i, word in enumerate(vocab)}
    matrix = np.zeros((len(records), len(vocab)+7))
    matrix[:, 0] = 1.
    for i, row in enumerate(records):
        for word in row['query_words']:
            matrix[i, lookup[word]] = 1.
        matrix[i, len(vocab)+1:] = row['features']
    targets = np.array([r['target'] for r in records])
    regularizer = np.diag([1e-6] + [penalty] * (len(vocab)+6))
    weights = np.linalg.solve(matrix.T @ matrix + regularizer, matrix.T @ targets)
    manifest = dict(schema='contextual_utility_ridge_v1',
                    training_origin=str(origin), training_plan_sha256=sha(origin/'plan.json'),
                    training_splits={'alfworld':[92601], 'clbench':[303]},
                    training_examples=len(records), training_by_benchmark={
                        b: sum(r['benchmark']==b for r in records) for b in ('alfworld','clbench')},
                    training_records=records, source_hashes=inputs,
                    ridge_penalty=penalty, threshold=threshold,
                    vocab=vocab, weights=weights.tolist(),
                    caveat='Development seed/repeat; held-out seeds reuse games or canonical tasks.')
    if output.exists() and read(output) != manifest:
        raise ValueError('Frozen gate differs; use another output path')
    save(output, manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--threshold', type=float, default=-.2)
    a = p.parse_args()
    result = fit(a.origin.resolve(), a.output.resolve(), a.threshold)
    print({k:result[k] for k in ('training_examples','training_by_benchmark','threshold')})


if __name__ == '__main__':
    main()
