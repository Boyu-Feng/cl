"""CPU-only, source-bound paired-credit learner for MemRL retrieval.

This module consumes audited *training* replays.  It never reads evaluation
rewards, and never treats a memory ID alone as a label binding.  The learned
decision is a conservative contextual-bandit decision: remove a retrieved
entry only when paired training outcomes support a negative marginal value.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics

import numpy as np

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .analyze_cl_credit_probe import analyze as audit_cl
from .analyze_credit_probe import analyze as audit_alf
from .analyze_remaining_credit_probe import analyze as audit_alf_remaining
from .credit_probe import memory_arms


FEATURES = (
    'bias', 'similarity', 'q_estimate', 'score', 'q_value', 'q_visits',
    'reward_ma', 'last_reward', 'official_reward', 'success',
    'text_characters', 'writer_limit', 'set_size', 'position',
)
RIDGE = 8.0
BOOTSTRAPS = 400
NEGATIVE_MARGIN = 0.05


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def features(memory_features: dict, set_size: int, position: int) -> list[float]:
    """Only retrieval-time values; no task/domain words or outcome of this task."""
    if set_size < 1 or not 0 <= position < set_size:
        raise ValueError('Invalid retrieval set position')
    metadata = memory_features['metadata']
    retrieval = memory_features['retrieval']
    success = metadata.get('success')
    result = [
        1.0,
        max(-1.0, min(1.0, _number(retrieval.get('similarity')))),
        math.tanh(_number(retrieval.get('q_estimate'))),
        math.tanh(_number(retrieval.get('score'))),
        math.tanh(_number(metadata.get('q_value'))),
        math.log1p(max(0.0, _number(metadata.get('q_visits')))) / 4.0,
        math.tanh(_number(metadata.get('reward_ma'))),
        math.tanh(_number(metadata.get('last_reward'))),
        math.tanh(_number(metadata.get('official_reward'))),
        1.0 if success is True else -1.0 if success is False else 0.0,
        math.log1p(max(0.0, _number(memory_features.get('text_characters')))) / 8.0,
        float(metadata.get('writer_token_limit_hit') is True),
        math.log1p(set_size) / 2.0,
        position / max(1, set_size - 1),
    ]
    assert len(result) == len(FEATURES) and all(math.isfinite(v) for v in result)
    return result


def _exact_two_memory_credit(rewards: dict, position: int) -> float:
    """Two-player Shapley value from the four matched coalition rewards."""
    if set(rewards) != {'none', 'full', 'drop_0', 'drop_1'}:
        raise ValueError('Exact two-memory credit needs all four coalitions')
    other_alone = rewards[f'drop_{position}']
    own_alone = rewards[f'drop_{1 - position}']
    return 0.5 * ((own_alone - rewards['none']) +
                  (rewards['full'] - other_alone))


def _bounded(value: float, scale: float) -> float:
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid paired reward or scale')
    return max(-1.0, min(1.0, value / scale))


def _scale(rewards: list[float]) -> float:
    """One robust unit formula for every domain, derived from train replays."""
    if not rewards or any(not math.isfinite(x) for x in rewards):
        raise ValueError('Missing or nonfinite training rewards')
    ordered = sorted(abs(x) for x in rewards)
    # Official rewards have very different units (binary success, small
    # continuous scores, chip returns).  The same robust rule gives each
    # training domain a usable signal while clipping rare extremes later.
    return max(0.05, ordered[int(0.75 * (len(ordered) - 1))])


def collect(alf_outputs: list[Path], cl_outputs: list[Path]) -> dict:
    if not alf_outputs or not cl_outputs:
        raise ValueError('Both ALFWorld train and CLBench calibration probes are required')
    observations = []
    input_hashes = {}
    for output in alf_outputs:
        second_wave = (output / 'training_selection.json').exists()
        audit = audit_alf_remaining(output) if second_wave else audit_alf(output)
        if (audit['source_split'] != 'train' or
                audit['completed'] != audit['expected'] or audit['missing']):
            raise ValueError(f'Incomplete or non-train ALFWorld probe: {output}')
        design_path = output / 'design.json'
        source_selection = (Path(read(output / 'training_selection.json')['selection_path'])
                            if second_wave else
                            Path(read(design_path)['origin']) / 'selected_probe_cases.json')
        input_hashes[str(design_path)] = sha(design_path)
        input_hashes[str(source_selection)] = sha(source_selection)
        if second_wave:
            input_hashes[str(output / 'training_selection.json')] = sha(
                output / 'training_selection.json')
        freeze_path = output / 'execution_freeze.json'
        freeze = read(freeze_path)
        if (freeze['design_sha256'] != sha(design_path) or
                freeze['selection_sha256'] != sha(source_selection)):
            raise ValueError(f'ALF probe execution freeze changed: {output}')
        for name, expected in freeze['files'].items():
            if (sha(Path(__file__).with_name(name)) != expected or
                    sha(output / 'source' / name) != expected):
                raise ValueError(f'ALF probe code changed: {name}')
        input_hashes[str(freeze_path)] = sha(freeze_path)
        for row in audit['rows']:
            observations.append(dict(
                source='alfworld_train', case=row['case'], task=row['task'],
                actor_repeat=row['actor_repeat'], memory_id=row['memory_id'],
                memory_features=row['memory_features'],
                memory_text_sha256=row['memory_text_sha256'],
                source_input_sha256=row['source_input_sha256'],
                retrieval_sha256=row['retrieval_sha256'],
                snapshot_sha256=row['snapshot_sha256'],
                set_size=len(read(output / 'alfworld' / row['task'] /
                                  row['case'].split('/')[2] /
                                  row['case'].split('/')[-1] / 'source.json')['ids']),
                position=int(read(output / 'design.json')['paired_drop_index']),
                reward_values=[row['full'], row['drop']],
                raw_delta=row['delta'], estimand='leave_one_out'))
    for output in cl_outputs:
        audit = audit_cl(output)
        if audit['completed'] != audit['expected'] or audit['missing']:
            raise ValueError(f'Incomplete CLBench calibration probe: {output}')
        design_path = output / 'design.json'
        selection_path = Path(read(design_path)['selection_path'])
        input_hashes[str(design_path)] = sha(design_path)
        input_hashes[str(selection_path)] = sha(selection_path)
        input_hashes[str(output / 'execution_freeze.json')] = sha(
            output / 'execution_freeze.json')
        for row in audit['rows']:
            ids = row['memory_ids']
            rewards = row['rewards']
            for position, mid in enumerate(ids):
                if len(ids) == 1:
                    delta = rewards['full'] - rewards['none']
                    shapley = delta
                elif len(ids) == 2:
                    delta = rewards['full'] - rewards[f'drop_{position}']
                    shapley = _exact_two_memory_credit(rewards, position)
                else:
                    delta = rewards['full'] - rewards[f'drop_{position}']
                    shapley = None
                observations.append(dict(
                    source='clbench_calibration', case=row['case'], task=row['task'],
                    actor_repeat=row['actor_repeat'], memory_id=mid,
                    memory_features=row['memory_features'][mid],
                    memory_text_sha256=row['memory_text_sha256'][mid],
                    source_input_sha256=row['source_input_sha256'],
                    retrieval_sha256=row['retrieval_sha256'],
                    snapshot_sha256=row['snapshot_sha256'],
                    set_size=len(ids), position=position,
                    reward_values=list(rewards.values()),
                    raw_delta=delta, shapley_delta=shapley,
                    estimand='leave_one_out'))
    scales = {task:_scale([v for row in observations if row['task'] == task
                           for v in row['reward_values']])
              for task in sorted({row['task'] for row in observations})}
    grouped = defaultdict(list)
    for row in observations:
        row['bounded_delta'] = _bounded(row['raw_delta'], scales[row['task']])
        binding = (row['source'], row['case'], row['memory_id'],
                   row['memory_text_sha256'], row['source_input_sha256'],
                   row['retrieval_sha256'], row['snapshot_sha256'])
        grouped[binding].append(row)
    examples = []
    for binding, rows in sorted(grouped.items()):
        first = rows[0]
        if any((r['memory_features'] != first['memory_features'] or
                r['set_size'] != first['set_size'] or
                r['position'] != first['position'] or
                r['estimand'] != first['estimand']) for r in rows):
            raise ValueError(f'Probe replicas disagree on bound input: {binding}')
        repeats = [r['actor_repeat'] for r in rows]
        if len(set(repeats)) != len(repeats):
            raise ValueError(f'Duplicate actor repeat: {binding}')
        examples.append(dict(
            binding=list(binding), task=first['task'],
            estimand=first['estimand'], actor_repeats=repeats,
            x=features(first['memory_features'], first['set_size'],
                       first['position']),
            raw_deltas=[r['raw_delta'] for r in rows],
            shapley_deltas=[r.get('shapley_delta') for r in rows],
            bounded_deltas=[r['bounded_delta'] for r in rows],
            label=statistics.fmean(r['bounded_delta'] for r in rows)))
    for path in (Path(__file__), Path(audit_alf.__code__.co_filename),
                 Path(audit_alf_remaining.__code__.co_filename),
                 Path(audit_cl.__code__.co_filename),
                 Path(memory_arms.__code__.co_filename)):
        input_hashes[str(path.resolve())] = sha(path)
    payload = dict(schema='paired_credit_training_v1', feature_names=FEATURES,
                   input_hashes=input_hashes, scales=scales, examples=examples,
                   observation_count=len(observations), example_count=len(examples))
    payload['dataset_sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload


def _fit(examples: list[dict]) -> np.ndarray:
    x = np.asarray([row['x'] for row in examples], dtype=np.float64)
    y = np.asarray([row['label'] for row in examples], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not len(x):
        raise ValueError('No complete paired-credit examples')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite training feature or label')
    group_counts = Counter(_group_key(row) for row in examples)
    weights = np.asarray([1.0 / group_counts[_group_key(row)]
                          for row in examples], dtype=np.float64)
    regularizer = np.eye(x.shape[1]) * RIDGE
    regularizer[0, 0] = RIDGE / 4.0
    return np.linalg.solve(x.T @ (weights[:, None] * x) + regularizer,
                           x.T @ (weights * y))


def _group_key(example: dict) -> str:
    """Same public input stays together across source repeats and memories."""
    binding = example['binding']
    return binding[0] + ':' + binding[4]


def fit(dataset: dict) -> dict:
    examples = dataset['examples']
    if not any(e['binding'][0] == 'alfworld_train' for e in examples) or not any(
            e['binding'][0] == 'clbench_calibration' for e in examples):
        raise ValueError('Both benchmark training sources must be represented')
    coefficients = _fit(examples)
    cases = sorted({_group_key(e) for e in examples})
    rng = random.Random(92791)
    by_case = defaultdict(list)
    for example in examples:
        by_case[_group_key(example)].append(example)
    bootstrap = []
    for _ in range(BOOTSTRAPS):
        sampled = []
        for _case in (rng.choice(cases) for _ in cases):
            group = by_case[_case]
            common_repeats = set(group[0].get('actor_repeats', []))
            for example in group[1:]:
                common_repeats.intersection_update(example.get('actor_repeats', []))
            repeat = rng.choice(sorted(common_repeats)) if common_repeats else None
            for example in group:
                if repeat is None:
                    sampled.append(example)
                else:
                    position = example['actor_repeats'].index(repeat)
                    sampled.append(dict(example,
                                        label=example['bounded_deltas'][position]))
        bootstrap.append(_fit(sampled).tolist())
    return dict(schema='paired_credit_policy_v1',
                dataset_sha256=dataset['dataset_sha256'],
                feature_names=FEATURES, ridge=RIDGE,
                bootstrap_seed=92791, bootstrap_count=BOOTSTRAPS,
                negative_margin=NEGATIVE_MARGIN,
                coefficients=coefficients.tolist(), bootstrap_coefficients=bootstrap,
                training_cases=len({e['binding'][1] for e in examples}),
                training_public_inputs=len(cases),
                training_examples=len(examples),
                decision='remove only if case-and-repeat bootstrap 95th percentile is below -margin')


def predict(policy: dict, x: list[float]) -> dict:
    if policy['feature_names'] != list(FEATURES) and tuple(policy['feature_names']) != FEATURES:
        raise ValueError('Feature schema changed')
    values = np.asarray(x, dtype=np.float64)
    if values.shape != (len(FEATURES),) or not np.isfinite(values).all():
        raise ValueError('Invalid retrieval feature')
    draws = np.asarray(policy['bootstrap_coefficients'], dtype=np.float64) @ values
    upper = float(np.quantile(draws, .95))
    mean = float(np.asarray(policy['coefficients']) @ values)
    return dict(mean=mean, upper_95=upper,
                remove=upper < -float(policy['negative_margin']))


def crossfit(dataset: dict) -> dict:
    """Content-held-out training diagnostic, never an online benchmark score."""
    examples = dataset['examples']
    by_task = defaultdict(set)
    for example in examples:
        by_task[example['task']].add(_group_key(example))
    folds = {}
    for cases in by_task.values():
        for position, case in enumerate(sorted(cases)):
            folds[case] = position % 5
    rows = []
    for fold in range(5):
        train = [e for e in examples if folds[_group_key(e)] != fold]
        held = [e for e in examples if folds[_group_key(e)] == fold]
        if not held:
            continue
        policy = fit(dict(dataset_sha256=dataset['dataset_sha256'],
                          examples=train))
        for example in held:
            decision = predict(policy, example['x'])
            rows.append(dict(case=example['binding'][1],
                             public_input_group=_group_key(example),
                             memory_text_sha256=example['binding'][3],
                             task=example['task'], fold=fold,
                             label=example['label'],
                             raw_mean_delta=statistics.fmean(example['raw_deltas']),
                             **decision))
    if len(rows) != len(examples):
        raise ValueError('Crossfit did not cover every training example')
    by_task = {}
    for task in sorted({row['task'] for row in rows}):
        subset = [row for row in rows if row['task'] == task]
        removed = [row for row in subset if row['remove']]
        by_task[task] = dict(examples=len(subset), removed=len(removed),
                             observed_negative=sum(row['label'] < 0 for row in subset),
                             observed_positive=sum(row['label'] > 0 for row in subset),
                             observed_zero=sum(row['label'] == 0 for row in subset),
                             isolated_bounded_gain=-sum(row['label'] for row in removed),
                             isolated_raw_gain=-sum(row['raw_mean_delta'] for row in removed))
    return dict(schema='paired_credit_crossfit_v1',
                dataset_sha256=dataset['dataset_sha256'],
                note=('Input-content-held-out fixed-snapshot deletion proxy only; multiple '
                      'simultaneous removals and online memory evolution are not replayed.'),
                by_task=by_task, rows=rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alf', type=Path, nargs='+', required=True)
    parser.add_argument('--cl', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Use a new training output: {output}')
    dataset = collect([p.resolve() for p in args.alf],
                      [p.resolve() for p in args.cl])
    policy = fit(dataset)
    diagnostic = crossfit(dataset)
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    for name in ('paired_credit_rl.py', 'analyze_credit_probe.py',
                 'analyze_remaining_credit_probe.py',
                 'analyze_cl_credit_probe.py', 'credit_probe.py'):
        path = Path(__file__).with_name(name)
        expected = dataset['input_hashes'][str(path.resolve())]
        if sha(path) != expected:
            raise ValueError(f'Trainer source changed before freeze: {name}')
        shutil.copy2(path, source / name)
    save(output / 'dataset.json', dataset)
    save(output / 'policy.json', policy)
    save(output / 'crossfit.json', diagnostic)
    print(json.dumps(dict(cases=policy['training_cases'],
                          examples=policy['training_examples'],
                          observations=dataset['observation_count'],
                          removed_in_crossfit=sum(x['removed'] for x in diagnostic['by_task'].values()),
                          dataset_sha256=dataset['dataset_sha256']), sort_keys=True))


if __name__ == '__main__':
    main()
