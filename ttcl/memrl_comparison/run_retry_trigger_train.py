"""Run the v34 invalid-heavy retry guided decoder on every official ALFWorld train family."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .evaluate_typed_grounded_v34 import evaluate
from .retry_trigger_actor import should_guide_retry


def prepare(origin: Path, output: Path, repeat: int, url: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    origin = origin.resolve()
    complete = read(origin / 'complete.json')
    if complete['completed'] != complete['expected']:
        raise ValueError('Incomplete official-train source')
    plan = read(origin / 'plan.json')
    sequences = [s for s in plan['alf']['sequences'] if s['repeat'] == repeat]
    if not sequences or len({s['family'] for s in sequences}) != len(sequences):
        raise ValueError('Missing or duplicate ALFWorld train families')
    bindings = {}
    for sequence in sequences:
        family = sequence['family']
        games = sequence['tasks']
        if not games or any(not game['path'].startswith('json_2.1.1/train/') or
                            sha(Path(plan['alf']['data_root']) / game['path']) != game['sha256']
                            for game in games):
            raise ValueError(f'Invalid official-train input binding: {family}')
        bindings[family] = [game['sha256'] for game in games]
    runner = Path(__file__)
    evaluator = Path(evaluate.__code__.co_filename)
    policy = runner.with_name('grounded_evidence_v27.py')
    actor_policy = runner.with_name('retry_trigger_actor.py')
    guided_cell = runner.with_name('retry_trigger_alf_cell.py')
    guided_base = runner.with_name('guided_choice_actor.py')
    design = dict(schema='retry_trigger_train_online_v1', origin=str(origin),
                  origin_plan_sha256=sha(origin / 'plan.json'),
                  origin_complete_sha256=sha(origin / 'complete.json'),
                  runner_sha256=sha(runner), evaluator_sha256=sha(evaluator),
                  policy_sha256=sha(policy), actor_policy_sha256=sha(actor_policy),
                  retry_cell_sha256=sha(guided_cell), guided_base_sha256=sha(guided_base),
                  repeat=repeat, url=url,
                  bindings=bindings, families=list(bindings),
                  note='All planned official-train families; candidate uses guided choice only after failed attempts with public invalid-action rate above 0.4; v27 native-text memory; baseline actor and memory remain native; independent online chains and paired seeds')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    (output / 'source').mkdir()
    for path in (runner, evaluator, policy, actor_policy, guided_cell, guided_base):
        shutil.copy2(path, output / 'source' / path.name)
    return design


def audit(output: Path, require_complete: bool = True) -> dict:
    design = read(output / 'design.json')
    if design['schema'] != 'retry_trigger_train_online_v1':
        raise ValueError('Unexpected retry-trigger train design')
    origin = Path(design['origin'])
    runner = Path(__file__)
    evaluator = Path(evaluate.__code__.co_filename)
    policy = runner.with_name('grounded_evidence_v27.py')
    actor_policy = runner.with_name('retry_trigger_actor.py')
    guided_cell = runner.with_name('retry_trigger_alf_cell.py')
    guided_base = runner.with_name('guided_choice_actor.py')
    for path, key in ((runner, 'runner_sha256'), (evaluator, 'evaluator_sha256'),
                      (policy, 'policy_sha256'),
                      (actor_policy, 'actor_policy_sha256'),
                      (guided_cell, 'retry_cell_sha256'),
                      (guided_base, 'guided_base_sha256')):
        if sha(path) != design[key] or sha(output / 'source' / path.name) != design[key]:
            raise ValueError(f'Frozen source changed: {path.name}')
    if (sha(origin / 'plan.json') != design['origin_plan_sha256'] or
            sha(origin / 'complete.json') != design['origin_complete_sha256']):
        raise ValueError('Official-train source changed')
    plan = read(origin / 'plan.json')
    planned = {s['family']: s for s in plan['alf']['sequences']
               if s['repeat'] == design['repeat']}
    if list(planned) != design['families']:
        raise ValueError('Family set or order changed')
    rows, missing = [], []
    for family in design['families']:
        sequence = planned[family]
        if [x['sha256'] for x in sequence['tasks']] != design['bindings'][family]:
            raise ValueError(f'Input binding changed: {family}')
        folder = output / family
        frozen = read(folder / 'design.json') if (folder / 'design.json').exists() else None
        if frozen and (frozen['implementation_sha256'] != design['policy_sha256'] or
                       frozen['runner_sha256'] != design['evaluator_sha256'] or
                       frozen['retry_actor_sha256'] != design['actor_policy_sha256'] or
                       frozen['retry_cell_sha256'] != design['retry_cell_sha256'] or
                       frozen['guided_base_sha256'] != design['guided_base_sha256'] or
                       frozen['bindings'] != design['bindings'][family] or
                       frozen['repeat'] != design['repeat'] or frozen['url'] != design['url']):
            raise ValueError(f'Family run design changed: {family}')
        for index, game in enumerate(sequence['tasks'], 1):
            paths = [folder / arm / f'episode_{index:03d}' / 'row.json'
                     for arm in ('vanilla', 'typed_grounded')]
            if not all(path.exists() for path in paths):
                missing.append(dict(family=family, index=index))
                continue
            native, candidate = [read(path) for path in paths]
            if any(row['input_sha256'] != game['sha256'] or
                   row['game'] != game['path'] or row['status'] != 'complete' or
                   row['reward'] not in (0, 1) or row['first_attempt'] not in (0, 1)
                   for row in (native, candidate)):
                raise ValueError(f'Invalid paired official-train row: {family}/{index}')
            if (candidate.get('retry_trigger_decoder') is not True or
                    candidate['actor_calls'] >
                    plan['alf']['max_steps'] * plan['alf']['max_attempts']):
                raise ValueError(f'Retry actor budget/marker mismatch: {family}/{index}')
            guided_steps = 0
            guided_attempts = 0
            total_steps = 0
            previous = None
            for attempt in range(1, candidate['attempts'] + 1):
                episode = read(folder / 'typed_grounded' /
                               f'episode_{index:03d}' /
                               f'attempt_{attempt}' / 'episode.json')
                expected = should_guide_retry(previous)
                previous_invalid_rate = (sum(not step['valid_command'] for step in
                    previous['trajectory']) / len(previous['trajectory'])
                    if previous else None)
                if (episode.get('retry_triggered') is not expected or
                        episode.get('guided_choice_decoder') is not expected or
                        episode.get('actor_adapter_enabled') is not expected or
                        episode.get('prior_invalid_rate') != previous_invalid_rate or
                        len(episode['trajectory']) != episode['steps'] or
                        len(episode['generations']) != episode['steps'] or
                        any(generation.get('guided_choice_used') is not expected
                            for generation in episode['generations']) or
                        (expected and (any(step['valid_command'] is not True
                                           for step in episode['trajectory']) or
                         any(not generation.get('guided_choice_sha256') or
                             generation.get('guided_choice_count', 0) < 1
                             for generation in episode['generations'])))):
                    raise ValueError(f'Invalid retry attempt: {family}/{index}/{attempt}')
                guided_steps += episode['steps'] if expected else 0
                guided_attempts += int(expected)
                total_steps += episode['steps']
                previous = episode
            if (total_steps != candidate['actor_calls'] or
                    guided_steps != candidate['guided_choice_calls'] or
                    guided_attempts != candidate['guided_attempts']):
                raise ValueError(f'Hidden or misrouted actor calls: {family}/{index}')
            rows.append(dict(family=family, index=index,
                             input_sha256=game['sha256'],
                             native=float(native['reward']),
                             candidate=float(candidate['reward']),
                             delta=float(candidate['reward'] - native['reward']),
                             first_delta=float(candidate['first_attempt'] -
                                               native['first_attempt']),
                             native_calls=native['actor_calls'],
                             candidate_calls=candidate['actor_calls']))
    if require_complete and missing:
        raise ValueError(f'Missing {len(missing)} train pairs')
    by_family = {}
    for family in design['families']:
        selected = [row for row in rows if row['family'] == family]
        deltas = [row['delta'] for row in selected]
        by_family[family] = dict(pairs=len(selected),
                                 expected=len(design['bindings'][family]),
                                 wins=sum(x > 0 for x in deltas),
                                 losses=sum(x < 0 for x in deltas),
                                 ties=sum(x == 0 for x in deltas),
                                 mean_delta=statistics.fmean(deltas) if deltas else None)
    result = dict(schema='retry_trigger_train_online_analysis_v1',
                  train_split='train', expected=sum(len(x) for x in design['bindings'].values()),
                  completed=len(rows), missing=missing, rows=rows,
                  by_family=by_family,
                  note='Exploratory online train result; not independent valid-unseen or CLBench evidence')
    if not missing:
        deltas = [row['delta'] for row in rows]
        rng = random.Random(20261003)
        bootstrap = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas)))
                           for _ in range(10000))
        result.update(mean_delta=statistics.fmean(deltas),
                      game_bootstrap_95=[bootstrap[250], bootstrap[9750]],
                      wins=sum(x > 0 for x in deltas),
                      losses=sum(x < 0 for x in deltas),
                      ties=sum(x == 0 for x in deltas),
                      actor_call_delta=sum(row['candidate_calls'] - row['native_calls']
                                           for row in rows))
    return result


def run(output: Path) -> None:
    design = read(output / 'design.json')
    audit(output, require_complete=False)
    for family in design['families']:
        evaluate(Path(design['origin']), output / family, 'alfworld', family,
                 design['repeat'], design['url'], len(design['bindings'][family]), .7)
        interim = audit(output, require_complete=False)
        save(output / 'progress.json', dict(completed=interim['completed'],
                                            expected=interim['expected']))
        print(f'{family}: {interim["completed"]}/{interim["expected"]}', flush=True)
    save(output / 'analysis.json', audit(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int)
    p.add_argument('--url')
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    if a.audit_only:
        report = audit(output)
        save(output / 'analysis.json', report)
        print(f'Audited {report["completed"]}/{report["expected"]} train pairs')
        return
    if not output.exists():
        if not a.origin or a.repeat is None or not a.url:
            p.error('Preparation requires --origin, --repeat and --url')
        prepare(a.origin, output, a.repeat, a.url)
    run(output)


if __name__ == '__main__':
    main()
