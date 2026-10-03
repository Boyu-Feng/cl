"""Training-only paired effect of memory on a third ALFWorld attempt.

The source online chain is read-only. Each replay starts the same game afresh,
uses the source's third-attempt retrieval, and holds the actor seed fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from .memory import digest


ACTOR_REPEATS = (92741, 92742, 92743)
HELPERS = (
    'ttcl/alfworld_comparison/environment.py',
    'ttcl/icl_mem0_comparison/protocol.py',
    'ttcl/memrl_comparison/memory.py',
)


def targets(source: Path) -> list[dict]:
    complete = read(source / 'complete.json')
    if complete['completed'] != complete['expected']:
        raise ValueError('Incomplete training source')
    result = []
    plan = read(source / 'plan.json')
    for path in sorted((source / 'runs' / 'alfworld').glob('*/*/memrl/episode_*/row.json')):
        episode = path.parent
        row = read(path)
        retrieval_path = episode / 'retrieval_3.json'
        if not retrieval_path.exists():
            continue
        retrieval = read(retrieval_path)
        if not retrieval['context']:
            continue
        prior = [read(episode / f'attempt_{i}' / 'episode.json') for i in (1, 2)]
        original = read(episode / 'attempt_3' / 'episode.json')
        if (row['status'] != 'complete' or not row['game'].startswith('json_2.1.1/train/') or
                any(p['reward'] != 0 or p['game'] != row['game'] for p in prior) or
                original['game'] != row['game'] or
                original['memory'] != retrieval['context'] or
                original['seed'] != seed(row['repeat'], row['game'], 2, 'actor') or
                retrieval['tokens'] > plan['memory_tokens'] or
                sha(Path(plan['alf']['data_root']) / row['game']) != row['input_sha256']):
            raise ValueError(f'Unmatched training retry target: {path}')
        result.append(dict(case=str(episode.relative_to(source / 'runs')),
                           game=row['game'], input_sha256=row['input_sha256'],
                           row_sha256=sha(path), retrieval_sha256=sha(retrieval_path),
                           context_sha256=hashlib.sha256(retrieval['context'].encode()).hexdigest(),
                           original_third_sha256=sha(episode / 'attempt_3' / 'episode.json')))
    if not result:
        raise ValueError('No eligible third-attempt training targets')
    return result


def prepare(source: Path, output: Path, url: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve()
    cases = targets(source)
    module = Path(__file__)
    root = module.parents[2]
    helper_sha256 = {name: sha(root / name) for name in HELPERS}
    design = dict(schema='alf_train_retry_context_v1', source=str(source),
                  source_complete_sha256=sha(source / 'complete.json'),
                  source_plan_sha256=sha(source / 'plan.json'),
                  runner_sha256=sha(module), helper_sha256=helper_sha256, url=url,
                  actor_repeats=list(ACTOR_REPEATS), cases=cases,
                  note='Outcome-blind all eligible third-attempt official train cases; same game, retrieval and actor seed; no online Q update')
    output.mkdir(parents=True)
    save(output / 'design.json', design)
    (output / 'source').mkdir()
    shutil.copy2(module, output / 'source' / module.name)
    for name in HELPERS:
        shutil.copy2(root / name, output / 'source' / Path(name).name)
    return design


def _run_case(plan: dict, case: dict, repeat: int, output: Path) -> None:
    from ttcl.alfworld_comparison.environment import Actor

    class LocalActor(Actor):
        def generate(self, messages, random_seed):
            response = client.complete(messages, random_seed,
                                       tokens=plan['alf']['actor_max_tokens'],
                                       temperature=plan['alf']['actor_temperature'],
                                       top_p=1.)
            return dict(text=response['raw_response'],
                        finish_reason=response['finish_reason'],
                        usage={'prompt_tokens': response['input_tokens'],
                               'completion_tokens': response['output_tokens']},
                        seed=random_seed, prompt_sha256=digest(messages),
                        seconds=response['seconds'],
                        rendered_prompt_sha256=response['rendered_prompt_sha256'])

    retrieval = read(Path(plan['_source']) / 'runs' / case['case'] / 'retrieval_3.json')
    context = retrieval['context']
    client = Client(plan, repeat)
    actor = LocalActor(plan['alf'])
    try:
        for name, memory in (('full', context), ('none', '')):
            target = output / name
            if (target / 'episode.json').exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            actor.run_many([dict(game=case['game'], memory=memory,
                                 seed=seed(repeat, case['game'], 2, 'actor'),
                                 output=target)])
    finally:
        actor.pool.shutdown(wait=True)


def audit(output: Path, require_complete: bool = True) -> dict:
    design = read(output / 'design.json')
    if (design['schema'] != 'alf_train_retry_context_v1' or
            design['actor_repeats'] != list(ACTOR_REPEATS)):
        raise ValueError('Unexpected frozen retry probe')
    source = Path(design['source'])
    root = Path(__file__).parents[2]
    if (sha(source / 'complete.json') != design['source_complete_sha256'] or
            sha(source / 'plan.json') != design['source_plan_sha256'] or
            sha(Path(__file__)) != design['runner_sha256'] or
            sha(output / 'source' / Path(__file__).name) != design['runner_sha256'] or
            targets(source) != design['cases']):
        raise ValueError('Frozen training source changed')
    for name, expected in design['helper_sha256'].items():
        if sha(root / name) != expected or sha(output / 'source' / Path(name).name) != expected:
            raise ValueError(f'Frozen execution helper changed: {name}')
    rows, missing = [], []
    for case in design['cases']:
        retrieval = read(source / 'runs' / case['case'] / 'retrieval_3.json')
        for repeat in ACTOR_REPEATS:
            folder = output / case['case'] / f'actor_repeat_{repeat}'
            paths = [folder / arm / 'episode.json' for arm in ('full', 'none')]
            if not all(p.exists() for p in paths):
                missing.append(dict(case=case['case'], actor_repeat=repeat))
                continue
            full, none = [read(p) for p in paths]
            expected_seed = seed(repeat, case['game'], 2, 'actor')
            if (full['status'] != 'complete' or none['status'] != 'complete' or
                    full['game'] != case['game'] or none['game'] != case['game'] or
                    full['seed'] != expected_seed or none['seed'] != expected_seed or
                    full['memory'] != retrieval['context'] or none['memory'] != '' or
                    full['reward'] not in (0, 1) or none['reward'] not in (0, 1)):
                raise ValueError(f'Unmatched retry replay: {folder}')
            rows.append(dict(case=case['case'], task=Path(case['case']).parts[1],
                             actor_repeat=repeat, input_sha256=case['input_sha256'],
                             retrieval_sha256=case['retrieval_sha256'],
                             full=float(full['reward']), none=float(none['reward']),
                             delta=float(full['reward'] - none['reward'])))
    if require_complete and missing:
        raise ValueError(f'Missing {len(missing)} paired retry replays')
    deltas = [r['delta'] for r in rows]
    return dict(expected=len(design['cases']) * len(ACTOR_REPEATS),
                completed=len(rows), missing=missing, train_split='train',
                rows=rows, mean_delta=statistics.fmean(deltas) if deltas else None,
                wins=sum(x > 0 for x in deltas), losses=sum(x < 0 for x in deltas),
                ties=sum(x == 0 for x in deltas),
                note='Full versus empty third-attempt context on fixed official train game and actor seed')


def run(output: Path) -> None:
    design = read(output / 'design.json')
    audit(output, require_complete=False)
    source = Path(design['source'])
    plan = read(source / 'plan.json')
    plan['url'] = design['url']
    plan['alf']['actor_url'] = design['url']
    plan['_source'] = str(source)
    for case in design['cases']:
        for repeat in ACTOR_REPEATS:
            target = output / case['case'] / f'actor_repeat_{repeat}'
            _run_case(plan, case, repeat, target)
            print(f'{case["case"]} seed={repeat}', flush=True)
    save(output / 'analysis.json', audit(output))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url')
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    if a.audit_only:
        result = audit(output)
        save(output / 'analysis.json', result)
        print(f'Audited {result["completed"]}/{result["expected"]} pairs')
        return
    if not output.exists():
        if not a.source or not a.url:
            p.error('Preparation requires --source and --url')
        prepare(a.source, output, a.url)
    run(output)


if __name__ == '__main__':
    main()
