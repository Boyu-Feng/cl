"""Independently audit a continued native MemRL official-train source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_source import audit as audit_prior
from .credit_probe import memory_arms


def audit(output: Path) -> dict:
    plan, design = read(output / 'plan.json'), read(output / 'design.json')
    prior = Path(design['prior'])
    prior_report = audit_prior(prior)
    if (not prior_report['complete'] or prior_report['missing'] or
            sha(output / 'plan.json') != design['plan_sha256'] or
            sha(prior / 'design.json') !=
            plan['credit_training_extension']['prior_design_sha256'] or
            sha(prior / 'complete.json') !=
            plan['credit_training_extension']['prior_complete_sha256'] or
            sha(Path(design['training_plan'])) != design['training_plan_sha256']):
        raise ValueError('Train extension lineage changed')
    ttcl_root = Path(__file__).resolve().parents[1]
    for name, expected in design['source_sha256'].items():
        if sha(output / 'source' / 'ttcl' / name) != expected or \
                sha(ttcl_root / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    old_design = read(prior / 'design.json')
    old_hashes = {item['sha256'] for values in old_design['selected_games'].values()
                  for item in values}
    seen = set(old_hashes)
    paired, missing = [], []
    repeat = design['repeat']
    for family in design['families']:
        boot = design['bootstrap'][family]
        if (boot['index'] != len(old_design['selected_games'][family]) or
                sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(Path(boot['snapshot']).parent / 'row.json') !=
                boot['prior_row_sha256']):
            raise ValueError(f'Previous bank changed: {family}')
        directory = output / 'runs' / 'alfworld' / family / str(repeat)
        bootstrap = directory / 'memrl' / f"episode_{boot['index']:03d}"
        if (sha(bootstrap / 'memory_after.json') != boot['sha256'] or
                read(bootstrap / 'bootstrap_lineage.json') != boot):
            raise ValueError(f'Copied bootstrap bank changed: {family}')
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            game = item['path']
            if (not game.startswith('json_2.1.1/train/') or
                    item['sha256'] in seen or
                    sha(Path(plan['alf']['data_root']) / game) != item['sha256']):
                raise ValueError(f'Non-train, duplicate or changed game: {game}')
            seen.add(item['sha256'])
            rows = {}
            for arm in ('none', 'memrl'):
                episode_dir = directory / arm / f'episode_{index:03d}'
                if not (episode_dir / 'row.json').exists():
                    missing.append(dict(family=family, index=index, arm=arm))
                    continue
                row = read(episode_dir / 'row.json')
                if (row['status'] != 'complete' or row['game'] != game or
                        row['input_sha256'] != item['sha256'] or
                        row['task'] != family or row['repeat'] != repeat or
                        row['arm'] != arm or not 1 <= row['attempts'] <=
                        plan['alf']['max_attempts']):
                    raise ValueError(f'Invalid completed row: {episode_dir}')
                rewards, actor_calls = [], 0
                for attempt in range(1, row['attempts'] + 1):
                    actual = read(episode_dir / f'attempt_{attempt}' / 'episode.json')
                    retrieval = read(episode_dir / f'retrieval_{attempt}.json')
                    if (actual['status'] != 'complete' or actual['game'] != game or
                            actual['seed'] != seed(repeat, game, attempt - 1, 'actor') or
                            actual['memory'] != retrieval['context']):
                        raise ValueError(f'Attempt or context changed: {episode_dir}')
                    actor_calls += len(actual['generations'])
                    rewards.append(actual['reward'])
                    if arm == 'none':
                        if retrieval['ids'] or retrieval['context']:
                            raise ValueError('No-memory arm exposed experience')
                    else:
                        if (hashlib.sha256(retrieval['context'].encode()).hexdigest()
                                != retrieval['context_sha256']):
                            raise ValueError('Retrieved context hash changed')
                        update = read(episode_dir / f'update_{attempt}.json')
                        if (set(update['q_updates']) != set(retrieval['ids']) or
                                update['input_binding']['game_sha256'] != item['sha256'] or
                                update['input_binding']['attempt'] != attempt - 1 or
                                update['input_binding']['repeat'] != repeat):
                            raise ValueError('Native Q or writer provenance changed')
                if (actor_calls != row['actor_calls'] or
                        row['first_attempt'] != rewards[0] or
                        row['within_three'] != max(rewards) or
                        row['reward'] != max(rewards)):
                    raise ValueError(f'Official reward or actor calls changed: {episode_dir}')
                if arm == 'memrl':
                    if sha(episode_dir / 'memory_after.json') != row['memory_after_sha256']:
                        raise ValueError('Memory after snapshot changed')
                    case = f'alfworld/{family}/{repeat}/memrl/episode_{index:03d}'
                    if read(episode_dir / 'retrieval_1.json')['ids']:
                        spec, _ = memory_arms(output, case)
                        if spec['source_input_sha256'] != item['sha256']:
                            raise ValueError('First retrieval source binding changed')
                rows[arm] = row
            if len(rows) == 2:
                paired.append(dict(family=family, index=index,
                                   game_sha256=item['sha256'],
                                   none_first=rows['none']['first_attempt'],
                                   memrl_first=rows['memrl']['first_attempt'],
                                   none_within_three=rows['none']['within_three'],
                                   memrl_within_three=rows['memrl']['within_three']))
    expected = sum(len(items) for items in design['selected_games'].values())
    complete_path = output / 'complete.json'
    if complete_path.exists():
        completed = read(complete_path)
        if (missing or completed['completed'] != expected * 2 or
                completed['expected'] != expected * 2 or
                completed['design_sha256'] != sha(output / 'design.json')):
            raise ValueError('Completed extension has missing or changed cells')
    return dict(design_sha256=sha(output / 'design.json'),
                complete=complete_path.exists(), expected_pairs=expected,
                audited_pairs=len(paired), missing=missing,
                pairs=paired,
                caveat='Official train source continuation; rewards are feedback, not reviewed annotation targets')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    if args.report:
        save(args.report.resolve(), result)
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
