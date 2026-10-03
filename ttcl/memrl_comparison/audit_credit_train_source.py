"""Independently audit completed native MemRL train-source ALFWorld pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha


def audit(source: Path) -> dict:
    plan, design = read(source / 'plan.json'), read(source / 'design.json')
    if sha(source / 'plan.json') != design['plan_sha256']:
        raise ValueError('Train-source plan changed')
    if sha(Path(design['origin']) / 'plan.json') != plan['credit_training_source']['original_plan_sha256']:
        raise ValueError('Original MemRL plan changed')
    if sha(Path(design['training_plan'])) != plan['credit_training_source']['training_plan_sha256']:
        raise ValueError('Historical ALF train selection changed')
    for name, expected in design['source_sha256'].items():
        if sha(source / 'source' / 'ttcl' / name) != expected:
            raise ValueError(f'Frozen collector source changed: {name}')
    pairs, missing = [], []
    for family in design['families']:
        for index, item in enumerate(design['selected_games'][family], 1):
            game = item['path']
            if not game.startswith('json_2.1.1/train/'):
                raise ValueError(f'Non-train game: {game}')
            if sha(Path(plan['alf']['data_root']) / game) != item['sha256']:
                raise ValueError(f'Train game content changed: {game}')
            root = source / 'runs' / 'alfworld' / family / str(design['repeat'])
            rows = {}
            for arm in ('none', 'memrl'):
                episode = root / arm / f'episode_{index:03d}'
                if not (episode / 'row.json').exists():
                    missing.append(dict(family=family, index=index, arm=arm))
                    continue
                row = read(episode / 'row.json')
                if (row['status'] != 'complete' or row['game'] != game or
                        row['input_sha256'] != item['sha256'] or
                        row['task'] != family or row['repeat'] != design['repeat'] or
                        row['arm'] != arm):
                    raise ValueError(f'Invalid source row: {episode}')
                rewards = []
                for attempt in range(1, row['attempts'] + 1):
                    actual = read(episode / f'attempt_{attempt}' / 'episode.json')
                    retrieval = read(episode / f'retrieval_{attempt}.json')
                    if (actual['game'] != game or
                            actual['seed'] != seed(design['repeat'], game, attempt - 1, 'actor') or
                            actual['memory'] != retrieval['context']):
                        raise ValueError(f'Attempt or memory differs: {episode} attempt {attempt}')
                    if arm == 'none' and (retrieval['ids'] or retrieval['context']):
                        raise ValueError(f'No-memory arm has context: {episode}')
                    if arm == 'memrl':
                        context_sha = hashlib.sha256(retrieval['context'].encode()).hexdigest()
                        if context_sha != retrieval['context_sha256']:
                            raise ValueError(f'Retrieval context hash differs: {episode}')
                        update = read(episode / f'update_{attempt}.json')
                        if set(update['q_updates']) != set(retrieval['ids']):
                            raise ValueError(f'Q update ids differ from retrieval: {episode}')
                        binding = update['input_binding']
                        if (binding['game_sha256'] != item['sha256'] or
                                binding['attempt'] != attempt - 1 or
                                binding['repeat'] != design['repeat']):
                            raise ValueError(f'Q update input binding differs: {episode}')
                    rewards.append(actual['reward'])
                if (not rewards or row['first_attempt'] != rewards[0] or
                        row['within_three'] != max(rewards) or row['reward'] != max(rewards)):
                    raise ValueError(f'Official ALF reward differs: {episode}')
                if arm == 'memrl' and sha(episode / 'memory_after.json') != row['memory_after_sha256']:
                    raise ValueError(f'Memory snapshot changed: {episode}')
                rows[arm] = row
            if len(rows) == 2:
                pairs.append(dict(family=family, index=index, game_sha256=item['sha256'],
                                  none=rows['none']['first_attempt'],
                                  memrl=rows['memrl']['first_attempt']))
    complete_path = source / 'complete.json'
    if complete_path.exists():
        complete = read(complete_path)
        if (missing or complete['completed'] != complete['expected'] or
                complete['design_sha256'] != sha(source / 'design.json')):
            raise ValueError('Completed train source has missing or changed cells')
    return dict(design_sha256=sha(source / 'design.json'),
                complete=complete_path.exists(), audited_pairs=len(pairs),
                expected_pairs=sum(len(items) for items in design['selected_games'].values()),
                missing=missing,
                first_attempt_none=sum(x['none'] for x in pairs),
                first_attempt_memrl=sum(x['memrl'] for x in pairs),
                pairs=pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.source.resolve())
    if args.output:
        save(args.output.resolve(), report)
    print(json.dumps({key: (len(value) if key == 'missing' else value)
                      for key, value in report.items() if key != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
