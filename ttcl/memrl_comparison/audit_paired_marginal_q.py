"""Audit official-train paired marginal-Q chain and its read-only replays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .paired_marginal_q import paired_advantage


def _entry(item: dict) -> str:
    md = item['metadata']
    return 'Task: ' + md['task_description'] + '\nExperience: ' + md['public_abstract']


def audit(output: Path) -> dict:
    plan, design = read(output / 'plan.json'), read(output / 'design.json')
    source = Path(design['source'])
    source_report = audit_source(source)
    source_design = read(source / 'design.json')
    if (not source_report['complete'] or source_report['missing'] or
            sha(output / 'plan.json') != design['plan_sha256'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source / 'complete.json') != design['source_complete_sha256'] or
            source_design['selected_games'] != design['selected_games'] or
            source_design['repeat'] != design['repeat'] or
            plan['paired_marginal_q']['cost_weight'] != 0.1 or
            plan['paired_marginal_q']['call_budget'] != 50):
        raise ValueError('Paired-Q source, plan or rule changed')
    ttcl_root = Path(__file__).resolve().parents[1]
    for name, expected in design['source_sha256'].items():
        if (sha(ttcl_root / name) != expected or
                sha(output / 'source' / 'ttcl' / name) != expected):
            raise ValueError(f'Frozen source changed: {name}')
    repeat = design['repeat']
    pairs, missing = [], []
    for family in design['families']:
        boot = design['bootstrap'][family]
        if boot['sha256'] != source_design['bootstrap'][family]['sha256']:
            raise ValueError('Bootstrap differs from native source')
        directory = output / 'runs' / 'alfworld' / family / str(repeat) / 'paired_q'
        bootstrap = directory / f"episode_{boot['index']:03d}"
        if (sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(bootstrap / 'memory_after.json') != boot['sha256'] or
                read(bootstrap / 'bootstrap_lineage.json') != boot):
            raise ValueError('Paired-Q root bank changed')
        previous = boot['sha256']
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            game = item['path']
            if (not game.startswith('json_2.1.1/train/') or
                    sha(Path(plan['alf']['data_root']) / game) != item['sha256']):
                raise ValueError('Invalid official train input')
            target = directory / f'episode_{index:03d}'
            before_path = directory / f'episode_{index:03d}_memory_before.json'
            if not (target / 'row.json').exists():
                missing.append(dict(family=family, index=index))
                continue
            if sha(before_path) != previous:
                raise ValueError('Paired-Q chain did not use previous memory bank')
            before = read(before_path)
            row = read(target / 'row.json')
            if (row['status'] != 'complete' or row['task'] != family or
                    row['repeat'] != repeat or row['arm'] != 'paired_q' or
                    row['game'] != game or row['input_sha256'] != item['sha256'] or
                    not 1 <= row['attempts'] <= plan['alf']['max_attempts']):
                raise ValueError('Invalid paired-Q task row')
            rewards, actor_calls, extra_calls = [], 0, 0
            selected, advantage = None, None
            for attempt in range(1, row['attempts'] + 1):
                retrieval = read(target / f'retrieval_{attempt}.json')
                episode_path = target / f'attempt_{attempt}' / 'episode.json'
                episode = read(episode_path)
                expected_seed = seed(repeat, game, attempt - 1, 'actor')
                if (episode['status'] != 'complete' or episode['game'] != game or
                        episode['seed'] != expected_seed or
                        episode['memory'] != retrieval['context'] or
                        hashlib.sha256(retrieval['context'].encode()).hexdigest()
                        != retrieval['context_sha256']):
                    raise ValueError('Paired-Q actor seed or context changed')
                if len(retrieval['ids']) != len(set(retrieval['ids'])):
                    raise ValueError('Duplicate retrieval ID')
                if attempt == 1:
                    texts = {mid: _entry(before['items'][mid])
                             for mid in retrieval['ids']}
                    if retrieval['context'] != '\n\n'.join(
                            texts[mid] for mid in retrieval['ids']):
                        raise ValueError('First retrieval differs from root bank')
                if attempt == 1 and retrieval['ids']:
                    selected = retrieval['ids'][0]
                    drop_context = '\n\n'.join(
                        texts[mid] for mid in retrieval['ids'][1:])
                    drop_path = target / 'counterfactual_1' / 'episode.json'
                    drop = read(drop_path)
                    credit = read(target / 'counterfactual_1.json')
                    advantage = paired_advantage(episode, drop)
                    expected_credit = dict(
                        selected_memory_id=selected,
                        selected_memory_text_sha256=hashlib.sha256(
                            texts[selected].encode()).hexdigest(),
                        full_context_sha256=hashlib.sha256(
                            retrieval['context'].encode()).hexdigest(),
                        drop_context_sha256=hashlib.sha256(
                            drop_context.encode()).hexdigest(),
                        full_episode_sha256=sha(episode_path),
                        drop_episode_sha256=sha(drop_path),
                        actor_seed=expected_seed, **advantage)
                    if (drop['memory'] != drop_context or
                            drop['initial_observation'] !=
                            episode['initial_observation'] or
                            credit != expected_credit):
                        raise ValueError('Counterfactual arm or paired Q credit changed')
                    extra_calls = len(drop['generations'])
                update = read(target / f'update_{attempt}.json')
                expected_rewards = {mid: (advantage['q_advantage']
                                          if attempt == 1 and mid == selected
                                          else 1. if episode['reward'] else -1.)
                                    for mid in retrieval['ids']}
                if (set(update['q_updates']) != set(retrieval['ids']) or
                        update['q_rewards'] != expected_rewards or
                        update['causal_memory_id'] !=
                        (selected if attempt == 1 else None) or
                        update['input_binding']['game_sha256'] != item['sha256'] or
                        update['input_binding']['attempt'] != attempt - 1 or
                        update['input_binding']['repeat'] != repeat):
                    raise ValueError('Paired-Q update or writer provenance changed')
                rewards.append(episode['reward'])
                actor_calls += len(episode['generations'])
            if (row['first_attempt'] != rewards[0] or
                    row['within_three'] != max(rewards) or
                    row['reward'] != max(rewards) or
                    row['actor_calls'] != actor_calls or
                    row['counterfactual_actor_calls'] != extra_calls or
                    row['total_actor_calls'] != actor_calls + extra_calls or
                    sha(target / 'memory_after.json') != row['memory_after_sha256']):
                raise ValueError('Paired-Q official reward, cost or bank changed')
            previous = row['memory_after_sha256']
            native = read(source / 'runs' / 'alfworld' / family / str(repeat) /
                          'memrl' / f'episode_{index:03d}' / 'row.json')
            pairs.append(dict(family=family, index=index,
                              game_sha256=item['sha256'], selected_id=selected,
                              q_advantage=advantage['q_advantage']
                              if advantage else None,
                              native_first=native['first_attempt'],
                              paired_first=row['first_attempt'],
                              native_within_three=native['within_three'],
                              paired_within_three=row['within_three'],
                              native_actor_calls=native['actor_calls'],
                              paired_actor_calls=actor_calls,
                              counterfactual_actor_calls=extra_calls))
    expected = sum(len(items) for items in design['selected_games'].values())
    complete_path = output / 'complete.json'
    if complete_path.exists():
        complete = read(complete_path)
        if (missing or complete['completed'] != expected or
                complete['expected'] != expected or
                complete['design_sha256'] != sha(output / 'design.json')):
            raise ValueError('Completed paired-Q chain has missing cells')
    return dict(schema='paired_marginal_q_online_audit_v1',
                design_sha256=sha(output / 'design.json'),
                source_design_sha256=design['source_design_sha256'],
                complete=complete_path.exists(), expected=expected,
                audited=len(pairs), missing=missing, pairs=pairs,
                caveat='Official train online chain; one repeat and three games per family, not valid_unseen')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit(args.output.resolve())
    if args.report:
        save(args.report.resolve(), report)
    print(json.dumps({key:value for key, value in report.items()
                      if key != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
