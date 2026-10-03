"""Audit an evolving ALFWorld source-success-filter chain against native MemRL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source


def _entry(item: dict) -> str:
    md = item['metadata']
    return 'Task: ' + md['task_description'] + '\nExperience: ' + md['public_abstract']


def audit(output: Path) -> dict:
    plan, design = read(output / 'plan.json'), read(output / 'design.json')
    if design['schema'] != 'alf_source_success_online_v1':
        raise ValueError('Wrong online experiment schema')
    source, validation = Path(design['source']), Path(design['validation'])
    source_report = audit_source(source)
    if (not source_report['complete'] or source_report['missing'] or
            sha(output / 'plan.json') != design['plan_sha256'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source / 'complete.json') != design['source_complete_sha256'] or
            sha(validation / 'design.json') != design['validation_design_sha256'] or
            sha(validation / 'analysis.json') != design['validation_analysis_sha256'] or
            read(source / 'design.json')['selected_games'] != design['selected_games']):
        raise ValueError('Online source, rule validation, or plan changed')
    root = Path(__file__).resolve().parents[1]
    for name, expected in design['source_sha256'].items():
        if (sha(root / name) != expected or
                sha(output / 'source' / 'ttcl' / name) != expected):
            raise ValueError(f'Frozen online source changed: {name}')
    repeat = design['repeat']
    pairs, missing = [], []
    for family in design['families']:
        boot = design['bootstrap'][family]
        directory = output / 'runs' / 'alfworld' / family / str(repeat) / 'success_filter'
        bootstrap = directory / f"episode_{boot['index']:03d}"
        if (sha(Path(boot['snapshot'])) != boot['sha256'] or
                sha(bootstrap / 'memory_after.json') != boot['sha256'] or
                read(bootstrap / 'bootstrap_lineage.json') != boot):
            raise ValueError('Online root memory bank changed')
        previous = boot['sha256']
        for offset, item in enumerate(design['selected_games'][family], 1):
            index = boot['index'] + offset
            game = item['path']
            if (not game.startswith('json_2.1.1/train/') or
                    sha(Path(plan['alf']['data_root']) / game) != item['sha256']):
                raise ValueError('Non-train or changed ALFWorld input')
            target = directory / f'episode_{index:03d}'
            before_path = directory / f'episode_{index:03d}_memory_before.json'
            if not (target / 'row.json').exists():
                missing.append(dict(family=family, index=index))
                continue
            if sha(before_path) != previous:
                raise ValueError('Online chain did not use previous memory bank')
            before = read(before_path)
            row = read(target / 'row.json')
            after_path = target / 'memory_after.json'
            if (row['status'] != 'complete' or row['task'] != family or
                    row['repeat'] != repeat or row['arm'] != 'success_filter' or
                    row['game'] != game or row['input_sha256'] != item['sha256'] or
                    not 1 <= row['attempts'] <= plan['alf']['max_attempts'] or
                    sha(after_path) != row['memory_after_sha256']):
                raise ValueError('Invalid completed success-filter task row')
            after = read(after_path)
            rewards, actor_calls, first_suppressed = [], 0, []
            for attempt in range(1, row['attempts'] + 1):
                retrieval = read(target / f'retrieval_{attempt}.json')
                episode = read(target / f'attempt_{attempt}' / 'episode.json')
                if (episode['status'] != 'complete' or episode['game'] != game or
                        episode['seed'] != seed(repeat, game, attempt - 1, 'actor') or
                        episode['memory'] != retrieval['context'] or
                        hashlib.sha256(retrieval['context'].encode()).hexdigest() !=
                        retrieval['context_sha256']):
                    raise ValueError('Actor seed or experience context changed')
                native_ids = retrieval['native_ids']
                if len(native_ids) != len(set(native_ids)):
                    raise ValueError('Duplicate native retrieval ID')
                bank = before if attempt == 1 else after
                texts = {mid:_entry(bank['items'][mid]) for mid in native_ids}
                source_success = {mid:bank['items'][mid]['metadata'].get('success') is True
                                  for mid in native_ids}
                native_context = '\n\n'.join(texts[mid] for mid in native_ids)
                selected = [mid for mid in native_ids if source_success[mid]]
                active = attempt == 1 and len(native_ids) == 3 and 0 < len(selected) < 3
                kept = selected if active else native_ids
                suppressed = [mid for mid in native_ids if mid not in kept]
                if (retrieval['source_success'] != source_success or
                        retrieval['native_context_sha256'] !=
                        hashlib.sha256(native_context.encode()).hexdigest() or
                        retrieval['success_filter_decision'] !=
                        dict(evaluated=attempt == 1, active=active, kept_ids=kept) or
                        retrieval['ids'] != kept or
                        retrieval['suppressed_ids'] != suppressed or
                        retrieval['context'] != '\n\n'.join(texts[mid] for mid in kept)):
                    raise ValueError('Filter decision differs from source metadata')
                if attempt == 1:
                    first_suppressed = suppressed
                update = read(target / f'update_{attempt}.json')
                if (set(update['q_updates']) != set(kept) or
                        update['input_binding']['game_sha256'] != item['sha256'] or
                        update['input_binding']['attempt'] != attempt - 1 or
                        update['input_binding']['repeat'] != repeat):
                    raise ValueError('Online Q or writer provenance changed')
                rewards.append(episode['reward'])
                actor_calls += len(episode['generations'])
            if (row['first_attempt'] != rewards[0] or
                    row['within_three'] != max(rewards) or
                    row['reward'] != max(rewards) or
                    row['actor_calls'] != actor_calls):
                raise ValueError('Official reward or actor calls changed')
            previous = row['memory_after_sha256']
            native = read(source / 'runs' / 'alfworld' / family / str(repeat) /
                          'memrl' / f'episode_{index:03d}' / 'row.json')
            pairs.append(dict(family=family, index=index, game_sha256=item['sha256'],
                              suppressed_ids=first_suppressed,
                              native_first=native['first_attempt'],
                              candidate_first=row['first_attempt'],
                              native_within_three=native['within_three'],
                              candidate_within_three=row['within_three'],
                              native_actor_calls=native['actor_calls'],
                              candidate_actor_calls=row['actor_calls']))
    expected = sum(len(items) for items in design['selected_games'].values())
    complete_path = output / 'complete.json'
    if complete_path.exists():
        complete = read(complete_path)
        if (missing or complete['completed'] != expected or
                complete['expected'] != expected or
                complete['design_sha256'] != sha(output / 'design.json')):
            raise ValueError('Completed online chain has missing cells')
    return dict(schema='alf_source_success_online_audit_v1',
                design_sha256=sha(output / 'design.json'),
                complete=complete_path.exists(), expected=expected,
                audited=len(pairs), missing=missing, pairs=pairs,
                caveat='Development official train chain; one repeat and three games per family, not valid_unseen')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit(args.output.resolve())
    if args.report:
        save(args.report.resolve(), report)
    print(json.dumps({k:v for k,v in report.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
