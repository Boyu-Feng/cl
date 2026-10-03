"""Independently audit an evolving ALFWorld causal-credit cache chain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .causal_credit_cache import decision


def _entry(item: dict) -> str:
    metadata = item['metadata']
    return ('Task: ' + metadata['task_description'] +
            '\nExperience: ' + metadata['public_abstract'])


def audit(output: Path) -> dict:
    plan, design = read(output / 'plan.json'), read(output / 'design.json')
    source = Path(design['source'])
    source_report = audit_source(source)
    if (not source_report['complete'] or source_report['missing'] or
            sha(output / 'plan.json') != design['plan_sha256'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(source / 'complete.json') != design['source_complete_sha256'] or
            read(source / 'design.json')['selected_games'] !=
            design['selected_games'] or
            sha(Path(design['cache_path'])) != design['cache_sha256']):
        raise ValueError('Online cache source or plan changed')
    cache = read(Path(design['cache_path']))
    ttcl_root = Path(__file__).resolve().parents[1]
    for name, digest in design['source_sha256'].items():
        if (sha(ttcl_root / name) != digest or
                sha(output / 'source' / 'ttcl' / name) != digest):
            raise ValueError(f'Frozen online source changed: {name}')
    repeat = design['repeat']
    pairs, missing = [], []
    for family in design['families']:
        boot = design['bootstrap'][family]
        directory = (output / 'runs' / 'alfworld' / family /
                     str(repeat) / 'cache')
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
                raise ValueError('Non-train or changed ALFWorld game')
            target = directory / f'episode_{index:03d}'
            before_path = directory / f'episode_{index:03d}_memory_before.json'
            if not (target / 'row.json').exists():
                missing.append(dict(family=family, index=index))
                continue
            if sha(before_path) != previous:
                raise ValueError('Online cache chain did not use previous memory bank')
            before = read(before_path)
            row = read(target / 'row.json')
            if (row['status'] != 'complete' or row['task'] != family or
                    row['repeat'] != repeat or row['arm'] != 'cache' or
                    row['game'] != game or
                    row['input_sha256'] != item['sha256'] or
                    not 1 <= row['attempts'] <= plan['alf']['max_attempts']):
                raise ValueError('Invalid online-cache task row')
            rewards, actor_calls = [], 0
            first_suppressed = []
            for attempt in range(1, row['attempts'] + 1):
                retrieval = read(target / f'retrieval_{attempt}.json')
                episode = read(target / f'attempt_{attempt}' / 'episode.json')
                if (episode['status'] != 'complete' or episode['game'] != game or
                        episode['seed'] !=
                        seed(repeat, game, attempt - 1, 'actor') or
                        episode['memory'] != retrieval['context'] or
                        hashlib.sha256(retrieval['context'].encode()).hexdigest()
                        != retrieval['context_sha256']):
                    raise ValueError('Online actor seed or memory context changed')
                if (retrieval['credit_cache_decision']['evaluated'] !=
                        (attempt == 1) or
                        len(retrieval['suppressed_ids']) > 1):
                    raise ValueError('Credit cache applied outside first attempt')
                if attempt == 1:
                    native_ids = retrieval['native_ids']
                    if len(native_ids) != len(set(native_ids)):
                        raise ValueError('Duplicate native retrieval ID')
                    texts = {mid:_entry(before['items'][mid])
                             for mid in native_ids}
                    native_context = '\n\n'.join(texts[mid]
                                                  for mid in native_ids)
                    if hashlib.sha256(native_context.encode()).hexdigest() \
                            != retrieval['native_context_sha256']:
                        raise ValueError('Native retrieval text differs from root bank')
                    hashes = {mid:hashlib.sha256(text.encode()).hexdigest()
                              for mid, text in texts.items()}
                    chosen = decision(dict(task=family, ids=native_ids,
                                           memory_text_sha256=hashes), cache)
                    expected_drop = ([chosen['drop_memory_id']]
                                     if chosen['drop_memory_id'] else [])
                    if (retrieval['suppressed_ids'] != expected_drop or
                            retrieval['credit_cache_decision']['drop_position'] !=
                            chosen['drop_position'] or
                            retrieval['credit_cache_decision']['drop_memory_id'] !=
                            chosen['drop_memory_id'] or
                            retrieval['credit_cache_decision']['donor_case'] !=
                            (chosen['donor']['source_case']
                             if chosen['donor'] else None)):
                        raise ValueError('Cache decision differs from frozen policy')
                    first_suppressed = expected_drop
                    kept = [mid for mid in native_ids if mid not in expected_drop]
                    if (retrieval['ids'] != kept or
                            retrieval['context'] !=
                            '\n\n'.join(texts[mid] for mid in kept)):
                        raise ValueError('Cache edited more than one memory entry')
                elif (retrieval['suppressed_ids'] or
                      retrieval['ids'] != retrieval['native_ids'] or
                      retrieval['context_sha256'] !=
                      retrieval['native_context_sha256']):
                    raise ValueError('Later attempt is not native retrieval')
                update = read(target / f'update_{attempt}.json')
                if (set(update['q_updates']) != set(retrieval['ids']) or
                        update['input_binding']['game_sha256'] !=
                        item['sha256'] or
                        update['input_binding']['attempt'] != attempt - 1 or
                        update['input_binding']['repeat'] != repeat):
                    raise ValueError('Online Q/writer provenance changed')
                rewards.append(episode['reward'])
                actor_calls += len(episode['generations'])
            if (row['first_attempt'] != rewards[0] or
                    row['within_three'] != max(rewards) or
                    row['reward'] != max(rewards) or
                    row['actor_calls'] != actor_calls or
                    sha(target / 'memory_after.json') !=
                    row['memory_after_sha256']):
                raise ValueError('Online official reward or snapshot changed')
            previous = row['memory_after_sha256']
            native = read(source / 'runs' / 'alfworld' / family /
                          str(repeat) / 'memrl' /
                          f'episode_{index:03d}' / 'row.json')
            pairs.append(dict(family=family, index=index,
                              game_sha256=item['sha256'],
                              suppressed_ids=first_suppressed,
                              native_first=native['first_attempt'],
                              cache_first=row['first_attempt'],
                              native_within_three=native['within_three'],
                              cache_within_three=row['within_three'],
                              native_actor_calls=native['actor_calls'],
                              cache_actor_calls=row['actor_calls']))
    expected = sum(len(items) for items in design['selected_games'].values())
    complete_path = output / 'complete.json'
    if complete_path.exists():
        complete = read(complete_path)
        if (missing or complete['completed'] != expected or
                complete['expected'] != expected or
                complete['design_sha256'] != sha(output / 'design.json')):
            raise ValueError('Completed online cache has missing cells')
    return dict(schema='causal_credit_cache_online_audit_v1',
                design_sha256=sha(output / 'design.json'),
                cache_sha256=design['cache_sha256'],
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
