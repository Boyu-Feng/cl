"""Audit a post-hoc native actor/writer trace replay of paired-Q ALF cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.experience_evolution.core import seed
from ttcl.icl_mem0_comparison.protocol import read, save, sha

from .audit_credit_train_extension import audit as audit_source
from .audit_paired_marginal_q import audit as audit_paired
from .paired_marginal_q import paired_advantage


TIME_FIELDS = frozenset({'q_updated_at', 'last_used_at', 'memory_time',
                         'updated_at'})


def _semantic(value):
    if isinstance(value, dict):
        return {key:_semantic(item) for key,item in value.items()
                if key not in TIME_FIELDS}
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    return value


def _public_actor(episode: dict) -> dict:
    return dict(game=episode['game'], memory=episode['memory'],
                seed=episode['seed'], initial_observation=
                episode['initial_observation'], reward=episode['reward'],
                actions=[step['action'] for step in episode['trajectory']],
                observations=[step['observation'] for step in
                              episode['trajectory']],
                prompts=[item['rendered_prompt_sha256'] for item in
                         episode['generations']],
                texts=[item['text'] for item in episode['generations']])


def audit(output: Path) -> dict:
    design = read(output / 'design.json')
    source, paired = Path(design['source']), Path(design['paired'])
    source_report, paired_report = audit_source(source), audit_paired(paired)
    if (not source_report['complete'] or not paired_report['complete'] or
            source_report['missing'] or paired_report['missing'] or
            sha(source / 'design.json') != design['source_design_sha256'] or
            sha(paired / 'design.json') != design['paired_design_sha256']):
        raise ValueError('Replay source lineage changed')
    for name, expected in design['source_sha256'].items():
        if sha(output / 'source' / 'ttcl' / name) != expected:
            raise ValueError(f'Frozen diagnostic source changed: {name}')
    source_design = read(source / 'design.json')
    family, repeat = design['family'], design['repeat']
    if (family not in source_design['families'] or
            repeat != source_design['repeat'] or
            design['selected_games'] != source_design['selected_games'][family]):
        raise ValueError('Replay selected games changed')
    boot = source_design['bootstrap'][family]
    native_dir = source / 'runs' / 'alfworld' / family / str(repeat) / 'memrl'
    root = native_dir / f"episode_{boot['index']:03d}" / 'memory_after.json'
    writer_log = native_dir / 'memory' / 'writer.jsonl'
    if (sha(root) != design['root_sha256'] or
            sha(writer_log) != design['writer_log_sha256'] or
            (output / 'failure.json').exists() or
            (output / 'stopped.json').exists()):
        raise ValueError('Replay root, writer log or completion changed')
    report = read(output / 'report.json')
    if (report['status'] != 'complete' or
            report['completed'] != len(design['selected_games']) or
            report['design_sha256'] != sha(output / 'design.json') or
            report['actor_collisions']):
        raise ValueError('Replay report is incomplete or ambiguous')
    previous = design['root_sha256']
    pairs = []
    for offset, item in enumerate(design['selected_games'], 1):
        index = boot['index'] + offset
        target = output / f'episode_{index:03d}'
        native = native_dir / f'episode_{index:03d}'
        if (sha(output / f'episode_{index:03d}_memory_before.json') !=
                previous or
                sha(Path(read(source / 'plan.json')['alf']['data_root']) /
                    item['path']) != item['sha256']):
            raise ValueError('Replay bank continuity or train input changed')
        row, native_row = read(target / 'row.json'), read(native / 'row.json')
        if (row['status'] != 'complete' or row['game'] != item['path'] or
                row['input_sha256'] != item['sha256'] or
                row['repeat'] != repeat or row['arm'] != 'paired_q'):
            raise ValueError('Replay official task row changed')
        rewards, calls, native_actor_equal = [], 0, []
        for attempt in range(1, row['attempts'] + 1):
            retrieval = read(target / f'retrieval_{attempt}.json')
            episode = read(target / f'attempt_{attempt}' / 'episode.json')
            if (episode['status'] != 'complete' or
                    episode['game'] != item['path'] or
                    episode['seed'] != seed(repeat, item['path'],
                                            attempt - 1, 'actor') or
                    episode['memory'] != retrieval['context']):
                raise ValueError('Replay task, actor seed or context changed')
            update = read(target / f'update_{attempt}.json')
            if (set(update['q_updates']) != set(retrieval['ids']) or
                    update['input_binding']['game_sha256'] != item['sha256']):
                raise ValueError('Replay Q or writer binding changed')
            native_path = native / f'attempt_{attempt}' / 'episode.json'
            native_actor_equal.append(native_path.exists() and
                                      _public_actor(episode) ==
                                      _public_actor(read(native_path)))
            rewards.append(episode['reward'])
            calls += len(episode['generations'])
        if (row['first_attempt'] != rewards[0] or
                row['within_three'] != max(rewards) or
                row['actor_calls'] != calls or
                sha(target / 'memory_after.json') !=
                row['memory_after_sha256']):
            raise ValueError('Replay official reward, call count or bank changed')
        first_retrieval = read(target / 'retrieval_1.json')
        credit = read(target / 'counterfactual_1.json')
        full = read(target / 'attempt_1' / 'episode.json')
        drop = read(target / 'counterfactual_1' / 'episode.json')
        if (credit['selected_memory_id'] != first_retrieval['ids'][0] or
                any(credit[key] != value for key,value in
                    paired_advantage(full, drop).items())):
            raise ValueError('Replay first-memory counterfactual changed')
        replay_bank = read(target / 'memory_after.json')
        native_bank = read(native / 'memory_after.json')
        semantic_match = _semantic(replay_bank) == _semantic(native_bank)
        pairs.append(dict(index=index, game_sha256=item['sha256'],
                          native_first=native_row['first_attempt'],
                          replay_first=row['first_attempt'],
                          native_within_three=native_row['within_three'],
                          replay_within_three=row['within_three'],
                          native_actor_equal=native_actor_equal,
                          semantic_bank_equal=semantic_match,
                          q_advantage=credit['q_advantage']))
        previous = row['memory_after_sha256']
    if len(report['rows']) != len(pairs):
        raise ValueError('Replay report rows differ from audited cells')
    return dict(schema='paired_q_trace_replay_audit_v1',
                design_sha256=sha(output / 'design.json'),
                source_design_sha256=design['source_design_sha256'],
                paired_design_sha256=design['paired_design_sha256'],
                complete=True, audited=len(pairs),
                actor_replay_hits=sum(hit['kind'] == 'actor' for hit in
                                      report['writer_replay_hits']),
                writer_replay_hits=sum(hit['kind'] == 'writer' for hit in
                                       report['writer_replay_hits']),
                actor_fallbacks=sum(hit['kind'] == 'actor' for hit in
                                    report['writer_fallbacks']),
                writer_fallbacks=sum(hit['kind'] == 'writer' for hit in
                                     report['writer_fallbacks']),
                pairs=pairs,
                caveat='Post-hoc matched trace replay for mechanism only; not an independent performance estimate')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit(args.output.resolve())
    if args.report:
        save(args.report.resolve(), report)
    print(json.dumps({key:value for key,value in report.items()
                      if key != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
