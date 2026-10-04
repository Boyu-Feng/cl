"""Audit paired raw-evidence pilots for source bindings and time ordering."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .evidence_selected_memory import public_rows


def rows_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest()


def source_for_update(cell, benchmark, attempt):
    if benchmark == 'alfworld':
        from ttcl.alfworld_comparison.methods import public_episode
        episode = read(cell / f'attempt_{attempt}' / 'episode.json')
        public = public_episode(episode)
    else:
        public = read(cell / 'public_trajectory.json')
    return public


def audit(root):
    design = read(root / 'design.json')
    benchmark = design['benchmark']
    if design['schema'] not in {'online_raw_evidence_pair_v1',
                                'online_raw_evidence_pair_v2',
                                'online_raw_evidence_pair_v3'}:
        raise ValueError('Unknown evidence experiment')
    if sha(Path(design['origin']) / 'plan.json') != design['plan_sha256']:
        raise ValueError('Original benchmark plan changed')
    if design['source_design'] and sha(Path(design['source_design'])) != design['source_design_sha256']:
        raise ValueError('Source selection changed')
    for name in ('memory.py', 'worker.py', 'credit_online_ablation.py'):
        if not (root / 'source' / name).is_file():
            raise ValueError(f'Missing frozen source {name}')
    suffix = {'online_raw_evidence_pair_v1': '',
              'online_raw_evidence_pair_v2': '_v2',
              'online_raw_evidence_pair_v3': '_v3'}[design['schema']]
    if (sha(root / 'source' / f'evaluate_evidence_selected{suffix}.py') != design['runner_sha256'] or
            sha(root / 'source' / f'evidence_selected_memory{suffix}.py') != design['memory_sha256']):
        raise ValueError('Frozen runner or evidence memory source changed')
    rows = read(root / 'rows.json')
    if len(rows) != len(design['input_bindings']) * 2:
        raise ValueError('Incomplete paired rows')
    native, candidate = design['arms']
    verified_evidence = 0
    card_exposures = 0
    reward_deltas = []
    for arm in (native, candidate):
        known = {}
        for index, input_sha in enumerate(design['input_bindings'], 1):
            cell = root / arm / f'episode_{index:03d}'
            row = read(cell / 'row.json')
            if (row['status'] != 'complete' or
                    row['input_sha256' if benchmark == 'alfworld' else 'initial_query_sha256'] != input_sha or
                    sha(cell / 'memory_after.json') != row['memory_after_sha256']):
                raise ValueError(f'Unscored or changed cell: {cell}')
            summary = next(x for x in rows if x['index'] == index-1 and x['arm'] == arm)
            if summary['input_sha256'] != input_sha or summary['reward'] != row['reward']:
                raise ValueError('Paired score or input mismatch')
            retrievals = (sorted(cell.glob('retrieval_*.json')) if benchmark == 'alfworld'
                          else [cell / 'retrieval.json'])
            updates = (sorted(cell.glob('update_*.json')) if benchmark == 'alfworld'
                       else [cell / 'update.json'])
            if len(retrievals) != len(updates):
                raise ValueError('Retrieval/update count mismatch')
            for attempt, (ret_path, update_path) in enumerate(zip(retrievals, updates), 1):
                retrieval, update = read(ret_path), read(update_path)
                if arm == candidate:
                    for card in retrieval.get('evidence_cards', []):
                        source = known.get(card['memory_id'])
                        if source is None:
                            raise ValueError('Future or unknown evidence was retrieved')
                        if (card['source_input_sha256'] not in source['source_binding'].values() or
                                card['steps'] != [x['step'] for x in source['selected']]):
                            raise ValueError('Retrieved evidence source changed')
                        card_exposures += 1
                public = source_for_update(cell, benchmark, attempt)
                if update['input_binding']['public_content_sha256'] != rows_sha(public):
                    raise ValueError('Public trajectory hash changed')
                mid = update['new_memory_id']
                if arm == candidate:
                    evidence_path = root / arm / 'memory' / f'evidence_{mid}.json'
                    if evidence_path.is_file():
                        record = read(evidence_path)
                        step_rows = public_rows(public)
                        if (record['source_binding'] != update['input_binding'] or
                                record['rows_sha256'] != rows_sha(step_rows)):
                            raise ValueError('Evidence source binding changed')
                        for entry in record['selected']:
                            step = entry['step']
                            if (not 0 <= step < len(step_rows) or
                                entry['action'] != step_rows[step]['action'] or
                                entry['feedback'] != step_rows[step]['feedback']):
                                raise ValueError('Selected row differs from public source')
                            verified_evidence += 1
                        known[mid] = record
            if arm == candidate:
                snapshot = read(cell / 'memory_after.json')
                if snapshot['evidence'] != known:
                    raise ValueError('Evidence snapshot continuity changed')
        if arm == candidate:
            for index in range(len(design['input_bindings'])):
                base = next(x for x in rows if x['index'] == index and x['arm'] == native)
                changed = next(x for x in rows if x['index'] == index and x['arm'] == candidate)
                reward_deltas.append(changed['reward'] - base['reward'])
    return dict(schema='raw_evidence_pair_audit_v1', design_sha256=sha(root / 'design.json'),
                cases=len(design['input_bindings']), benchmark=benchmark,
                task=design['task'], repeat=design['repeat'],
                candidate_arm=candidate, reward_deltas=reward_deltas,
                selected_public_rows_verified=verified_evidence,
                card_exposures=card_exposures,
                caveat='Provenance/time audit; it does not prove causal efficacy or independently replay official task scoring')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    save(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
