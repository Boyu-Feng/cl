"""Descriptive comparison of complete, audited v63/v65 own-policy streams.

Does not treat temporally dependent tasks as independent statistical trials.
Requires equal task order and rollout budgets, not equal realized training data.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_checked(path):
    raw = path.read_bytes()
    run = json.loads(raw)
    if not run.get('complete') or run['failures'] or len(run['rows']) != 240:
        raise ValueError(f'Complete failure-free 240-task run required: {path}')
    audit_path = path.with_suffix('.audit.json')
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if not (audit['passed'] and audit['episodes_replayed'] == 240
            and audit['report_sha256'] == sha(raw) and audit['summary'] == run['summary']):
        raise ValueError(f'Audit does not bind current complete report: {path}')
    for i, row in enumerate(run['rows']):
        ep = row['episode']
        if row['index'] != i or ep['status'] != 'complete' or ep['reward'] not in (0, 1):
            raise ValueError(f'Invalid chronological episode at {i}: {path}')
    provenance = {'path': str(path), 'report_sha256': sha(raw),
                  'audit_path': str(audit_path), 'audit_sha256': sha(audit_raw)}
    return run, provenance


def counts(pairs):
    left = sum(a['episode']['reward'] for a, b in pairs)
    right = sum(b['episode']['reward'] for a, b in pairs)
    return {'n': len(pairs), 'left_wins': left, 'right_wins': right,
            'net_left_minus_right': left-right,
            'left_only': sum(a['episode']['reward'] > b['episode']['reward'] for a, b in pairs),
            'right_only': sum(b['episode']['reward'] > a['episode']['reward'] for a, b in pairs),
            'activation_disagreements': sum(a['adapter_active'] != b['adapter_active'] for a, b in pairs)}


def compare(left, right):
    for key in ('seed', 'source_report_sha256', 'model_config_sha256', 'rank',
                'layers', 'max_steps', 'max_new_tokens', 'history_turns', 'loop_guard_max'):
        if left[key] != right[key]:
            raise ValueError(f'Incompatible rollout/setup field: {key}')
    pairs = list(zip(left['rows'], right['rows'], strict=True))
    for a, b in pairs:
        if a['game'] != b['game'] or any(a['episode'][k] != b['episode'][k]
                for k in ('initial_observation', 'initial_commands_sha256')):
            raise ValueError(f'Task/input mismatch at {a["index"]}')
    families = sorted({Path(a['game']).parts[2].split('-')[0] for a, _ in pairs})
    return {'overall': counts(pairs),
            'blocks': [{'start': i, 'end_exclusive': i+40, **counts(pairs[i:i+40])}
                       for i in range(0, 240, 40)],
            'families': {f: counts([(a,b) for a,b in pairs
                                   if Path(a['game']).parts[2].split('-')[0] == f]) for f in families},
            'training_steps': [sum(r['update']['steps'] for r in run['rows']) for run in (left, right)],
            'discordant_tasks': [{'index': a['index'], 'game': a['game'],
                                  'left_reward': a['episode']['reward'],
                                  'right_reward': b['episode']['reward']}
                                 for a,b in pairs if a['episode']['reward'] != b['episode']['reward']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if len(args.reports) < 2 or len(set(p.resolve() for p in args.reports)) != len(args.reports):
        parser.error('At least two distinct complete reports are required')
    loaded = [load_checked(p) for p in args.reports]
    result = {'scope': 'Descriptive paired comparison of independently evolving own-policy streams',
              'analysis_sha256': sha(Path(__file__).read_bytes()),
              'sources': [p for _, p in loaded],
              'comparisons': [{'left': str(args.reports[i]), 'right': str(args.reports[j]),
                               **compare(loaded[i][0], loaded[j][0])}
                              for i,j in itertools.combinations(range(len(loaded)), 2)],
              'limitations': ['Exposed development tasks and one order/seed.',
                              'Own trajectories and training budgets can diverge after bootstrap.',
                              'Read/write/training differences are not isolated causal interventions.',
                              'No independent-task significance test: online outcomes depend on prior tasks.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'comparisons': len(result['comparisons'])}))


if __name__ == '__main__':
    main()
