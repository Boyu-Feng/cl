"""Describe causal source coverage and failure symptoms without assigning causes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path


def summarize(rows):
    failed = [r for r in rows if r['online']['reward'] == 0]
    updates = [r['update'] for r in rows if r['update']['steps'] == 20]
    symptoms = Counter()
    confirmed_operations = Counter()
    operation_episodes = Counter()
    operation_goal_matches = Counter()
    for row in failed:
        ep = row['online']
        commands = [step['command'] for step in ep['trajectory']]
        observations = [step['observation'] for step in ep['trajectory']]
        symptoms['reached_30_steps'] += len(commands) >= 30
        symptoms['no_take_command'] += not any(c.startswith('take ') for c in commands)
        symptoms['repeated_command_in_last_10'] += len(set(commands[-10:])) < len(commands[-10:])
        symptoms['nothing_happens_feedback'] += any('Nothing happens' in o for o in observations)
        present = set()
        for step in ep['trajectory']:
            match = re.fullmatch(r'(clean|heat|cool) (.+) with (.+)', step['command'])
            if match:
                verb, obj, receptacle = match.groups()
                if step['observation'].strip() == f'You {verb} the {obj} using the {receptacle}.':
                    confirmed_operations[verb] += 1
                    present.add(verb)
                    goal = ep['initial_observation'].split('Your task is to:')[-1].strip()
                    adjective = {'clean':'clean', 'heat':'hot', 'cool':'cool'}[verb]
                    goal_match = re.fullmatch(rf'{verb} some (\w+) and put it in (\w+)\.', goal)
                    goal_match = goal_match or re.fullmatch(rf'put a {adjective} (\w+) in (\w+)\.', goal)
                    if goal_match:
                        object_type = re.sub(r' \d+$', '', obj)
                        key = 'matched' if object_type == goal_match.group(1) else 'different_object'
                        operation_goal_matches[f'{verb}_{key}'] += 1
                    else:
                        operation_goal_matches[f'{verb}_goal_not_parsed'] += 1
        operation_episodes.update(present)
    return {'tasks': len(rows), 'wins': sum(r['online']['reward'] for r in rows),
            'adapter_active': sum(r['adapter_active'] for r in rows),
            'active_wins': sum(r['online']['reward'] for r in rows if r['adapter_active']),
            'online_update_events': len(updates),
            'online_training_steps': sum(u['steps'] for u in updates),
            'mean_update_loss': sum(u['loss_mean'] for u in updates)/len(updates) if updates else None,
            'failed_episodes':len(failed), 'overlapping_failure_symptoms': dict(symptoms),
            'feedback_confirmed_operations_in_failures':dict(confirmed_operations),
            'confirmed_operations_public_goal_check':dict(operation_goal_matches),
            'failed_episodes_with_confirmed_operation':dict(operation_episodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/trajectory_hyperlora/alfworld_online_own_trajectory_v52b_stream240_20261010.json'))
    parser.add_argument('--families', type=Path, default=Path('results/trajectory_hyperlora/alf_sibling_train240_taskpair_current1000_20261006.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    family_raw = args.families.read_bytes()
    families = {r['game']:r['family'] for r in json.loads(family_raw)['games']}
    rows = report['rows']
    assert [r['index'] for r in rows] == list(range(len(rows)))
    assert all(r['online']['status'] == 'complete' for r in rows)
    result = {'report_sha256': hashlib.sha256(raw).hexdigest(),
              'family_report_sha256': hashlib.sha256(family_raw).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'partial_run': len(rows) != 240, 'overall': summarize(rows),
              'families': {family:summarize([r for r in rows if families[r['game']] == family])
                           for family in sorted(set(families.values()))},
              'limitations': 'Descriptive symptoms overlap and do not identify root causes. Feedback-confirmed operations are not approved training labels: they may concern an irrelevant object or repeat an already completed operation. No-take commands need not imply a bug. Family coverage differs; no causal efficacy inference or stopping decision.'}
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
