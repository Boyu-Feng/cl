"""Read-only snapshot: trigger coverage, action changes, gains and damage.

Outcome-conditioned groups are retrospective diagnostics, never write labels
or runtime trigger inputs. Partial snapshots are not completed-run results.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/trajectory_hyperlora/alfworld_within_attempt_correction_v61_dev24_20261010.json'))
    p.add_argument('--retry', type=Path, default=Path('results/trajectory_hyperlora/alfworld_failure_retry_v31b_dev24_20261009.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw, retry_raw = args.input.read_bytes(), args.retry.read_bytes()
    report, retry = json.loads(raw), json.loads(retry_raw)
    if report['failures'] or retry['failures']:
        raise ValueError('Failed runs require separate diagnosis')
    old = {x['game']: x for x in retry['games']}
    result = {'input_sha256': hashlib.sha256(raw).hexdigest(),
        'retry_sha256': hashlib.sha256(retry_raw).hexdigest(),
        'n': len(report['rows']), 'full_dev24': len(report['rows']) == 24,
        'scope': 'retrospective exposed-development snapshot; no new policy labels or efficacy test',
        'rows': []}
    latencies = []
    for row in report['rows']:
        base, away, random = [row[k] for k in ('none', 'away', 'random')]
        prior = old[row['game']]
        assert prior['first_reward'] == base['reward']
        event = away['within_attempt_correction']
        first_difference = next((i for i,(a,b) in enumerate(zip(base['trajectory'], away['trajectory']))
            if a['command'] != b['command']), None)
        latency = first_difference-event['before_turn'] if event and first_difference is not None else None
        if latency is not None:
            assert latency >= 0
            latencies.append(latency)
        previous_retry = prior['retries'].get('away_from_failure', {})
        previous_retry = previous_retry.get('episode', previous_retry)
        retry_rescue = bool(base['reward'] == 0 and previous_retry.get('reward') == 1)
        item = {'index': row['index'], 'game': row['game'],
            'goal': base['initial_observation'].split('Your task is to:')[-1].strip(),
            'base_reward': base['reward'], 'away_reward': away['reward'], 'random_reward': random['reward'],
            'base_steps': base['steps'], 'away_steps': away['steps'],
            'trigger_turn': event['before_turn'] if event else None,
            'trigger_reason': event['reason'] if event else None,
            'first_different_command_turn': first_difference,
            'trigger_to_first_command_change': latency,
            'old_full_failure_retry_rescued': retry_rescue,
            'gain': away['reward'] > base['reward'], 'damage': away['reward'] < base['reward']}
        if item['gain'] or item['damage'] or retry_rescue:
            start = max(0, (first_difference or (event['before_turn'] if event else 12))-2)
            item['command_window'] = {arm: [{'turn': x['turn'], 'command': x['command'],
                'feedback': x['observation']} for x in row[arm]['trajectory'][start:start+7]] for arm in ('none','away')}
        result['rows'].append(item)
    rows = result['rows']
    result['summary'] = {'wins': {k: sum(r[k+'_reward'] for r in rows) for k in ('base','away','random')},
        'triggered': sum(r['trigger_turn'] is not None for r in rows),
        'trigger_reasons': dict(Counter(r['trigger_reason'] for r in rows if r['trigger_reason'])),
        'changed_commands': sum(r['first_different_command_turn'] is not None for r in rows),
        'gains': [r['index'] for r in rows if r['gain']], 'damage': [r['index'] for r in rows if r['damage']],
        'triggers_on_eventual_baseline_success': [r['index'] for r in rows if r['trigger_turn'] is not None and r['base_reward']==1],
        'old_retry_rescues_without_trigger': [r['index'] for r in rows if r['old_full_failure_retry_rescued'] and r['trigger_turn'] is None],
        'old_retry_rescues_recovered_first_attempt': [r['index'] for r in rows if r['old_full_failure_retry_rescued'] and r['away_reward']==1],
        'median_trigger_to_action_change': statistics.median(latencies) if latencies else None}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'n': result['n'], **result['summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
