"""Read-only checks of scored artifacts; writes only additional analysis files."""

import hashlib
import json
from pathlib import Path
import statistics
import sys

from ttcl.experience_diagnostic.core import ARMS, paired_summary


def read(path):
    return json.loads(path.read_text())


def pairwise_controls(rows):
    groups = {}
    for row in rows:
        key = (row['source_episode'], row['canonical_index'], row['repeat'])
        groups.setdefault(key, {})[row['arm']] = row
    result = {}
    for arm in ARMS:
        values = [v[arm]['reward'] - v['keep']['reward'] for v in groups.values()
                  if all(v.get(a, {}).get('status') == 'complete' for a in [arm, 'keep'])]
        result[arm] = {
            'paired_count': len(values),
            'mean_delta_vs_keep': statistics.mean(values) if values else None,
            'completed_cells': sum(r['arm'] == arm and r['status'] == 'complete' for r in rows),
            'failed_cells': sum(r['arm'] == arm and r['status'] != 'complete' for r in rows),
        }
    return result


def analyze(root):
    records = []
    errors = []
    for path in sorted((root / 'scores').rglob('row.json')):
        row = read(path)
        source = Path(row.get('reused_from', path.parent))
        expected = read(root / 'candidates' / row['task'] / str(row['source_episode']) / f'{row["arm"]}_context.json')
        if expected['sha256'] != row['bank_context_sha256']:
            errors.append(f'Context hash mismatch: {path}')
        contexts = [json.loads(s) for s in (source / 'memory_contexts.jsonl').read_text().splitlines()]
        if contexts and contexts[0]['memory_context'] != expected['context']:
            errors.append(f'Actual actor context mismatch: {path}')
        response_path = source / 'responses.jsonl'
        responses = [json.loads(s) for s in response_path.read_text().splitlines()] if response_path.exists() else []
        for response in responses:
            if response.get('writer_adapter_enabled') is not False:
                errors.append(f'Actor adapter flag not explicitly false: {path}')
        episode = read(source / 'trajectory.json')
        steps = episode['steps']
        facts = {
            'task': row['task'], 'arm': row['arm'], 'source_episode': row['source_episode'],
            'probe_episode': row['episode'], 'repeat': row['repeat'], 'status': row['status'],
            'reward': row['reward'], 'actor_calls': row['actor_calls'],
            'actor_input_tokens': row['actor_input_tokens'], 'actor_output_tokens': row['actor_output_tokens'],
            'context_tokens': expected['tokens'], 'seconds': row['actor_seconds'],
            'reused': 'reused_from' in row, 'artifact': str(source),
            'empty_query_results': sum('(no results)' in s['public_feedback'] for s in steps),
            'tool_error_steps': sum('ERROR:' in s['public_feedback'] for s in steps),
            'incorrect_feedback': any('INCORRECT' in s['public_feedback'] for s in steps),
            'correct_feedback': any(': CORRECT' in s['public_feedback'] for s in steps),
            'actions': [s['action'].get('tool_call', s['action']) for s in steps],
        }
        records.append(facts)
    # Same actor generation seeds at matched turn/format-retry positions.
    by_cell = {}
    for path in sorted((root / 'scores').rglob('row.json')):
        row = read(path)
        src = Path(row.get('reused_from', path.parent))
        rp = src / 'responses.jsonl'
        responses = [json.loads(s) for s in rp.read_text().splitlines()] if rp.exists() else []
        for response in responses:
            key = (row['task'], row['source_episode'], row['canonical_index'], row['repeat'], response['turn'], response['format_retry'])
            seed = response.get('actual_generation_seed')
            previous = by_cell.setdefault(key, seed)
            if previous != seed or seed is None:
                errors.append(f'Seed pairing mismatch: {key}')
    summary = {}
    for task in ['database_exploration', 'cohort_studies']:
        task_rows = [read(p) for p in sorted((root / 'scores' / task).rglob('row.json'))]
        paired = paired_summary(task_rows)
        strata = {}
        for ep in [13, 17]:
            strata[str(ep)] = paired_summary([r for r in task_rows if r['source_episode'] == ep])
        summary[task] = {'paired': paired, 'by_source': strata,
                         'secondary_pairwise_available': pairwise_controls(task_rows)}
    actual = [r for r in records if not r['reused']]
    result = {
        'summaries': summary, 'checks_passed': not errors, 'errors': errors,
        'recorded_cells': len(records), 'executed_episodes': len(actual),
        'reused_cells': len(records) - len(actual),
        'actual_actor_calls': sum(r['actor_calls'] for r in actual),
        'actual_actor_input_tokens': sum(r['actor_input_tokens'] for r in actual),
        'actual_actor_output_tokens': sum(r['actor_output_tokens'] for r in actual),
        'actual_actor_seconds_sum': sum(r['seconds'] for r in actual),
    }
    (root / 'behavior.json').write_text(json.dumps(records, indent=2))
    (root / 'analysis.json').write_text(json.dumps(result, indent=2))
    (root / 'analysis_script.sha256').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}, indent=2))


if __name__ == '__main__':
    analyze(Path(sys.argv[1]).resolve())
