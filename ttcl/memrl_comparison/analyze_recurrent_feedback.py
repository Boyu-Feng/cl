"""Audit paired v30 CLBench runs and their source-bound feedback readouts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save, sha
from .memory import digest
from .recurrent_feedback import stable_feedback


def analyze(output: Path, *, require_complete: bool = True) -> dict:
    design = read(output / 'design.json')
    if design['benchmark'] != 'clbench':
        raise ValueError('Not a CLBench run')
    versions = [version for version in (30, 32)
                if (output / 'source' / f'grounded_evidence_v{version}.py').exists()
                and design['implementation_sha256'] == sha(
                    output / 'source' / f'grounded_evidence_v{version}.py')]
    if len(versions) != 1:
        raise ValueError('Unknown frozen recurrent-feedback policy')
    version = versions[0]
    for filename in (f'evaluate_typed_grounded_v{version}.py',
                     'recurrent_feedback.py'):
        path = Path(__file__).with_name(filename)
        if sha(path) != sha(output / 'source' / filename):
            raise ValueError(f'Frozen source changed: {filename}')
    if sha(Path(__file__).with_name(f'evaluate_typed_grounded_v{version}.py')) != design[
            'runner_sha256']:
        raise ValueError('Frozen evaluator changed')
    origin = Path(design['origin'])
    if sha(origin / 'plan.json') != design['origin_plan_sha256']:
        raise ValueError('Origin plan changed')
    total = read(origin / 'plan.json')['tasks'][design['task']]
    if design['limit'] > total or len(design['bindings']) != design['limit']:
        raise ValueError('Invalid frozen task count or bindings')
    rows, missing = [], []
    for index in range(1, design['limit'] + 1):
        paths = [output / arm / f'episode_{index:03d}'
                 for arm in ('vanilla', 'typed_grounded')]
        if not all((path / 'row.json').exists() for path in paths):
            missing.append(index)
            continue
        native, candidate = [read(path / 'row.json') for path in paths]
        if any(row['status'] != 'complete' or
               row['initial_query_sha256'] != design['bindings'][index - 1] or
               sha(path / 'memory_after.json') != row['memory_after_sha256']
               for row, path in zip((native, candidate), paths)):
            raise ValueError(f'Invalid paired outcome or snapshot: {index}')
        nr, cr = [read(path / 'retrieval.json') for path in paths]
        selected = cr.get('recurrent_feedback', [])
        rejected = cr.get('rejected_recurrent_feedback', [])
        if rejected and (version != 32 or selected or len(rejected) != 1):
            raise ValueError(f'Invalid readout abstention: {index}')
        if selected or rejected:
            if index == 1:
                raise ValueError('First task has no prior feedback')
            before = read(output / 'typed_grounded' /
                          f'episode_{index-1:03d}' / 'memory_after.json')
            prior = before['grounded_evidence']['events']
            groups = stable_feedback(prior)
            available = {(digest(group['action']), hashlib.sha256(
                group['feedback'].encode()).hexdigest()): group for group in groups}
            for item in selected + rejected:
                key = (item['action_sha256'], item['feedback_sha256'])
                group = available.get(key)
                if (group is None or group['episode_count'] != item['episode_count'] or
                        group['source_sha256'] != item['source_sha256'] or
                        group['public_trace_sha256'] != item['public_trace_sha256']):
                    raise ValueError(f'Unbound recurrent feedback: {index}')
        if cr['tokens'] > read(origin / 'plan.json')['memory_tokens']:
            raise ValueError(f'Actor memory budget exceeded: {index}')
        rows.append(dict(index=index, native=native['reward'],
                         candidate=candidate['reward'],
                         delta=candidate['reward'] - native['reward'],
                         native_calls=native['actor_calls'],
                         candidate_calls=candidate['actor_calls'],
                         readouts=len(selected),
                         abstained_readouts=len(rejected),
                         native_ids_match=nr['ids'] == cr['ids'],
                         native_context_match=nr['context_sha256'] ==
                         cr['context_sha256']))
    if missing and require_complete:
        raise ValueError(f'Missing {len(missing)} paired tasks')
    full_design = design['limit'] == total
    suffix = ([row for row in rows if row['index'] > int(.2 * total)]
              if full_design else [])
    suffix_summary = dict(
        pairs=len(suffix),
        expected=(total - int(.2 * total) if full_design else None),
        mean_delta=(statistics.fmean(row['delta'] for row in suffix)
                    if suffix else None),
        wins=sum(row['delta'] > 0 for row in suffix),
        losses=sum(row['delta'] < 0 for row in suffix),
        ties=sum(row['delta'] == 0 for row in suffix))
    result = dict(schema='recurrent_feedback_analysis_v1',
                policy_version=version, task=design['task'], repeat=design['repeat'],
                expected=design['limit'], completed=len(rows), missing=missing,
                readout_tasks=sum(row['readouts'] > 0 for row in rows),
                total_readouts=sum(row['readouts'] for row in rows),
                abstained_tasks=sum(row['abstained_readouts'] > 0 for row in rows),
                native_ids_matched=sum(row['native_ids_match'] for row in rows),
                actor_call_delta=sum(row['candidate_calls'] - row['native_calls']
                                     for row in rows),
                prefix_mean_delta=(statistics.fmean(row['delta'] for row in rows)
                                   if rows else None),
                rows=rows,
                note='Official runner rewards; tool-action correctness not independently re-scored')
    if full_design and not missing:
        result['official_suffix'] = suffix_summary
    elif full_design:
        result['partial_suffix'] = suffix_summary
    else:
        result['calibration_prefix'] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    report = analyze(output, require_complete=not args.allow_partial)
    save(output / ('analysis_partial.json' if args.allow_partial else 'analysis.json'),
         report)
    print(f'Audited {report["completed"]}/{report["expected"]} paired tasks')


if __name__ == '__main__':
    main()
