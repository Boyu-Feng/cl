"""Recheck saved CL prefix probe rewards against official task outputs.

Spectrum and Cohort additionally re-score the final submitted action with a
fresh task instance.  Poker and Database have stateful interaction histories;
for those, compare the result with the independent runner outcome record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save


def _action_score(task_name: str, task_seed: int, index: int, action: dict) -> float:
    if task_name == 'blind_spectrum_monitoring':
        from src.tasks.blind_spectrum_monitoring.task import (
            BlindSpectrumMonitoringTask, ScanReport, _score_report)
        task = BlindSpectrumMonitoringTask(seed=task_seed, schedule='default',
                                           response_timeout_seconds=0)
        try:
            task.reset_baseline_instance(index)
            return float(_score_report(
                ScanReport.model_validate(action),
                all_latent_channel_defs=task._get_all_latent_channel_defs(),
                W=task.W, G=task.G, band_width=task.band_width)['score'])
        finally:
            connection = getattr(task, '_conn', None)
            if connection is not None:
                connection.close()
    if task_name == 'cohort_studies':
        from src.tasks.cohort_studies.task import CohortStudiesTask
        from src.tasks.cohort_studies.scoring_cohorts import ALL_COHORTS
        task = CohortStudiesTask(seed=task_seed)
        try:
            task.reset_baseline_instance(index)
            estimates = [dict(
                cohort_id=c.id,
                estimated_survival_12m=action[f'{c.id}__s12'],
                estimated_survival_24m=action[f'{c.id}__s24'],
                estimated_survival_36m=action[f'{c.id}__s36'])
                for c in ALL_COHORTS]
            return float(task._score_submission(estimates).score)
        finally:
            connection = getattr(task, '_conn', None)
            if connection is not None:
                connection.close()
    raise ValueError('Independent final-action scorer unavailable')


def audit(output: Path) -> dict:
    design = read(output / 'design.json')
    origin_plan = read(Path(design['origin']) / 'plan.json')
    rows, missing = [], []
    for case in design['cases']:
        benchmark, task, repeat, arm, episode = Path(case).parts
        if benchmark != 'clbench' or arm != 'memrl':
            raise ValueError(f'Unexpected CL case: {case}')
        index = int(episode.split('_')[1]) - 1
        case_dir = output / benchmark / task / repeat / episode
        for actor_repeat in design['cl_sampling_repeats']:
            trial = case_dir / f'actor_repeat_{actor_repeat}'
            summary_path = trial / 'summary.json'
            if not summary_path.exists():
                missing.append(dict(case=case, actor_repeat=actor_repeat))
                continue
            summary = read(summary_path)
            for branch, recorded in summary['replay'].items():
                directory = trial / branch
                result = read(directory / 'result.json')
                progress = read(directory / 'progress.json')
                outcomes = progress['outcomes']
                indexed_globally = task in ('blind_spectrum_monitoring',
                                            'exploitable_poker')
                if (recorded != result or len(outcomes) != 1 or
                        (indexed_globally and
                         outcomes[0]['instance_index'] != index) or
                        abs(float(outcomes[0]['reward']) - result['reward']) > 1e-8):
                    raise ValueError(f'Official runner outcome mismatch: {directory}')
                independent = None
                if task in ('blind_spectrum_monitoring', 'cohort_studies'):
                    events = [json.loads(line) for line in
                              (directory / 'public_observations.jsonl').read_text().splitlines()]
                    final = [event for event in events if event['instance_complete']]
                    if len(final) != 1 or not isinstance(final[0]['action'], dict):
                        raise ValueError(f'Missing final submitted action: {directory}')
                    independent = _action_score(task, origin_plan['task_seed'],
                                                index, final[0]['action'])
                    if abs(independent - result['reward']) > 1e-5:
                        raise ValueError(f'Independent official score mismatch: {directory}')
                rows.append(dict(case=case, actor_repeat=actor_repeat,
                                 branch=branch, reward=result['reward'],
                                 independent_score=independent))
    return dict(expected_cases=len(design['cases']) *
                len(design['cl_sampling_repeats']),
                completed_cases=len(design['cases']) *
                len(design['cl_sampling_repeats']) - len(missing),
                verified_branches=len(rows),
                independently_rescored=sum(row['independent_score'] is not None
                                            for row in rows),
                missing=missing, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report = audit(output)
    save(output / 'action_audit.json', report)
    print(json.dumps({key:value for key,value in report.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
