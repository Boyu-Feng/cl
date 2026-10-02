"""Retrospective public-submission ensemble check for CLBench Cohort Studies.

The candidate action averages only earlier raw submitted predictions. Official
ground truth is accessed only by the scorer after that action is fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save


def run(origin: Path, output: Path):
    from src.tasks.cohort_studies.task import CohortStudiesTask
    from src.tasks.cohort_studies.scoring_cohorts import ALL_COHORTS
    if output.exists():
        raise FileExistsError(output)

    def score(task, report):
        estimates = [dict(cohort_id=c.id,
                          estimated_survival_12m=report[f'{c.id}__s12'],
                          estimated_survival_24m=report[f'{c.id}__s24'],
                          estimated_survival_36m=report[f'{c.id}__s36'])
                     for c in ALL_COHORTS]
        return task._score_submission(estimates).score

    cells = []
    root = origin / 'runs' / 'clbench' / 'cohort_studies'
    for arm in ('none', 'memrl'):
        for repeat in (303, 404):
            prior = []
            for index in range(20):
                directory = root / str(repeat) / arm / f'episode_{index+1:03d}'
                trace = read(directory / 'public_trajectory.json')
                report = next((step['action'] for step in reversed(trace)
                               if len(step.get('action', {})) == 108), None)
                if report is None:
                    continue
                task = CohortStudiesTask(seed=42)
                task.reset_baseline_instance(index)
                current_score = score(task, report)
                row = read(directory / 'row.json')
                if row['reward'] is not None and abs(current_score - row['reward']) > 1.e-5:
                    raise ValueError(f'Official score mismatch: {arm}/{repeat}/{index+1}')
                if prior:
                    average = {key: statistics.fmean(item[key] for item in prior)
                               for key in report}
                    ensemble_score = score(task, average)
                else:
                    ensemble_score = current_score
                cells.append(dict(arm=arm, repeat=repeat, index=index,
                                  prior_reports=len(prior), current_score=current_score,
                                  ensemble_score=ensemble_score,
                                  scored=row['reward'] is not None))
                prior.append(report)
                connection = getattr(task, '_conn', None)
                if connection is not None:
                    connection.close()
    summaries = {}
    for arm in ('none', 'memrl'):
        for repeat in (303, 404):
            selected = [r for r in cells if r['arm'] == arm and r['repeat'] == repeat
                        and r['index'] >= 4 and r['scored']]
            summaries[f'{arm}/{repeat}'] = dict(
                n=len(selected), current_mean=statistics.fmean(r['current_score'] for r in selected),
                ensemble_mean=statistics.fmean(r['ensemble_score'] for r in selected),
                wins=sum(r['ensemble_score'] > r['current_score'] for r in selected),
                losses=sum(r['ensemble_score'] < r['current_score'] for r in selected),
                ties=sum(r['ensemble_score'] == r['current_score'] for r in selected))
    save(output, dict(note=__doc__, origin=str(origin), summaries=summaries, cells=cells))
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.origin.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
