"""Independently re-score saved CLBench contextual-utility actions."""
from __future__ import annotations

import argparse
from pathlib import Path

from ttcl.icl_mem0_comparison.protocol import read, save


def audit(directory):
    design = read(directory/'design.json')
    task_name = design['task']
    if design['benchmark'] != 'clbench' or task_name not in (
            'blind_spectrum_monitoring', 'cohort_studies'):
        raise ValueError('Only BSM and Cohort have supported independent scorers')
    if task_name == 'blind_spectrum_monitoring':
        from src.tasks.blind_spectrum_monitoring.task import (
            BlindSpectrumMonitoringTask, ScanReport, _score_report)
    else:
        from src.tasks.cohort_studies.task import CohortStudiesTask
        from src.tasks.cohort_studies.scoring_cohorts import ALL_COHORTS
    rows = []
    for index in range(design['limit']):
        episode = directory/'contextual_utility'/f'episode_{index+1:03d}'
        if not (episode/'row.json').exists():
            continue
        row = read(episode/'row.json')
        if row['status'] != 'complete':
            continue
        action = read(episode/'policy_action.json')
        if task_name == 'blind_spectrum_monitoring':
            task = BlindSpectrumMonitoringTask(seed=42, schedule='default',
                                               response_timeout_seconds=0)
            task.reset_baseline_instance(index)
            kwargs = dict(all_latent_channel_defs=task._get_all_latent_channel_defs(),
                          W=task.W, G=task.G, band_width=task.band_width)
            def score(value):
                parsed = ScanReport.model_validate(value)
                return _score_report(parsed, **kwargs)['score']
        else:
            task = CohortStudiesTask(seed=42)
            task.reset_baseline_instance(index)
            def score(value):
                estimates = [dict(cohort_id=c.id,
                                  estimated_survival_12m=value[f'{c.id}__s12'],
                                  estimated_survival_24m=value[f'{c.id}__s24'],
                                  estimated_survival_36m=value[f'{c.id}__s36'])
                             for c in ALL_COHORTS]
                return task._score_submission(estimates).score
        try:
            raw = score(action['actor'])
            final = score(action['final'])
        finally:
            connection = getattr(task, '_conn', None)
            if connection is not None:
                connection.close()
        if abs(final-row['reward']) > 1e-5:
            raise ValueError(f'Final score mismatch: {episode}')
        rows.append(dict(index=index, actor=raw, final=final,
                         changed=action['actor'] != action['final'],
                         recorded=row['reward'], operator=action['operator']))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    rows = audit(a.directory.resolve())
    if a.output:
        save(a.output, rows)
    print(dict(verified=len(rows), changed=sum(x['changed'] for x in rows),
               raw_total=sum(x['actor'] for x in rows),
               final_total=sum(x['final'] for x in rows)))


if __name__ == '__main__':
    main()
