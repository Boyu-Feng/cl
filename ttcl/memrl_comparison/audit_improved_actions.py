"""Re-score saved raw/final CL actions to attribute task-state gains."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from ttcl.icl_mem0_comparison.protocol import read, save


def audit(root: Path, output: Path):
    from src.tasks.blind_spectrum_monitoring.task import (
        BlindSpectrumMonitoringTask, ScanReport, _score_report,
    )
    from src.tasks.cohort_studies.task import CohortStudiesTask
    from src.tasks.cohort_studies.scoring_cohorts import ALL_COHORTS
    if output.exists():
        raise FileExistsError(output)
    result = {}
    for task_name, prefix, n, stage_size in (
        ('blind_spectrum_monitoring', 'bsm_action_v2', 90, 30),
        ('cohort_studies', 'cohort_action_v3', 20, 4),
    ):
        rows = []
        for repeat in (303, 404):
            run = root / f'20261001_{prefix}_{repeat}'
            for index in range(int(.2*n), n):
                folder = f'episode_{index+1:03d}'
                vanilla = read(run / 'vanilla' / folder / 'row.json')
                improved = read(run / 'public_evidence' / folder / 'row.json')
                if vanilla['status'] != 'complete' or improved['status'] != 'complete':
                    continue
                action = read(run / 'public_evidence' / folder / 'policy_action.json')
                if task_name == 'blind_spectrum_monitoring':
                    if action['prior_scan_count'] != index:
                        raise ValueError('Spectrum policy read non-prior scans')
                    task = BlindSpectrumMonitoringTask(seed=42, schedule='default',
                                                       response_timeout_seconds=0)
                    task.reset_baseline_instance(index)
                    kw = dict(all_latent_channel_defs=task._get_all_latent_channel_defs(),
                              W=task.W, G=task.G, band_width=task.band_width)
                    score = lambda value: _score_report(
                        ScanReport.model_validate({'transmitters': value}), **kw)['score']
                    raw = score(action['actor']['transmitters'])
                    after_current = score(action['actor']['transmitters'] + action['added_current'])
                    final = score(action['final']['transmitters'])
                    extra = dict(after_current=after_current,
                                 added_current=len(action['added_current']),
                                 added_history=len(action['added_history']))
                else:
                    if action['prior_report_count'] > index:
                        raise ValueError('Cohort policy read future reports')
                    task = CohortStudiesTask(seed=42)
                    task.reset_baseline_instance(index)

                    def score(value):
                        estimates = [dict(cohort_id=c.id,
                                          estimated_survival_12m=value[f'{c.id}__s12'],
                                          estimated_survival_24m=value[f'{c.id}__s24'],
                                          estimated_survival_36m=value[f'{c.id}__s36'])
                                     for c in ALL_COHORTS]
                        return task._score_submission(estimates).score

                    raw = score(action['actor'])
                    final = score(action['final'])
                    extra = dict(prior_report_count=action['prior_report_count'])
                    connection = getattr(task, '_conn', None)
                    if connection is not None:
                        connection.close()
                if abs(final-improved['reward']) > 1.e-5:
                    raise ValueError(f'Final action/official reward mismatch: {run}/{folder}')
                rows.append(dict(index=index, repeat=repeat, stage=index//stage_size+1,
                                 vanilla=vanilla['reward'], actor_raw=raw,
                                 final=final, **extra))

        def aggregate(values):
            return dict(n=len(values), vanilla_mean=statistics.fmean(x['vanilla'] for x in values),
                        actor_raw_mean=statistics.fmean(x['actor_raw'] for x in values),
                        final_mean=statistics.fmean(x['final'] for x in values)) if values else None

        result[task_name] = dict(all=aggregate(rows),
                                 stages={str(s):aggregate([x for x in rows if x['stage']==s])
                                         for s in range(1, 1+n//stage_size)},
                                 rows=rows)
    save(output, result)
    print({name:{'all':data['all'],'stages':data['stages']}
           for name,data in result.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
