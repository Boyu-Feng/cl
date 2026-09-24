"""Online evaluation, paired per-domain results, development-only promotion."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics
import random
import time

from ttcl.experience_evolution.core import read, save, seed
from .collect import Environments, messages_for, generate
from .prepare import TOTALS


def evaluate_shard(root, run_id, stage, sequence_id):
    root=Path(root);plan=read(root/'plan.json');directory=root/run_id/stage
    settings=read(directory/'evaluation_plan.json')
    sequence=next(s for s in settings['sequences'] if s['id']==sequence_id)
    actor=Environments(plan)
    try:
        for repeat in settings['seeds']:
            memories={a:'' for a in settings['arms']}
            for position,spec in enumerate(sequence['tasks']):
                dest=directory/'episodes'/sequence_id/str(repeat)/f'task_{position:03}'
                jobs=[(spec,memories[a],seed(repeat,sequence_id,spec['id']),dest/a/'actor')
                      for a in settings['arms']]
                results=actor.run_many(jobs)
                if len({str((r['task_identity'],r['initial_query_hash'])) for r in results})!=1:
                    raise ValueError('Evaluation branches do not share the task state')
                for arm,result in zip(settings['arms'],results):
                    row={'domain':sequence['domain'],'family':sequence.get('family'),
                         'sequence':sequence_id,'position':position,'task':spec['id'],'seed':repeat,
                         'arm':arm,'reward':result['reward'],'status':result['row']['status'],
                         'memory_before':memories[arm],'actor_frozen':True,
                         'cost':result['row'],'reused_from':result.get('reused_from')}
                    if arm!='none':
                        messages=messages_for(memories[arm],spec,result)
                        update=generate(actor.client,messages,'frozen-actor' if arm=='untrained' else arm,dest/arm/'writer.json',
                                        seed(repeat,sequence_id,position,'writer'),plan)
                        # Fixed conservative policy applied to ALL writers.
                        if update['usable']: memories[arm]=update['raw_response']
                        row['writer_accepted']=update['usable']
                        row['writer_cost']={k:update[k] for k in ['input_tokens','output_tokens','seconds']}
                    save(dest/arm/'result.json',row)
                save(directory/'workers'/f'{sequence_id}.json', {'phase':'evaluating','seed':repeat,
                     'completed_tasks':position+1,'expected_tasks':len(sequence['tasks']),'updated_at':time.time()})
        save(directory/'workers'/f'{sequence_id}.json',{'phase':'complete'})
    finally: actor.close()


def summarize(directory):
    directory=Path(directory);settings=read(directory/'evaluation_plan.json')
    rows=[read(p) for p in (directory/'episodes').glob('*/*/task_*/*/result.json')]
    expected=sum(len(s['tasks']) for s in settings['sequences'])*len(settings['seeds'])*len(settings['arms'])
    groups=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        if row['position']>0:
            groups[row['domain']][row['arm']][(row['sequence'],row['task'],row['seed'])]=row
    domains={}
    for domain,arms in groups.items():
        common=set.intersection(*(set(arms[a]) for a in settings['arms']))
        complete={k for k in common if all(arms[a][k]['reward'] is not None for a in settings['arms'])}
        domains[domain]={}
        for arm in settings['arms']:
            if not complete:continue
            diffs=[arms[arm][k]['reward']-arms['original_delta'][k]['reward'] for k in sorted(complete)]
            per_seed={str(s):statistics.mean(arms[arm][k]['reward']-arms['original_delta'][k]['reward']
                         for k in complete if k[-1]==s) for s in settings['seeds']
                         if any(k[-1]==s for k in complete)}
            clusters=defaultdict(list)
            for key in sorted(complete):
                clusters[key[1]].append(arms[arm][key]['reward']-arms['original_delta'][key]['reward'])
            rng=random.Random(seed(92471,domain,arm,'bootstrap'))
            vectors=list(clusters.values());draws=[]
            for _ in range(2000):
                sampled=[vectors[rng.randrange(len(vectors))] for _ in vectors]
                draws.append(sum(sum(v) for v in sampled)/sum(len(v) for v in sampled))
            draws.sort()
            domains[domain][arm]={'paired_n':len(complete),'unique_tasks':len(vectors),
                'paired_task_bootstrap_95':[draws[49],draws[1949]],'excluded_missing':len(common)-len(complete),
                'mean_reward':statistics.mean(arms[arm][k]['reward'] for k in complete),
                'vs_original_delta':statistics.mean(diffs),'by_seed_delta':per_seed,
                'wins':sum(d>1e-8 for d in diffs),'ties':sum(abs(d)<=1e-8 for d in diffs),
                'losses':sum(d < -1e-8 for d in diffs)}
    result={'complete':len(rows)==expected,'completed':len(rows),'expected':expected,
            'domains':domains,'score_rule':'official per-domain rewards; first task of each chain excluded',
            'missing_scores':sum(r['reward'] is None for r in rows)}
    save(directory/'summary.json',result)
    lines=['# Experience writer exploration', '', f'Completed {len(rows)}/{expected}; missing official scores {result["missing_scores"]}.',
           '', 'Official rewards are reported separately by domain; intervals cluster all seeds of a task.',
           '', '| Domain | Arm | Paired records | Mean reward | Delta original | 95% interval |',
           '|---|---|---:|---:|---:|---|']
    for domain,arms in domains.items():
        for arm,value in arms.items():
            lo,hi=value['paired_task_bootstrap_95']
            lines.append(f'| {domain} | {arm} | {value["paired_n"]} | {value["mean_reward"]:.5f} | {value["vs_original_delta"]:+.5f} | [{lo:+.5f}, {hi:+.5f}] |')
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def promote(summary, candidate_arms):
    """A screening gate, not a significance claim or permission to touch test data."""
    if not summary['complete'] or summary['missing_scores']:
        return None
    accepted=[]
    for arm in candidate_arms:
        if any(arm not in summary['domains'].get(d,{}) for d in ['alfworld',*TOTALS]):continue
        alf=summary['domains']['alfworld'][arm]
        cl=[summary['domains'][d][arm] for d in TOTALS]
        # Both execution seeds must show nonnegative ALF change.
        if (alf['vs_original_delta']>1e-8 and min(alf['by_seed_delta'].values())>=-1e-8
                and all(v['vs_original_delta']>=-1e-8 for v in cl)
                and any(v['vs_original_delta']>1e-8 for v in cl)):
            # Compare CL domains via equal-weight paired win-minus-loss rates.
            score=alf['vs_original_delta']+statistics.mean((v['wins']-v['losses'])/v['paired_n'] for v in cl)
            accepted.append((score,arm))
    return max(accepted)[1] if accepted else None
