"""Component removals on complete paired ALFWorld and CLBench grids."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics
import time

from ttcl.experience_evolution.core import read, save, seed
from ttcl.experience_lab.collect import messages_for
from ttcl.experience_lab.evaluate import summarize as base_summarize, promote
from .environments import Environments, writer


def evaluate_shard(root, stage, sequence_id):
    root=Path(root);plan=read(root/'plan.json');directory=root/'evaluation'/stage
    settings=read(directory/'evaluation_plan.json')
    sequence=next(s for s in settings['sequences'] if s['id']==sequence_id)
    envs={model:Environments(plan,model) for model in {r['reader'] for r in settings['routes'].values()}}
    try:
        for repeat in settings['seeds']:
            memories={a:'' for a in settings['arms']}
            for position,spec in enumerate(sequence['tasks']):
                dest=directory/'episodes'/sequence_id/str(repeat)/f'task_{position:03}'
                identity=None
                for arm,route in settings['routes'].items():
                    env=envs[route['reader']]
                    result=env.run(spec,memories[arm],seed(repeat,sequence_id,spec['id']),dest/arm/'actor')
                    current=(result['task_identity'],result['initial_query_hash'])
                    if identity is not None and identity!=current: raise ValueError('Evaluation initial state mismatch')
                    identity=current
                    row={'domain':sequence['domain'],'family':sequence.get('family'),
                        'sequence':sequence_id,'position':position,'task':spec['id'],'seed':repeat,
                        'arm':arm,'reward':result['reward'],'status':result['row']['status'],
                        'memory_before':memories[arm],'route':route,'weights_frozen':True,'cost':result['row']}
                    if route['writer']:
                        update=writer(env.client,messages_for(memories[arm],spec,result),route['writer'],
                            dest/arm/'writer.json',seed(repeat,sequence_id,position,'writer'),plan)
                        if update['usable']: memories[arm]=update['raw_response']
                        row['writer_accepted']=update['usable']
                        row['writer_cost']={k:update[k] for k in ['input_tokens','output_tokens','seconds']}
                    save(dest/arm/'result.json',row)
                save(directory/'workers'/f'{sequence_id}.json',{'phase':'evaluating','seed':repeat,
                    'completed_tasks':position+1,'expected_tasks':len(sequence['tasks']),'time':time.time()})
        save(directory/'workers'/f'{sequence_id}.json',{'phase':'complete'})
    finally:
        for env in envs.values(): env.close()


def summarize(directory):
    directory=Path(directory);summary=base_summarize(directory)
    rows=[read(p) for p in (directory/'episodes').glob('*/*/task_*/*/result.json')]
    groups=defaultdict(dict)
    for row in rows:
        if row['position']>0:
            groups[(row['domain'],row['sequence'],row['task'],row['seed'])][row['arm']]=row['reward']
    contrasts=defaultdict(lambda:defaultdict(list))
    for key,values in groups.items():
        needed=['dual','dual_writer_base','old_writer_reader','original_delta','reader_no_text','ppo8_writer']
        if any(values.get(a) is None for a in needed): continue
        d=contrasts[key[0]]
        d['reader_added_to_dual_writer'].append(values['dual']-values['dual_writer_base'])
        d['new_writer_with_same_reader'].append(values['dual']-values['old_writer_reader'])
        d['text_added_to_reader'].append(values['dual']-values['reader_no_text'])
        d['dual_vs_writer_only_training'].append(values['dual']-values['ppo8_writer'])
        d['component_interaction'].append(values['dual']-values['dual_writer_base']-
                                           values['old_writer_reader']+values['original_delta'])
    summary['component_contrasts']={d:{k:{'mean':statistics.mean(v),'paired_n':len(v)}
                                       for k,v in items.items()} for d,items in contrasts.items()}
    summary['contrast_scope']='Online component removal with independently evolving trajectories, not fixed-history causal mediation or significance.'
    save(directory/'summary.json',summary)
    return summary
