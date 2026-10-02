from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from ttcl.icl_mem0_comparison.protocol import Client, read, save, sha
from ttcl.memrl_comparison.memory import Memory, digest


def alf_query(observation):
    marker = 'Your task is to:'
    return observation.split(marker,1)[1].strip() if marker in observation else observation


def alf_cell(plan, client, memory, item, repeat, arm, output):
    from ttcl.alfworld_comparison.environment import Actor, make_env
    from ttcl.alfworld_comparison.methods import public_episode
    from ttcl.experience_evolution.core import seed
    class LocalActor(Actor):
        def generate(self, messages, random_seed):
            result = client.complete(messages, random_seed, tokens=plan['alf']['actor_max_tokens'],
                                     temperature=plan['alf']['actor_temperature'],top_p=1.)
            return dict(text=result['raw_response'], finish_reason=result['finish_reason'],
                usage={'prompt_tokens':result['input_tokens'],'completion_tokens':result['output_tokens']},
                seed=random_seed, prompt_sha256=digest(messages), seconds=result['seconds'],
                rendered_prompt_sha256=result['rendered_prompt_sha256'])
    output.mkdir(parents=True,exist_ok=False)
    before = (memory.calls,memory.input_tokens,memory.output_tokens) if memory else (0,0,0)
    limits_before=memory.limit_hits if memory else 0
    game = Path(plan['alf']['data_root'])/item['path']
    if sha(game)!=item['sha256']:
        raise ValueError('ALFWorld input content changed')
    env = make_env(game)
    try:
        observation = str(env.reset()['feedback'])
    finally:
        env.close()
    query = alf_query(observation)
    actor = LocalActor(plan['alf'])
    episodes = []
    row = dict(benchmark='alfworld', task=item['family'],repeat=repeat,arm=arm,
        game=item['path'],input_sha256=item['sha256'],status='failed',reward=None)
    try:
        for attempt in range(plan['alf']['max_attempts']):
            retrieval = memory.retrieve(query) if memory else {'context':'','ids':[]}
            save(output/f'retrieval_{attempt+1}.json',retrieval)
            episode = actor.run_many([dict(game=item['path'],memory=retrieval['context'],
                seed=seed(repeat,item['path'],attempt,'actor'),output=output/f'attempt_{attempt+1}')])[0]
            if episode['initial_observation']!=observation:
                raise ValueError('Task reset observation changed')
            episodes.append(episode)
            if memory:
                public = public_episode(episode)
                trace = json.dumps(public,ensure_ascii=False)
                update = memory.update(query,trace,1. if episode['reward'] else -1.,bool(episode['reward']),
                    retrieval, {'game_sha256':item['sha256'],'public_content_sha256':digest(public),
                                'attempt':attempt,'repeat':repeat})
                save(output/f'update_{attempt+1}.json',update)
            if episode['reward']:
                break
        row.update(status='complete',reward=max(ep['reward'] for ep in episodes),
                   first_attempt=episodes[0]['reward'],within_three=max(ep['reward'] for ep in episodes))
    except Exception as exc:
        row.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        actor.pool.shutdown(wait=True)
    after = (memory.calls,memory.input_tokens,memory.output_tokens) if memory else (0,0,0)
    generations = [g for e in episodes for g in e['generations']]
    row.update(attempts=len(episodes), actor_calls=len(generations),
        actor_input_tokens=sum(g['usage']['prompt_tokens'] for g in generations),
        actor_output_tokens=sum(g['usage']['completion_tokens'] for g in generations),
        writer_calls=after[0]-before[0],writer_input_tokens=after[1]-before[1],writer_output_tokens=after[2]-before[2],
        writer_token_limit_hits=(memory.limit_hits if memory else 0)-limits_before,
        initial_query_sha256=hashlib.sha256(observation.encode()).hexdigest(),
        first_prompt_sha256=generations[0]['rendered_prompt_sha256'] if generations else None)
    if memory:
        memory.snapshot(output/'memory_after.json')
        row['memory_after_sha256']=sha(output/'memory_after.json')
    save(output/'row.json',row)
    return row


def cl_cell(plan, client, memory, name, index, arm, output):
    from ttcl.icl_mem0_comparison.worker import System, make_task, base
    from ttcl.icl_mem0_comparison.protocol import normalize_prompt
    output.mkdir(parents=True,exist_ok=False)
    task = make_task(name,plan['task_seed'])
    query = task.reset_baseline_instance(index)
    brief = task.get_agent_brief()
    brief = base.format_task_agent_brief(brief) if brief else ''
    system = System(plan,client,'none',[],None,output,brief,index,plan['tasks'][name])
    # Reuse the common interaction loop, but preserve the actual arm in native
    # task artifacts and system names as well as in our outer row record.
    system.arm=arm;system.mode=arm
    task_query = normalize_prompt(query.prompt,brief,index,plan['tasks'][name])
    retrieval = memory.retrieve(task_query) if memory else {'context':'','ids':[]}
    save(output/'retrieval.json',retrieval)
    if retrieval['context']:
        system.messages[0]['content'] += '\n\nPast experience:\n'+retrieval['context']
    if memory is not None and hasattr(memory, 'decorate_system'):
        memory.decorate_system(system)
    recorder = base.Recorder(output,system,1)
    before = (memory.calls,memory.input_tokens,memory.output_tokens) if memory else (0,0,0)
    limits_before=memory.limit_hits if memory else 0
    record = dict(benchmark='clbench',task=name,repeat=client.repeat,arm=arm,canonical_index=index,
        episode=index+1,instance_id=query.instance_id,status='failed',reward=None,
        initial_query_sha256=hashlib.sha256(query.prompt.encode()).hexdigest())
    try:
        result = base.run_task(task,system,trace_recorder=recorder,show_progress=False,
                               reset_system=False,initial_query=query)
        if len(result.instance_outcomes)!=1:
            raise ValueError('Expected one official outcome')
        outcome = result.instance_outcomes[0]
        record.update(status='complete',reward=float(outcome.reward),success=outcome.success)
        # Scalar feedback goes only to the Q updater. Writer sees public trace.
        if memory:
            trace = json.dumps(system.public_steps,ensure_ascii=False)
            update = memory.update(task_query,trace,float(outcome.reward),outcome.success,retrieval,
                {'initial_query_sha256':record['initial_query_sha256'],
                 'public_content_sha256':digest(system.public_steps),'task':name,'index':index,'repeat':client.repeat})
            save(output/'update.json',update)
            record['memory_update_status'] = 'complete'
    except Exception as exc:
        if record['status']=='complete':
            record['memory_update_status'] = 'failed'
        record.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        connection = getattr(task,'_conn',None)
        if connection is not None:connection.close()
    after = (memory.calls,memory.input_tokens,memory.output_tokens) if memory else (0,0,0)
    record.update(actor_calls=system.calls,actor_input_tokens=system.input_tokens,
        actor_output_tokens=system.output_tokens,max_input_tokens=system.max_input_tokens,
        writer_calls=after[0]-before[0],writer_input_tokens=after[1]-before[1],writer_output_tokens=after[2]-before[2],
        writer_token_limit_hits=(memory.limit_hits if memory else 0)-limits_before)
    save(output/'public_trajectory.json',system.public_steps)
    if memory:
        memory.snapshot(output/'memory_after.json')
        record['memory_after_sha256']=sha(output/'memory_after.json')
    events = output/'responses.jsonl'
    if events.exists():record['first_prompt_sha256']=json.loads(events.open().readline())['rendered_prompt_sha256']
    save(output/'row.json',record)
    return record


def work(root, benchmark, task, repeat):
    plan=read(root/'plan.json');os.chdir(os.environ['TTCL_BENCH'])
    directory=root/'runs'/benchmark/task/str(repeat)
    directory.mkdir(parents=True,exist_ok=True)
    client=Client(plan,repeat)
    memory=Memory(dict(plan,q_min_threshold=plan['q_min_thresholds'][benchmark]),client,directory/'memory',
                  plan['calibration'][benchmark if benchmark=='alfworld' else task])
    items=(next(s['tasks'] for s in plan['alf']['sequences'] if s['family']==task and s['repeat']==repeat)
           if benchmark=='alfworld' else list(range(plan['tasks'][task])))
    try:
        for index,item in enumerate(items):
            rows=[]
            for arm in plan['arms']:
                output=directory/arm/f'episode_{index+1:03d}'
                save(directory/'progress.json',dict(phase='running',index=index,arm=arm,updated_at=time.time()))
                if (output/'row.json').exists():
                    row=read(output/'row.json')
                    if (row['benchmark'],row['task'],row['repeat'],row['arm'])!=(benchmark,task,repeat,arm):
                        raise ValueError('Completed cell identity changed')
                    if arm=='memrl':
                        if sha(output/'memory_after.json')!=row['memory_after_sha256']:
                            raise ValueError('Memory snapshot changed')
                        memory.restore(output/'memory_after.json')
                else:
                    row=(alf_cell(plan,client,memory if arm=='memrl' else None,item,repeat,arm,output)
                         if benchmark=='alfworld' else
                         cl_cell(plan,client,memory if arm=='memrl' else None,task,item,arm,output))
                rows.append(row)
                print(json.dumps({k:row.get(k) for k in ('benchmark','task','repeat','arm','episode','status','reward','error')}),flush=True)
                if row.get('memory_update_status')=='failed':raise RuntimeError(row['error'])
                if row['status']!='complete' and not (benchmark=='clbench' and any(x in row.get('error','') for x in
                    ('no schema-valid JSON action','Safety cap exceeded','Context overflow'))):
                    raise RuntimeError(row.get('error','Unscored infrastructure failure'))
            if len({r['initial_query_sha256'] for r in rows})!=1:raise ValueError('Paired task mismatch')
            if index==0 and len({r.get('first_prompt_sha256') for r in rows})!=1:
                raise ValueError('Empty-memory initial prompts differ')
        save(directory/'progress.json',dict(phase='complete',updated_at=time.time()))
    except Exception as exc:
        save(directory/'progress.json',dict(phase='failed',error=repr(exc),traceback=traceback.format_exc(),updated_at=time.time()))
        raise
