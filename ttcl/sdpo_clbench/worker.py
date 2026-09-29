from __future__ import annotations
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

from .model import OnlineModel
from .protocol import ARMS, append, binding, read, save, selected_calls


def work(root, task_name, repeat):
    root=Path(root);plan=read(root/'plan.json')
    from ttcl.structured_memory import run_benchmark as base
    from ttcl.structured_memory.online_bank import run_episode
    original=base.make_task
    def make_task(name, random_seed, independent=False):
        if name=='blind_spectrum_monitoring':
            from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
            return BlindSpectrumMonitoringTask(seed=random_seed,schedule='default',response_timeout_seconds=0)
        return original(name,random_seed,independent)
    base.make_task=make_task;base.MEMORIES['blind_spectrum_monitoring']=base.DatabaseMemory
    os.chdir(os.environ['TTCL_BENCH'])
    directory=root/'runs'/task_name/str(repeat);directory.mkdir(parents=True,exist_ok=False)
    save(directory/'status.json',{'phase':'loading_model','time':time.time()})
    args=SimpleNamespace(task=task_name,seed=plan['task_seed'],num_instances=plan['tasks'][task_name],
        memory_chars=20000,max_turns_per_instance=64,action_retries=2,
        normalize_action=True,allow_initial_experience=True)
    try:
        model=OnlineModel(plan,repeat)
        save(directory/'initialization.json',{'base_fingerprint':model.base_fingerprint,
            'initial_adapter':model.initial_adapter,'trainable_parameters':sum(p.numel() for p in model.parameters),
            'zero_residual':True,'checkpoint_initialization':'public base only; no historical trained adapters'})
        for index in range(plan['tasks'][task_name]):
            paired=[]
            for arm in ARMS:
                model.arm=arm;model.captures=[]
                dest=directory/arm/f'episode_{index+1:03}'
                save(directory/'status.json',{'phase':'acting','index':index,'arm':arm,
                    'parameter_updates':model.updates,'time':time.time()})
                row,ep=run_episode(args,model,index,dest,'')
                events=[json.loads(line) for line in (dest/'responses.jsonl').read_text().splitlines()] if (dest/'responses.jsonl').exists() else []
                row.update(task=task_name,repeat=repeat,arm=arm,
                    parameter_updates_before=model.updates if arm!='frozen' else 0,
                    initial_query_sha256=binding(ep['initial_public_query']),
                    scored_before_update=True,scored_at=time.time(),
                    feedback_protocol='public observations plus completed-episode scalar; hidden score details excluded')
                save(dest/'row.json',row);paired.append(row)
                print(json.dumps({k:row.get(k) for k in ['task','repeat','arm','episode','reward','status','parameter_updates_before']}),flush=True)
                allowed=['no schema-valid JSON action','Safety cap exceeded','Context overflow']
                if row['status']!='complete' and not any(s in row.get('error','') for s in allowed):
                    raise RuntimeError('Infrastructure failure preserved: '+row.get('error',''))
                if arm=='sdpo_online':
                    if len(events)!=len(model.captures):raise AssertionError('Missing raw action captures')
                    save(dest/'action_capture.json',model.captures)
                    if index+1<plan['tasks'][task_name]:
                        save(directory/'status.json',{'phase':'updating_parameters','index':index,
                            'parameter_updates':model.updates,'time':time.time()})
                        # Sampling precedes and is independent of reward and length checks.
                        indices=selected_calls(events,repeat,index,plan['actions_per_update'])
                        update=model.update(model.captures,events,ep['steps'],row['reward'],indices,dest/'update')
                        append(directory/'updates.jsonl',dict(update,index=index,scored_at=row['scored_at']))
                    model.checkpoint(Path(plan['checkpoint_root'])/task_name/str(repeat),index)
            if len({(r['instance_id'],r['initial_query_sha256']) for r in paired})!=1:
                raise AssertionError('Paired tasks/initial queries differ')
            if index==0 and len({r['first_prompt_sha256'] for r in paired})!=1:
                raise AssertionError('Baseline and zero-LoRA first prompts differ')
        base_after=model.fingerprint('base')
        if base_after!=model.base_fingerprint:raise AssertionError('Base weights changed')
        checkpoint=Path(plan['checkpoint_root'])/task_name/str(repeat)
        model.model.save_pretrained(checkpoint/'final_adapters',selected_adapters=['default','teacher'])
        save(directory/'status.json',{'phase':'complete','records':plan['tasks'][task_name]*2,
             'parameter_updates':model.updates,'base_unchanged':True,
             'final_adapter':model.fingerprint('default'),'time':time.time()})
    except Exception as exc:
        save(directory/'status.json',{'phase':'failed','error':repr(exc),
            'traceback':traceback.format_exc(),'time':time.time()})
        raise
