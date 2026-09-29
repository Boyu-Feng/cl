from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import gzip
import json
from pathlib import Path
import random
import time
import traceback
from .common import append,binding,read,save,seed,status,workspace
from .data import load_rows
from .judge import Judge
from .memory import MemoryBank
from .rollouts import Roles,TextExecutor,Planner,task_text
from .upstream import module


def archive(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Refusing to overwrite captured rollouts')
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False)


def trace_row(row):
    return {k:row[k] for k in ['id','source','input_binding','supervision_binding']}


def frozen_plan(root,row,planner,random_seed):
    path=root/'frozen_initial_plans'/(row['input_binding']+'.json')
    if path.exists():return read(path)['raw_response']
    messages=planner.messages(task_text(row))
    result=planner.generate([messages],random_seed,temperature=0.)[0]
    save(path,dict(result,source_row=trace_row(row)));return result['raw_response']


def evolve_skills(bank,roles,prompts,parse,root,label):
    grouped=defaultdict(list)
    for e in bank.entries:grouped[(e['category'],e['modality'])].append(e)
    for (category,modality),memories in sorted(grouped.items()):
        current=None
        for start in range(0,len(memories),10):
            batch=[dict(e,status='new',question=e['question'][:50],workflow_summary=e['workflow_summary'][:200]) for e in memories[start:start+10]]
            messages=prompts.build_writer_prompt(category,modality,json.dumps(current) if current else 'null',batch)
            model=roles.get('writer');ids=model.render(messages)
            if len(ids)>4096:
                append(root/'skill_failures.jsonl',{'label':label,'category':category,'start':start,'reason':'writer_prompt_budget','input_binding':binding(messages)})
                continue
            event=model.generate(messages,seed(label,category,modality,start),2048,6144,0.)
            parsed,valid=parse(event['raw_response'])
            archive(root/'skills'/label/category/modality/f'{start:06}.json.gz',{'event':event,'memories':batch,'input_binding':binding(messages)})
            if valid and isinstance(parsed,dict) and isinstance(parsed.get('skill'),dict):
                if parsed.get('operation')=='skip' and current:continue
                complementary=parsed.get('operation')=='create' and bool(current)
                proposed=parsed['skill']
                supported=memories[:start+len(batch)]
                proposed.update(category=category,modality=modality,evidence_count=len(supported),
                    win_rate=sum(e['judgement']=='correct' for e in supported)/max(1,len(supported)))
                if complementary:bank.skills[category+'_'+modality+'_v2']=proposed
                else:current=proposed
        if current:bank.skills[category+'_'+modality]=current
    bank.save(root/(label+'_memory.json'))


def build_writer_data(root,bank,roles,judge):
    builder=module('Writer-Train/Train/writer_skill/build_writer_data.py','apex_writer_data')
    ordered=sorted(bank.entries,key=lambda e:binding(['writer_holdout',e['data_id']]))
    n=max(1,int(.2*len(ordered)));validation=ordered[:n];training=ordered[n:]
    # Skills used as old inputs are built only from the writer training histories.
    original=bank.entries;bank.entries=training
    evolve_skills(bank,roles,judge.writer_prompts,judge.writer._parse_skill_json,root,'initial_writer_skills')
    def build(entries,split):
        store={mod:{cat:[] for cat in builder.CATEGORIES} for mod in builder.MODALITIES}
        for e in entries:
            store[e['modality']][e['category']].append(dict(e,status='new',
                question=builder.truncate(e['question'],builder.MAX_QUESTION_LEN),
                workflow_summary=builder.truncate(e['workflow_summary'],builder.MAX_WORKFLOW_LEN)))
        samples=builder.batch_memories(store,bank.skills,seed=seed('writer_split',split))
        accepted=[]
        model=roles.get('writer')
        for item in samples:
            mem=json.loads(item['memories'])
            messages=judge.writer_prompts.build_writer_prompt(item['category'],item['modality'],item['current_skill'],mem)
            item['input_binding']=binding(messages);item['data_id']='writer_'+item['input_binding']
            item['source_bindings']=[m['source_binding'] for m in mem]
            if len(model.render(messages))>4096:
                append(root/'writer_exclusions.jsonl',{'id':item['data_id'],'reason':'official_4096_prompt_filter'});continue
            accepted.append(item)
        return accepted
    train=build(training,'train');val=build(validation,'validation')
    if not train:raise ValueError('No valid fresh Writer training batches')
    save(root/'writer_dataset.json',{'train':train,'validation':val,
        'training_history_ids':[e['data_id'] for e in training],
        'validation_history_ids':[e['data_id'] for e in validation],
        'fresh_supervision':'Official quality/evolution reward applied to newly bound input and generated skill.'})
    # Planner's retrieval bank contains only histories admitted to writer training.
    bank.entries=training;bank.vectors=[bank.encode(e['question']) for e in training]
    bank.caption_vectors=[bank.encode(e['image_caption']) for e in training]
    bank.save(root/'writer_training_memory.json')
    return train


def train(root):
    import torch
    torch.set_num_threads(8)
    plan=read(root/'plan.json');dataset=read(root/'dataset.json');data=Path(dataset['directory'])
    model_root=Path(plan['model_output']);model_root.mkdir(parents=True,exist_ok=True)
    roles=Roles({r:plan['base_model'] for r in ['executor','writer','planner']})
    judge=Judge(plan['judge_url'],root/'judge_calls.jsonl')
    executor=TextExecutor(roles,plan['retrieval_url'],plan['executor'])
    planner=Planner(roles,plan['planner'])
    executor_data=load_rows(data/'executor_train.jsonl');planner_data=load_rows(data/'planner_train.jsonl')
    for role,rows in [('executor',executor_data),('writer',None),('planner',planner_data)]:
        if role=='writer':
            status(root,'collecting_fresh_executor_memories',expected=len(executor_data))
            bank=MemoryBank(judge)
            for i,row in enumerate(executor_data):
                p=frozen_plan(root,row,planner,seed('frozen_plan',row['id']))
                result=executor.run(row,[p],seed('writer_collection',row['id']))[0]
                correct=judge.correct(task_text(row),result['answer'],row['target'])
                bank.add(row['question'],row['caption'],result['trace'],correct,p,memory_id='fresh_'+binding([row['input_binding'],result['trajectory_binding']]))
                archive(root/'fresh_memories'/f'{i:06}.json.gz',{'source':trace_row(row),'rollout':result,'correct':correct})
                status(root,'collecting_fresh_executor_memories',completed=i+1,expected=len(executor_data))
                if (i+1)%100==0:bank.save(root/'collected_memory_progress.json')
            bank.save(root/'collected_memory.json')
            rows=build_writer_data(root,bank,roles,judge)
        if role=='planner':executor.settings=dict(plan['executor'],max_tool_tokens=plan['planner']['max_tool_tokens'])
        cfg=plan[role];step=0
        for epoch in range(cfg['epochs']):
            order=list(range(len(rows)));random.Random(seed('epoch_order',role,epoch)).shuffle(order)
            for start in range(0,len(order),cfg['batch_size']):
                batch=[];group_rows=order[start:start+cfg['batch_size']]
                for j,index in enumerate(group_rows):
                    row=rows[index];group_id=f'{role}:{epoch}:{start}:{index}'
                    status(root,'collecting_training_rollouts',role=role,epoch=epoch+1,epochs=cfg['epochs'],
                        batch=step+1,batch_queries_completed=j,batch_queries=len(group_rows),
                        completed_epochs_queries=epoch*len(rows)+start+j,queries_per_epoch=len(rows),
                        total_updates_completed=roles.models.get(role).updates if role in roles.models else 0)
                    if role=='executor':
                        p=frozen_plan(root,row,planner,seed('frozen_plan',row['id']))
                        results=executor.run(row,[p]*cfg['group_size'],seed(role,epoch,row['id']))
                        for t in results:
                            t['accuracy']=judge.correct(task_text(row),t['answer'],row['target'])
                            t['reward']=judge.executor_score(row['source'],t['solution'],row['target'],{'acc_score':float(t['accuracy'])})
                        source=trace_row(row)
                    elif role=='writer':
                        mem=json.loads(row['memories'])
                        messages=judge.writer_prompts.build_writer_prompt(row['category'],row['modality'],row['current_skill'],mem)
                        events=roles.get('writer').generate_batch([messages]*cfg['group_size'],seed(role,epoch,row['data_id']),
                            cfg['max_response_tokens'],cfg['context'],1.)
                        results=[{'events':[event],'reward':judge.quality(event['raw_response'],row)} for event in events]
                        source={'input_binding':row['input_binding'],'source_bindings':row['source_bindings']}
                    else:
                        memory,skill,used,category,modality=bank.retrieve(row['question'],row['caption'],row['input_binding'])
                        results=planner.loop(row,memory,skill,executor,judge,cfg['group_size'],seed(role,epoch,row['id']))
                        source=dict(trace_row(row),used_memory_indices=used)
                    for t in results:t['group_id']=group_id
                    archive(root/'rollouts'/role/f'epoch_{epoch:02}'/f'group_{start+j:06}.json.gz',
                            {'source':source,'group_id':group_id,'trajectories':results})
                    batch.extend({'events':t['events'],'reward':t['reward'],'group_id':group_id} for t in results)
                status(root,'optimizing_full_parameters',role=role,epoch=epoch+1,batch=step+1,
                       trajectories=len(batch),queries_per_epoch=len(rows),learning_rate=cfg['learning_rate'])
                result=roles.get(role).update(batch,cfg['learning_rate'],root/'updates'/role/f'{step:06}.json')
                append(root/'metrics.jsonl',dict(result,role=role,epoch=epoch+1,batch=step+1))
                print(json.dumps(dict(result,role=role,epoch=epoch+1,batch=step+1)),flush=True)
                del batch;step+=1
        path=model_root/role
        roles.get(role).save_model(path)
        save(root/(role+'_complete.json'),{'checkpoint':str(path),'batches':step,'epochs':cfg['epochs'],
            'fixed_final_checkpoint':True,'no_selection_on_clbench':True,'time':time.time()})
        roles.replace(role,path)
        if role=='writer':
            bank.skills={}
            evolve_skills(bank,roles,judge.writer_prompts,judge.writer._parse_skill_json,root,'trained_writer_skills')
            bank.save(root/'final_offline_memory.json')
    save(root/'training_complete.json',{'models':{r:str(model_root/r) for r in ['executor','writer','planner']},
         'full_parameter':True,'clbench_used_for_offline_training':False,'time':time.time()})
    status(root,'offline_training_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    try:train(root)
    except BaseException as exc:
        status(root,'training_failed',error=repr(exc),traceback=traceback.format_exc());raise
