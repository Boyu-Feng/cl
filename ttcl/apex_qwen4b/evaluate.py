"""Pre-update CLBench scoring with matched four-rollout environment budgets."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import append,binding,read,save,seed,status,workspace
from .judge import Judge
from .memory import MemoryBank
from .rollouts import Roles,Planner
from .train import archive
from ttcl.structured_memory import run_benchmark as base
from ttcl.icl_mem0_comparison.protocol import normalize_prompt


def make_task(name,random_seed):
    if name=='blind_spectrum_monitoring':
        from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
        return BlindSpectrumMonitoringTask(seed=random_seed,schedule='default',response_timeout_seconds=0)
    if name=='sales_prediction':
        from src.tasks.sales_prediction.task import SalesPredictionTask
        return SalesPredictionTask(seed=random_seed,schedule='default',clean_workspace_between_instances=True)
    return base.make_task(name,random_seed,independent=True)


class System(base.ContinualLearningSystem):
    def __init__(self,roles,planner,bank,arm,settings,repeat,index,candidate,count,output,brief):
        self.roles,self.planner,self.bank,self.arm=roles,planner,bank,arm
        self.settings,self.repeat,self.index,self.candidate,self.count=settings,repeat,index,candidate,count
        self.output,self.brief=output,brief;self.mode=arm
        self.calls=self.input_tokens=self.output_tokens=self.max_input_tokens=self.turn=0
        self.messages=[{'role':'system','content':
            'Solve the task using only its supplied action schema and public tool feedback. '
            'Return one JSON action. Do not invent tool results or call tools absent from the schema.\n'+brief}]
        self.public_steps=[];self.planner_events=[];self.plan='';self.planner_messages=[]
        self.question='';self.used=[];self.skill=None;self.last=None;self.replanned=False;self.format_messages=[]

    @property
    def name(self):return 'apex_qwen4b_'+self.arm
    def reset(self):raise RuntimeError('Episode state is constructed explicitly')
    def _record_plan(self,event):
        self.planner_events.append(event)
        append(self.output/'planner_events.jsonl',event)

    def respond(self,query):
        self.turn+=1
        if self.turn>self.settings['max_turns']:raise RuntimeError('Safety cap exceeded; no official outcome')
        prompt=normalize_prompt(query.prompt,self.brief,self.index,self.count)
        schema=json.dumps(query.response_schema.model_json_schema(),ensure_ascii=False,sort_keys=True)
        if not self.question:
            self.question=prompt
            if self.arm!='base':
                context,self.skill,self.used,_,_=self.bank.retrieve(prompt)
                self.planner_messages=self.planner.messages(prompt,context,self.skill,
                    tools='Use only the native task tools described by the task instructions and this action schema:\n'+schema)
                event=self.planner.generate([self.planner_messages],seed('cl_plan',self.repeat,self.index,self.candidate))[0]
                self._record_plan(event);self.plan=event['raw_response'];self.format_messages=[self.plan]
                self.messages[0]['content']+='\n\nPlan:\n'+self.plan
                self.planner_messages.append({'role':'assistant','content':self.plan})
        elif self.arm!='base' and self.turn==9 and not self.replanned:
            # At most one replan from already observed public feedback, within the same task.
            trace=self.planner.bounded_trace(json.dumps(self.public_steps,ensure_ascii=False))
            budget=self.settings['planner_response_budget']-sum(e['output_tokens'] for e in self.planner_events)
            if budget>0:
                msg=self.planner_messages+[{'role':'user','content':self.planner.decision.format(trace=trace)}]
                event=self.planner.generate([msg],seed('cl_decision',self.repeat,self.index,self.candidate),max_tokens=budget)[0]
                self._record_plan(event);self.format_messages.extend([trace,event['raw_response']]);msg.append({'role':'assistant','content':event['raw_response']})
                budget-=event['output_tokens']
                if event['raw_response'].split('</think>')[-1].strip().lower()=='yes' and budget>0:
                    msg.append({'role':'user','content':self.planner.prompts.REPLAN_PROMPT.replace('**The only tool you can recommend is `search`.**','Use only the native task tools and the supplied action schema.').format(question=self.question)})
                    event=self.planner.generate([msg],seed('cl_replan',self.repeat,self.index,self.candidate),max_tokens=budget)[0]
                    self._record_plan(event);self.plan=event['raw_response'];self.format_messages.extend(['replan',self.plan,trace])
                    self.messages.append({'role':'user','content':'Updated plan based on observed feedback:\n'+self.plan})
            self.replanned=True
        self.messages.append({'role':'user','content':prompt+'\n\nReturn only JSON. Action schema:\n'+schema})
        role='base' if self.arm=='base' else 'executor'
        for retry in range(1+self.settings['action_retries']):
            model=self.roles.get(role)
            event=model.generate(self.messages,seed('cl_actor',self.repeat,self.index,self.candidate,self.turn,retry),
                self.settings['max_action_tokens'],self.settings['actor_context'],1.)
            self.calls+=1;self.input_tokens+=event['input_tokens'];self.output_tokens+=event['output_tokens']
            self.max_input_tokens=max(self.max_input_tokens,event['input_tokens'])
            self.messages.append({'role':'assistant','content':event['raw_response']})
            try:
                action,repair=base.normalize_action(event['raw_response'],query.response_schema)
                append(self.output/'responses.jsonl',dict(event,action=action.model_dump(),packaging_repair=repair,turn=self.turn,format_retry=retry))
                self.last={'query':prompt,'action':action.model_dump()}
                return base.Response(action=action)
            except ValueError as exc:
                append(self.output/'responses.jsonl',dict(event,parse_error=str(exc),turn=self.turn,format_retry=retry))
                if retry==self.settings['action_retries']:raise
                self.messages.append({'role':'user','content':'Formatting error: return one JSON object matching the exact schema. No action was executed.'})

    def observe(self,observation,next_query=None):
        complete=base.observation_marks_instance_complete(observation)
        step={'turn':self.turn,'public_feedback':observation.content,'instance_complete':complete}
        if self.last:step.update(self.last)
        self.public_steps.append(step);append(self.output/'public_observations.jsonl',step)
        if observation.content:self.messages.append({'role':'user','content':'Tool feedback:\n'+observation.content})

    def get_run_artifacts(self):return {'public_steps':self.public_steps,'plan':self.plan}


def episode(root,plan,roles,planner,bank,arm,domain,repeat,index,candidate):
    cfg=plan['clbench'];output=root/'clbench'/domain/str(repeat)/arm/f'episode_{index:03}'/f'candidate_{candidate}'
    output.mkdir(parents=True,exist_ok=False)
    task=make_task(domain,plan['task_seed']);query=task.reset_baseline_instance(index)
    brief=task.get_agent_brief();brief=base.format_task_agent_brief(brief) if brief else ''
    system=System(roles,planner,bank,arm,cfg,repeat,index,candidate,cfg['tasks'][domain],output,brief)
    recorder=base.Recorder(output,system,1)
    row={'task':domain,'repeat':repeat,'arm':arm,'canonical_index':index,'candidate':candidate,
        'initial_query_binding':binding(query.prompt if query else None),
        'status':'failed','reward':None,'success':None,'parameter_updates_before':roles.models.get('planner').updates if 'planner' in roles.models else 0}
    try:
        result=base.run_task(task,system,trace_recorder=recorder,show_progress=False,reset_system=False,initial_query=query)
        if len(result.instance_outcomes)!=1:raise ValueError('Expected exactly one official outcome')
        outcome=result.instance_outcomes[0];row.update(status='complete',reward=float(outcome.reward),success=outcome.success,instance_id=outcome.instance_id)
    except Exception as e:
        row.update(error=repr(e),traceback=traceback.format_exc())
        if not any(x in str(e) for x in ['no schema-valid JSON action','Safety cap exceeded','Declared generation context exceeded','Planner context exhausted']):
            save(output/'row.json',row);raise
    finally:
        connection=getattr(task,'_conn',None)
        if connection is not None:connection.close()
        # Follow the official task cleanup hook where provided.
        close=getattr(task,'close',None)
        if callable(close):close()
    row.update(actor_calls=system.calls,actor_input_tokens=system.input_tokens,actor_output_tokens=system.output_tokens,
        planner_calls=len(system.planner_events),planner_output_tokens=sum(e['output_tokens'] for e in system.planner_events),
        elapsed_seconds=recorder.metrics()['elapsed_seconds'])
    save(output/'row.json',row)
    # This is the only payload passed to learning; evaluator outcomes are excluded.
    public={'question':system.question,'trace':json.dumps(system.public_steps,ensure_ascii=False),
        'answer':json.dumps(system.last['action'],ensure_ascii=False) if system.last else '',
        'plan':system.plan,'planner_responses':system.format_messages,'skill':system.skill,'used':system.used,'planner_events':system.planner_events,
        'messages':system.messages,'public_binding':binding(system.public_steps)}
    save(output/'public_learning_input.json',public)
    return row,public,output


def online_skill(bank,roles,judge,category,modality,output):
    memories=[dict(e,status='new',question=e['question'][:50],workflow_summary=e['workflow_summary'][:200]) for e in bank.entries
              if e['category']==category and e['modality']==modality][-15:]
    current=bank.skills.get(category+'_'+modality)
    messages=judge.writer_prompts.build_writer_prompt(category,modality,json.dumps(current) if current else 'null',memories)
    model=roles.get('writer')
    if len(model.render(messages))>4096:
        save(output,{'status':'skipped','reason':'writer_prompt_budget','input_binding':binding(messages)});return
    event=model.generate(messages,seed('cl_skill',binding(messages)),2048,6144,0.)
    parsed,valid=judge.writer._parse_skill_json(event['raw_response'])
    updated=bank.update_skill(category,modality,parsed) if valid else False
    save(output,dict(event,updated=updated,source_bindings=[m['source_binding'] for m in memories]))


def chain(root,domain,repeat,arm):
    import torch
    torch.set_num_threads(8)
    plan=read(root/'plan.json');cfg=plan['clbench'];os.chdir(os.environ['TTCL_BENCH'])
    if not (root/'training_complete.json').exists():raise RuntimeError('Offline training incomplete')
    paths=read(root/'training_complete.json')['models'];paths['base']=plan['base_model']
    roles=Roles(paths);planner=Planner(roles,plan['planner'])
    judge=Judge(plan['judge_url'],root/'clbench'/domain/str(repeat)/arm/'judge_calls.jsonl')
    bank=MemoryBank(judge);bank.load(root/'final_offline_memory.json')
    progress=root/'clbench_status'/f'{domain}_{repeat}_{arm}.json';new_counts=defaultdict(int)
    for batch_start in range(0,cfg['tasks'][domain],cfg['batch_size']):
        updates=[];memories=[]
        for index in range(batch_start,min(batch_start+cfg['batch_size'],cfg['tasks'][domain])):
            for candidate in range(cfg['group_size']):
                save(progress,{'phase':'evaluating_pre_update','canonical_index':index,'candidate':candidate,'time':time.time()})
                row,public,directory=episode(root,plan,roles,planner,bank,arm,domain,repeat,index,candidate)
                if candidate==0:
                    save(directory.parent/'result.json',dict(row,selection='candidate_zero_fixed_before_sampling',first_attempt=True))
                if arm!='base' and public['question']:
                    reward,feedback=judge.nogt(public['question'],public['trace'],public['answer'],public['plan'],public['skill'],public['planner_responses'])
                    save(directory/'online_reward.json',{'reward':reward,'feedback':feedback,'public_binding':public['public_binding']})
                    updates.append({'group_id':str(index),'events':public['planner_events'],'reward':reward})
                    memories.append((index,candidate,public,feedback['correct']))
                print(json.dumps({k:row.get(k) for k in ['task','repeat','arm','canonical_index','candidate','status','reward']}),flush=True)
        if arm=='apex_ttrl' and any(t['events'] for t in updates):
            save(progress,{'phase':'updating_planner_full_parameters','completed_through_index':index,'time':time.time()})
            result=roles.get('planner').update(updates,cfg['learning_rate'],
                root/'clbench'/domain/str(repeat)/arm/'updates'/f'{batch_start:03}.json')
            append(root/'online_metrics.jsonl',dict(result,domain=domain,repeat=repeat,arm=arm,batch_start=batch_start))
        # Consolidation happens only after all current-batch queries have been scored.
        for index,candidate,public,correct in memories:
            category,modality=bank.add(public['question'],'',public['trace'],correct,public['plan'],public['used'],
                                      memory_id=binding([domain,repeat,arm,index,candidate,public['public_binding']]))
            key=(category,modality);new_counts[key]+=1
        for key,count in list(new_counts.items()):
            if count>=5:
                category,modality=key
                online_skill(bank,roles,judge,category,modality,
                    root/'clbench'/domain/str(repeat)/arm/'skills'/f'{batch_start:03}_{category}_{modality}.json')
                new_counts[key]=0
        bank.save(root/'clbench'/domain/str(repeat)/arm/'memory_latest.json')
    save(progress,{'phase':'complete','tasks':cfg['tasks'][domain],'group_size':cfg['group_size'],'time':time.time()})


from collections import defaultdict
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--domain',required=True);p.add_argument('--repeat',type=int,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();chain(a.root.resolve(),a.domain,a.repeat,a.arm)
