from __future__ import annotations
import copy
import json
import re
import requests
from .common import binding,seed
from .judge import constants
from .model import FullPolicy
from .upstream import module


class Roles:
    def __init__(self,paths):
        self.paths=dict(paths);self.models={};self.active=None

    def get(self,role):
        if self.active!=role:
            if self.active is not None:self.models[self.active].to('cpu')
            if role not in self.models:self.models[role]=FullPolicy(self.paths[role],device='cpu')
            self.models[role].to('cuda:0');self.active=role
        return self.models[role]

    def replace(self,role,path):
        if role in self.models:
            self.models[role].to('cpu');del self.models[role]
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
        if self.active==role:self.active=None
        self.paths[role]=str(path)


def task_text(row):
    return row['question']+('\n\nImage description supplied by the official dataset:\n'+row['caption'] if row.get('caption') else '')


def answer(text):
    found=re.findall(r'<answer>(.*?)</answer>',text,re.S)
    return found[-1].strip() if found else ''


class TextExecutor:
    def __init__(self,roles,retrieval_url,settings):
        self.roles,self.url,self.settings=roles,retrieval_url,settings
        self.prompt=module('Executor-Train/Train/local_search/prompt.py','apex_exec_prompt').PROMPT_TEXT_ONLY
        self.session=requests.Session();self.session.trust_env=False

    def tool(self,text,tokenizer):
        calls=re.findall(r'<tool_call>\s*({.*?})\s*</tool_call>',text,re.S)
        if len(calls)!=1:return 'Tool error: emit exactly one search tool_call with query_list.',False
        try:
            call=json.loads(calls[0]);args=call.get('arguments',{})
            if call.get('name')!='search' or set(args)!={'query_list'}:raise ValueError('Only search(query_list) is available in the caption-based text adaptation')
            queries=args['query_list']
            if not isinstance(queries,list) or not queries or len(queries)>10 or not all(isinstance(q,str) and q.strip() for q in queries):raise ValueError('query_list must contain 1 to 10 nonempty strings')
        except (ValueError,TypeError) as exc:return 'Tool error: '+str(exc),False
        response=self.session.post(self.url+'/retrieve',json={'queries':queries,'topk':3,'return_scores':True},timeout=600)
        response.raise_for_status();results=response.json()['result']
        snippets=[]
        for query,docs in zip(queries,results):
            snippets.append('Search query: '+query)
            snippets.extend(f"Doc {i+1}: {d['document']['contents']}" for i,d in enumerate(docs))
        raw='\n\n'.join(snippets);ids=tokenizer.encode(raw,add_special_tokens=False)
        limit=self.settings['max_tool_tokens']
        if len(ids)>limit:raw=tokenizer.decode(ids[:limit])+f'\n[tool result truncated: {len(ids)-limit} tokens omitted]'
        return raw,True

    def run(self,row,plans,random_seed,initial=None):
        model=self.roles.get('executor');cfg=self.settings
        states=[]
        for i,plan in enumerate(plans):
            messages=copy.deepcopy(initial[i]['messages']) if initial else [
                {'role':'system','content':self.prompt},
                {'role':'user','content':task_text(row)+'\n\nPlan to follow:\n'+plan+
                 '\n\nAvailable tool: <tool_call>{"name":"search","arguments":{"query_list":["query"]}}</tool_call>'}]
            if initial:messages.append({'role':'user','content':'I have a revised plan for you to follow:\n'+plan})
            states.append({'messages':messages,'events':[],'answer':'','done':False,'valid_tools':0,'stop_reason':None})
        for turn in range(cfg['max_turns']):
            pending=[i for i,s in enumerate(states) if not s['done']]
            if not pending:break
            # Different remaining budgets are batched only when identical.
            buckets={}
            for i in pending:
                s=states[i];remaining=cfg['max_response_tokens']-sum(e['output_tokens'] for e in s['events'])
                context_remaining=cfg['context']-len(model.render(s['messages']))
                budget=min(remaining,context_remaining)
                if budget<=0:s.update(done=True,stop_reason='token_budget');continue
                buckets.setdefault(budget,[]).append(i)
            for budget,indices in buckets.items():
                events=model.generate_batch([states[i]['messages'] for i in indices],seed(random_seed,turn,budget),
                    budget,cfg['context'],temperature=1.)
                for i,event in zip(indices,events):
                    s=states[i];s['events'].append(event);raw=event['raw_response']
                    s['messages'].append({'role':'assistant','content':raw})
                    if '<tool_call>' in raw:
                        feedback,valid=self.tool(raw,model.tokenizer)
                        s['valid_tools']+=int(valid)
                        s['messages'].append({'role':'user','content':'<tool_response>\n'+feedback+'\n</tool_response>'})
                    else:
                        s['answer']=answer(raw);s.update(done=True,stop_reason='answer' if s['answer'] else 'invalid_answer')
        for s in states:
            if not s['done']:s['stop_reason']='turn_budget'
            s['trace']='\n'.join(m['role']+': '+m['content'] for m in s['messages'])
            s['solution']='\n'.join(m['content'] for m in s['messages'] if m['role']=='assistant')
            s['trajectory_binding']=binding(s['messages'])
        return states


class Planner:
    def __init__(self,roles,settings):
        self.roles,self.settings=roles,settings
        self.prompts=module('Planner-Train/mem-plan/local_search/prompt.py','apex_plan_prompt')
        self.decision=constants('Memory-Serve/memory_serve.py',['judge_prompt'])['judge_prompt']

    def messages(self,question,memory='',skill=None,tools=None):
        text=self.prompts.PLAN_PROMPT.format(question=question,memory=memory or '(no prior memory)',
            skill=json.dumps(skill,ensure_ascii=False) if skill else '(no reliable skill)')
        if tools:
            text=text.replace('- `search`: perform text-based web queries to retrieve external information.',tools)
        return [{'role':'system','content':self.prompts.SYSTEM_PLAN_PROMPT},{'role':'user','content':text}]

    def generate(self,messages,random_seed,temperature=1.,max_tokens=None):
        model=self.roles.get('planner');c=self.settings
        budget=c['max_response_tokens'] if max_tokens is None else min(max_tokens,c['max_response_tokens'])
        budgets=[min(budget,c['context']-len(model.render(m))) for m in messages]
        if min(budgets)<=0:raise ValueError('Planner context exhausted')
        return model.generate_batch(messages,random_seed,min(budgets),c['context'],temperature)

    def bounded_trace(self,trace):
        tokenizer=self.roles.get('planner').tokenizer
        ids=tokenizer.encode(trace,add_special_tokens=False)
        limit=self.settings.get('max_tool_tokens',8192)
        return tokenizer.decode(ids[:limit]) if len(ids)>limit else trace

    def loop(self,row,memory,skill,executor,judge,group,random_seed):
        inputs=self.messages(task_text(row),memory,skill)
        plans=self.generate([inputs]*group,random_seed)
        first=executor.run(row,[e['raw_response'] for e in plans],seed(random_seed,'executor1'))
        decision_inputs=[inputs+[{'role':'assistant','content':p['raw_response']},
            {'role':'user','content':self.decision.format(trace=self.bounded_trace(s['trace']))}] for p,s in zip(plans,first)]
        output=[]
        for i,(p,s) in enumerate(zip(plans,first)):
            remaining=self.settings['max_response_tokens']-p['output_tokens']
            events=[p];second=s;replan=None;choice='';d=None
            if remaining>0:
                d=self.generate([decision_inputs[i]],seed(random_seed,'decisions',i),max_tokens=remaining)[0]
                events.append(d);remaining-=d['output_tokens']
                choice=d['raw_response'].split('</think>')[-1].strip().lower()
            if choice=='yes' and remaining>0:
                msg=decision_inputs[i]+[{'role':'assistant','content':d['raw_response']},
                    {'role':'user','content':self.prompts.REPLAN_PROMPT.format(question=task_text(row))}]
                replan=self.generate([msg],seed(random_seed,'replan',i),max_tokens=remaining)[0];events.append(replan)
                second=executor.run(row,[replan['raw_response']],seed(random_seed,'executor2',i),initial=[s])[0]
            acc1=judge.correct(task_text(row),s['answer'],row['target'])
            acc2=judge.correct(task_text(row),second['answer'],row['target']) if replan else acc1
            fmt=float(choice in {'yes','no'} and all(e['raw_response'].count('<think>')==e['raw_response'].count('</think>') for e in events))
            decision_score=float((acc1 and choice=='no') or (not acc1 and choice=='yes'))
            reward=.7*acc2+.2*acc1+.05*fmt+.05*decision_score
            output.append({'events':events,'reward':reward,'acc1':acc1,'acc2':acc2,'decision':choice,
                'format_score':fmt,'decision_score':decision_score,'first_execution':s,'last_execution':second,
                'plan':p['raw_response'],'replan':replan['raw_response'] if replan else None})
        return output
