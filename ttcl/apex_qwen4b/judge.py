from __future__ import annotations
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace
import time
from .common import append,binding
from .upstream import module,functions,source_root,writer_reward,executor_reward


def constants(relative,names):
    result={}
    for n in ast.parse((source_root()/relative).read_text()).body:
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in names:result[target.id]=ast.literal_eval(n.value)
    if set(result)!=set(names):raise ValueError(f'Missing constants: {set(names)-set(result)}')
    return result


class Judge:
    def __init__(self,url,log):
        from openai import OpenAI
        self.client=OpenAI(api_key='local-only',base_url=url,timeout=1800,max_retries=2)
        self.log=Path(log);self.failed=[]
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.prompts=module('Planner-Train/mem-plan/local_search/prompt.py','apex_planner_prompts')
        self.writer_prompts=module('Writer-Train/Train/writer_skill/prompt.py','apex_writer_prompts')
        self.writer=writer_reward(self,self.writer_prompts)
        self.executor_score=executor_reward()
        self.nogt_module=module('TTRL/TTRL-nogt/local_search/judge_nogt.py','apex_nogt')
        self.nogt_module.call_llm=self.call
        self.memory_prompts=constants('Memory-Serve/memory_serve.py',['classify_question_prompt','get_trace_prompt'])
        self.alignment_prompt=constants('TTRL/TTRL-nogt/local_search/mmsearch_skill.py',['SKILL_ALIGNMENT_PROMPT'])['SKILL_ALIGNMENT_PROMPT']
        self._categories={}

    def create(self,**kwargs):
        kwargs.setdefault('max_tokens',1024)
        kwargs.setdefault('extra_body',{'chat_template_kwargs':{'enable_thinking':False}})
        kwargs.setdefault('model','qwen')
        start=time.monotonic();event={'input':kwargs,'input_binding':binding(kwargs)}
        try:
            response=self.client.chat.completions.create(**kwargs)
            event.update(response=response.model_dump(),seconds=time.monotonic()-start)
            return response
        except Exception as e:
            event['error']=repr(e);self.failed.append(event);raise
        finally:append(self.log,event)

    def call(self,prompt_text,temperature=.0):
        return self.create(messages=[{'role':'user','content':prompt_text}],temperature=temperature).choices[0].message.content.strip()

    def correct(self,question,answer,target):
        if not answer.strip():return False
        prompt=self.prompts.JUDGE_PROMPT.format(question=question,correct_answer=str(target),response=answer)
        result=self.call(prompt,0.).split('</think>')[-1].strip()
        if result not in {'A','B','C'}:
            append(self.log,{'event':'invalid_accuracy_judgement','raw':result,'treated_as_incorrect_per_upstream':True})
        return result=='A'

    def quality(self,text,extra):
        parsed,valid=self.writer._parse_skill_json(text)
        skill=parsed.get('skill') if isinstance(parsed,dict) else None
        if valid and (not isinstance(parsed,dict) or (isinstance(skill,dict) and skill.get('procedure',[]) is None)):
            append(self.log,{'event':'invalid_writer_structure','input_binding':binding(text),'reward':0.})
            return 0.
        old=len(self.failed)
        result=self.writer.compute_score('writer_skill',text,'',extra)
        if len(self.failed)!=old:raise RuntimeError('Official judge transport failed; refusing rule-only fallback reward')
        return result

    def category(self,question):
        key=binding(question)
        if key not in self._categories:
            response=self.call(self.memory_prompts['classify_question_prompt'].format(question=question),0.)
            cats={'location','human','time','career','process','definition','event','function','property','others'}
            try:cat=json.loads(response)['answer'].strip().lower()
            except (ValueError,KeyError,TypeError):cat=response.strip().lower()
            self._categories[key]=cat if cat in cats else 'others'
        return self._categories[key]

    def summary(self,trace):
        return self.create(messages=[{'role':'user','content':self.memory_prompts['get_trace_prompt'].format(trace=trace)}],temperature=0.,max_tokens=4096).choices[0].message.content.strip()

    def nogt(self,question,trajectory,answer,plan,skill=None,planner_responses=None):
        # No target/score/metadata parameter exists on this boundary.
        result=self.nogt_module.evaluate_agent_without_gt(question,trajectory,answer)
        if result['raw_verdict'] not in {'A','B'}:raise ValueError('Invalid no-ground-truth arbiter verdict')
        correct=result['is_correct']=='correct'
        # The native CL task may terminate before a replan decision. Missing decisions
        # earn no format point; they are never fabricated as a 'no' response.
        fmt=functions('TTRL/TTRL-nogt/local_search/mmsearch_skill.py',
            ['extract_after_think','compute_format_score'])
        responses=planner_responses or []
        format_score=fmt.compute_format_score(responses) if len(responses) in (3,6) else 0.
        reward=.9*float(correct)+.1*format_score
        confidence=0.;alignment=None
        if skill and (skill.get('procedure') or skill.get('pitfalls')):
            confidence=float(skill.get('win_rate',0))*min(float(skill.get('evidence_count',0))/10,1)
            formatter=functions('TTRL/TTRL-nogt/local_search/mmsearch_skill.py',['_format_procedure','_format_pitfalls'])
            prompt=self.alignment_prompt.format(procedure=formatter._format_procedure(skill.get('procedure',[])),
                pitfalls=formatter._format_pitfalls(skill.get('pitfalls',[])),plan=plan)
            raw=self.call(prompt,0.).strip();alignment=float(raw)
            if not 0<=alignment<=1:raise ValueError('Invalid alignment judgement')
            coefficient=.1*confidence;reward=(1-coefficient)*reward+coefficient*alignment
        return reward,dict(result,confidence=confidence,alignment=alignment,format_score=format_score,
                           reward_source='nogt',correct=correct)
