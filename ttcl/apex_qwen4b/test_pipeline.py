"""Synthetic checks only; never train or select on a CLBench outcome."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from pydantic import BaseModel
from .common import read
from .judge import Judge
from .rollouts import Planner
from .memory import MemoryBank
from . import evaluate
from src.interface import Query,Observation,InstanceOutcome,TaskStepResult,standard_evaluate


class Action(BaseModel):
    answer:int


class SyntheticTask:
    def __init__(self):self.outcomes=[]
    def reset_baseline_instance(self,index):return Query('Return the integer 2.',Action,instance_id='synthetic',instance_index=index,metadata={'hidden':'NEVER_INCLUDE_THIS'})
    def get_agent_brief(self):return None
    def get_instance_outcomes(self):return self.outcomes
    def step(self,response):
        outcome=InstanceOutcome('synthetic',0,37.,True,metadata={'hidden':'NEVER_INCLUDE_THIS'})
        self.outcomes.append(outcome)
        return TaskStepResult(Observation('Action accepted.',metadata={'hidden':'NEVER_INCLUDE_THIS'}),None,True,outcome)
    def evaluate(self):return standard_evaluate(self.outcomes)


class FakeActor:
    def generate(self,messages,*args,**kwargs):
        assert 'NEVER_INCLUDE_THIS' not in json.dumps(messages)
        return dict(raw_response='{"answer":2}',input_tokens=10,output_tokens=5)


class PipelineTests(unittest.TestCase):
    def test_official_score_stays_out_of_learning_payload(self):
        roles=SimpleNamespace(models={},get=lambda role:FakeActor())
        config={'task_seed':42,'clbench':dict(tasks={'synthetic':1},max_turns=64,action_retries=2,max_action_tokens=32,actor_context=512)}
        with tempfile.TemporaryDirectory() as tmp,patch.object(evaluate,'make_task',return_value=SyntheticTask()):
            row,public,path=evaluate.episode(Path(tmp),config,roles,None,None,'base','synthetic',303,0,0)
            self.assertEqual(row['reward'],37.)
            self.assertEqual(row['status'],'complete')
            self.assertNotIn('NEVER_INCLUDE_THIS',json.dumps(public))
            self.assertNotIn('37.0',json.dumps(public))
            self.assertNotIn('reward',public)
            self.assertEqual(public['planner_events'],[])

    def test_online_reward_uses_official_format_check_without_ground_truth(self):
        judge=Judge('http://127.0.0.1:1/v1',Path('/tmp/not_called_apex_judge.jsonl'))
        judge.nogt_module=SimpleNamespace(evaluate_agent_without_gt=lambda *args:dict(raw_verdict='A',is_correct='correct'))
        score,result=judge.nogt('question','public trace','answer','plan',planner_responses=['<think>x</think>plan','trace','no'])
        self.assertAlmostEqual(score,1.)
        score,_=judge.nogt('question','public trace','answer','plan',planner_responses=['<think>unclosed','trace','no'])
        self.assertAlmostEqual(score,.9)
        score,_=judge.nogt('question','public trace','answer','plan')
        self.assertAlmostEqual(score,.9)
        with self.assertRaises(TypeError):judge.nogt('q','t','a','p',ground_truth='forbidden')

    def test_malformed_writer_structure_is_retained_as_failure_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            judge=Judge('http://127.0.0.1:1/v1',Path(tmp)/'judge.jsonl')
            extra={'current_skill':'{}','memories':'[]'}
            self.assertEqual(judge.quality('{"operation":"refine","skill":{"procedure":null}}',extra),0.)
            self.assertEqual(judge.quality('invalid json',extra),0.)
            self.assertIn('invalid_writer_structure',(Path(tmp)/'judge.jsonl').read_text())

    def test_planner_generation_honors_remaining_total_budget(self):
        requested=[]
        model=SimpleNamespace(render=lambda messages:[1,2,3],generate_batch=lambda messages,seed,tokens,context,temp:requested.append(tokens) or [])
        planner=Planner(SimpleNamespace(get=lambda role:model),{'max_response_tokens':8,'context':20})
        planner.generate([[{'role':'user','content':'x'}]],42,max_tokens=2)
        self.assertEqual(requested,[2])
        with self.assertRaises(ValueError):planner.generate([[{'role':'user','content':'x'}]],42,max_tokens=0)

    def test_skip_and_complementary_skill_do_not_overwrite_existing_skill(self):
        bank=MemoryBank.__new__(MemoryBank);old={'procedure':['old']};bank.skills={'process_text-only':old};bank.entries=[]
        self.assertFalse(bank.update_skill('process','text-only',{'operation':'skip','skill':{'procedure':['bad overwrite']}}))
        self.assertIs(bank.skills['process_text-only'],old)
        self.assertTrue(bank.update_skill('process','text-only',{'operation':'create','skill':{'procedure':['new']}}))
        self.assertIs(bank.skills['process_text-only'],old)
        self.assertEqual(bank.skills['process_text-only_v2']['procedure'],['new'])


if __name__=='__main__':unittest.main()
