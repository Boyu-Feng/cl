import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from pydantic import BaseModel
from ttcl.icl_mem0_comparison.protocol import public_messages
from ttcl.icl_mem0_comparison.worker import System,base
from src.interface import Query,Observation
from ttcl.icl_mem0_comparison.run import summarize

class Action(BaseModel):
    answer: str

class FakeClient:
    def __init__(self, replies=None):
        self.messages=[];self.replies=iter(replies or ['{"answer":"ok"}']*10)
    def generate(self,messages,random_seed):
        self.messages.append(copy.deepcopy(messages))
        return dict(raw_response=next(self.replies),input_tokens=10,output_tokens=5,
                    rendered_prompt_sha256='test')

class FakeMemory:
    def __init__(self):self.inputs=[];self.searches=[]
    def search(self,query):
        self.searches.append(query)
        return [{'memory':'Remember the public strategy'}]
    def add(self,messages):
        self.inputs.append(copy.deepcopy(messages));return {}

class ProtocolTests(unittest.TestCase):
    def system(self,root,client,arm,history=(),memory=None,index=0):
        return System(dict(task_seed=42,action_retries=2,max_turns_per_instance=64),
            client,arm,history,memory,root,'',index,90)
    def query(self):
        return Query(prompt='Question 1/1: public task',response_schema=Action,
                          instance_id='HIDDEN_OPPONENT_POLICY',metadata={'secret_answer':'forbidden'})
    def test_icl_preserves_raw_history_terminal_feedback_and_format_retries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);client=FakeClient(['malformed','{"answer":"ok"}','{"answer":"next"}'])
            s=self.system(root,client,'full_history_icl')
            s.respond(self.query())
            s.observe(Observation('recorded',True,{'hidden_reward':999}))
            history=copy.deepcopy(s.messages[1:])
            s2=self.system(root,client,'full_history_icl',history,index=1)
            s2.respond(self.query())
            supplied=client.messages[-1]
            self.assertEqual(supplied[1:1+len(history)],history)
            text=json.dumps(supplied)
            self.assertIn('malformed',text);self.assertIn('recorded',text)
            self.assertNotIn('HIDDEN_OPPONENT_POLICY',text)
            self.assertNotIn('secret_answer',text);self.assertNotIn('999',text)
            self.assertIn('Question 2/90',text)
    def test_mem0_gets_only_public_turn_and_retrieval_reaches_actor(self):
        with tempfile.TemporaryDirectory() as d:
            m=FakeMemory();client=FakeClient();s=self.system(Path(d),client,'mem0',memory=m)
            s.respond(self.query());s.observe(Observation('public result',True,{'reward':999}))
            self.assertIn('Remember the public strategy',json.dumps(client.messages[-1]))
            self.assertEqual(m.inputs,[public_messages('Question 1/90: public task',{'answer':'ok'},'public result')])
            self.assertNotIn('999',json.dumps(m.inputs))
            other=self.system(Path(d),client,'mem0',memory=m,index=1)
            other.respond(self.query())
            self.assertNotIn('PREVIOUS QUERY',m.searches[-1])
    def test_missing_scores_are_excluded_from_joint_pairs(self):
        from ttcl.icl_mem0_comparison.protocol import save,ARMS
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);save(root/'plan.json',dict(tasks={'poker':2},expected_cells=6))
            for arm in ARMS:
                for index in [0,1]:
                    failed=arm=='mem0' and index==1
                    row=dict(task='poker',arm=arm,repeat=303,canonical_index=index,
                        instance_id=str(index),initial_query_sha256=str(index),status='failed' if failed else 'complete',
                        reward=None if failed else 1.,memory_warnings=0)
                    save(root/'runs/poker/303'/arm/f'episode_{index+1:03}'/'row.json',row)
            stats=summarize(root)
            self.assertEqual(stats['failed_cells'],1)
            self.assertEqual(stats['domains']['poker']['all']['paired_count'],1)
            self.assertIsNone(stats['domains']['poker']['after_first']['means']['none'])

if __name__=='__main__':unittest.main()
