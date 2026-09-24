import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
from .protocol import PUBLIC_FIELDS, public_messages, training_label
from ttcl.experience_evolution.core import read, save


def episode(reward,completed=True):
    return dict(public_task_brief='task',initial_public_query='query',response_schemas={},
                steps=[{'query':'observation','action':'attempt','public_feedback':'failed action'}],
                completed=completed,format_failures=[],reward=reward,
                reward_scope='Official scalar for this whole episode only; higher is better.',
                local_execution_error=None if completed else 'Safety cap exceeded')


class ProtocolTest(unittest.TestCase):
    def test_reward_visible_hidden_metadata_excluded(self):
        ep=episode(-2);ep.update(next_task='secret next',hidden_answer='secret answer',baseline_reward=100)
        payload=json.loads(public_messages('old',ep)[1]['content'])
        self.assertEqual(payload['previous_experience'],'old')
        self.assertEqual(set(payload['completed_interaction']),set(PUBLIC_FIELDS))
        self.assertEqual(payload['completed_interaction']['reward'],-2)
        self.assertNotIn('secret',json.dumps(payload))

    def test_failures_and_missing_reward_remain_distinct(self):
        for reward,complete in [(-1,True),(0,True),(None,False)]:
            got=json.loads(public_messages('old',episode(reward,complete))[1]['content'])['completed_interaction']
            self.assertEqual(got['reward'],reward)
            self.assertEqual(len(got['steps']),1)
        with self.assertRaises(ValueError): public_messages('',episode(None))
        with self.assertRaises(ValueError): public_messages('',episode(float('nan')))

    def test_matched_objectives_do_not_redefine_delta(self):
        pairs=[({'reward':1},{'reward':1}),({'reward':0},{'reward':0})]
        self.assertEqual(training_label(pairs,'absolute'),(.5,0))
        self.assertEqual(training_label(pairs,'delta'),(0,0))
        bad=[({'reward':0},{'reward':1})]*2
        self.assertEqual(training_label(bad,'delta'),(-1,-1))
        self.assertEqual(training_label(bad,'absolute'),(0,-1))

    def test_online_feedback_chain_and_resume(self):
        from . import evaluate
        from .report import audit_chain
        calls=[]
        class Client:
            def __init__(self,*a,**kw): pass
            def complete(self,messages,model,**kw):
                calls.append((model,messages))
                r=json.loads(messages[1]['content'])['completed_interaction']['reward']
                return {'raw_response':f'updated from reward {r}', 'served_model':model}
        def run_episode(args,model,index,dest,context):
            r=[-1,0,None][index];ep=episode(r,index<2)
            row={'reward':r,'status':'complete' if index<2 else 'failed','episode':index+1,
                 'canonical_index':index,'error':'' if index<2 else 'Safety cap exceeded'}
            save(dest/'trajectory.json',ep)
            return row,ep
        stub=ModuleType('ttcl.structured_memory.online_bank');stub.run_episode=run_episode
        cwd=os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                save(root/'plan.json',{'suites':{'test':{'url':'unused','tasks':{'test':3},'repeats':[303],
                                                    'arms':{'none':'frozen-actor','untrained':'frozen-actor','delta':'delta'}}}})
                with patch.dict(sys.modules,{'ttcl.structured_memory.online_bank':stub}),patch.object(evaluate,'configure_tasks'),patch.object(evaluate,'BENCH',root),patch.object(evaluate,'Client',Client):
                    evaluate.clbench(root,'test','test',303)
                    self.assertEqual(len(calls),6)
                    d=root/'test/clbench/test/303/delta'
                    self.assertEqual(read(d/'episode_002/memory_before.json')['text'],'updated from reward -1')
                    self.assertEqual(read(d/'episode_003/memory_before.json')['text'],'updated from reward 0')
                    audit=audit_chain(root,'test','test',303)
                    self.assertEqual(audit['missing_rewards'],2)
                    self.assertEqual(audit['negative_rewards'],2)
                    evaluate.clbench(root,'test','test',303)
                    self.assertEqual(len(calls),6)
        finally: os.chdir(cwd)

if __name__=='__main__': unittest.main()
