import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from .memory import Memory


class Embeddings:
    def embed(self,texts):
        return [[1.,0.] for _ in texts]


class Client:
    repeat=303
    tokenizer=SimpleNamespace(encode=lambda text,**kwargs:text.split())
    def __init__(self):self.messages=[]
    def complete(self,messages,*args,**kwargs):
        self.messages.append(copy.deepcopy(messages))
        return dict(raw_response='Inspect the current evidence before acting.',input_tokens=10,output_tokens=8,finish_reason='stop')


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.plan=dict(upstream=str(Path(__file__).resolve().parents[2]/'current_work/MemRL'),
            writer_max_tokens=768,memory_tokens=2048,rl_config=dict(epsilon=0.,alpha=.3,gamma=0.,topk=3))
        self.client=Client()
        self.memory=Memory(self.plan,self.client,self.tmp.name,dict(threshold=.5,mean=.5,std=.1),Embeddings())

    def test_native_update_retrieval_credit_and_restore(self):
        m=self.memory;r=m.retrieve('public task')
        self.assertEqual(r['ids'],[])
        m.update('public task','public trace',1.,True,r,{'new_content':'sha-a'})
        r=m.retrieve('public task');self.assertEqual(r['ids'],['memory_000000'])
        m.update('public task','next public trace',-2.,None,r,{'new_content':'sha-b'})
        self.assertAlmostEqual(m.store.items['memory_000000'].metadata.model_extra['q_value'],-.6)
        self.assertEqual(m.store.items['memory_000000'].metadata.model_extra['q_visits'],1)
        ranked=m.retrieve('public task')
        self.assertEqual(ranked['ids'][0],'memory_000001')
        self.assertIsNone(m.store.items['memory_000001'].metadata.model_extra['success'])
        self.assertNotIn('sha-b',str(self.client.messages))
        self.assertNotIn('-2.0',str(self.client.messages))
        p=Path(self.tmp.name)/'state.json';m.snapshot(p)
        other=Memory(self.plan,self.client,Path(self.tmp.name)/'other',dict(threshold=.5,mean=.5,std=.1),Embeddings())
        other.restore(p);self.assertEqual(other.retrieve('public task')['ids'],ranked['ids'])

    def test_missing_reward_never_becomes_failure(self):
        with self.assertRaises(ValueError):
            self.memory.update('x','y',float('nan'),False,self.memory.retrieve('x'),{})
        self.assertEqual(self.memory.store.items,{})

    def test_budget_excluded_memories_get_no_credit(self):
        m=self.memory;m.update('x','trace',1.,True,m.retrieve('x'),{})
        m.plan['memory_tokens']=1;r=m.retrieve('x')
        self.assertEqual(r['ids'],[]);self.assertEqual(len(r['dropped_whole_entries']),1)
        m.update('x','trace',-1.,False,r,{})
        self.assertEqual(m.store.items['memory_000000'].metadata.model_extra['q_visits'],0)

    def test_swallowed_writer_failure_is_rejected(self):
        self.client.complete=lambda *a,**k: (_ for _ in ()).throw(ValueError('simulated outage'))
        with self.assertRaises(RuntimeError):self.memory.update('x','trace',-1.,False,self.memory.retrieve('x'),{})

    def test_no_memory_sharing_across_chains_or_protocols(self):
        self.memory.update('x','public trace',1.,True,self.memory.retrieve('x'),{})
        fresh=Memory(self.plan,self.client,Path(self.tmp.name)/'fresh',dict(threshold=.5,mean=.5,std=.1),Embeddings())
        self.assertEqual(fresh.retrieve('x')['ids'],[])
        p=Path(self.tmp.name)/'snapshot.json';self.memory.snapshot(p)
        changed=Memory(dict(self.plan,memory_tokens=1024),self.client,Path(self.tmp.name)/'changed',dict(threshold=.5,mean=.5,std=.1),Embeddings())
        with self.assertRaises(ValueError):changed.restore(p)

    def test_cl_continuous_reward_scale_does_not_drop_memory(self):
        m=Memory(dict(self.plan,q_min_threshold=None),self.client,Path(self.tmp.name)/'cl',
                 dict(threshold=.5,mean=.5,std=.1),Embeddings())
        m.update('x','trace',1.,None,m.retrieve('x'),{})
        m.update('x','trace',-100.,None,m.retrieve('x'),{})
        self.assertAlmostEqual(m.store.items['memory_000000'].metadata.model_extra['q_value'],-30.)
        self.assertIn('memory_000000',m.retrieve('x')['ids'])

    def test_capped_generation_preserves_budget_and_audits(self):
        original=self.client.complete
        self.client.complete=lambda *a,**k:dict(original(*a,**k),finish_reason='length')
        self.memory.update('x','trace',-1.,False,self.memory.retrieve('x'),{})
        self.assertEqual(self.memory.calls,1)
        self.assertEqual(self.memory.limit_hits,1)
        self.assertTrue(self.memory.store.items['memory_000000'].metadata.model_extra['writer_token_limit_hit'])


if __name__=='__main__':unittest.main()
