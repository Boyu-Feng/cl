import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from .upstream import grpo,executor_reward
from .retrieval import Corpus


class CoreTests(unittest.TestCase):
    def test_original_grpo_centers_within_each_group(self):
        f=grpo();r=torch.tensor([[0.],[1.],[7.],[7.]])
        a,_=f.compute_grpo_outcome_advantage(r,torch.ones_like(r),np.array([0,0,1,1]))
        self.assertAlmostEqual(a[0,0].item(),-2**-.5,places=5)
        self.assertAlmostEqual(a[1,0].item(),2**-.5,places=5)
        self.assertEqual(a[2:].abs().sum().item(),0.)
        lp=torch.tensor([[0.,0.]],requires_grad=True)
        loss,*_=f.compute_policy_loss(torch.zeros_like(lp),lp,torch.ones_like(lp),torch.ones_like(lp),cliprange=.2)
        loss.backward();self.assertTrue(torch.all(lp.grad<0))

    def test_executor_uses_supplied_offline_accuracy_and_original_weights(self):
        f=executor_reward()
        text='<think>Check.</think><answer>x</answer>'
        self.assertAlmostEqual(f('',text,'',{'acc_score':1}),.8)
        self.assertAlmostEqual(f('',text,'',{'acc_score':0}),.1)

    def test_raw_wiki_shard_boundaries_do_not_drop_or_duplicate_documents(self):
        data=b'{"id":1,"contents":"one"}\n{"id":2,"contents":"two"}\n'
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,chunk in enumerate([data[:7],data[7:34],data[34:]]):
                p=Path(tmp)/str(i);p.write_bytes(chunk);paths.append(p)
            c=Corpus(paths);rows=list(c.lines())
            self.assertEqual([r[2]['id'] for r in rows],[1,2])
            for offset,length,row in rows:self.assertEqual(c.read(offset,length),row)


if __name__=='__main__':unittest.main()
