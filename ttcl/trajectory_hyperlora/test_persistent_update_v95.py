import unittest,torch
from ttcl.trajectory_hyperlora.persistent_update_v95 import PersistentUpdateRule
class TestPersistent(unittest.TestCase):
 def test_state_and_future_gradient(self):
  torch.manual_seed(95);rule=PersistentUpdateRule([(7,4)],state_dim=4,width=16);aa=[torch.randn(4,5)];old=[torch.zeros(1,7,4)];m=torch.zeros(1,16);ep=[]
  for i in range(3):ep.append(dict(before=torch.randn(1,4),action=torch.tensor([i]),after=torch.randn(1,4),reward=torch.tensor([float(i==2)]),done=torch.tensor([0.]),feedback_change=torch.tensor([1.])))
  m,new,g,c=rule.update(m,ep,old,aa,[10.]);loss=new[0].square().mean()+m.square().mean();loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(float(rule.memory.weight_hh.grad.abs().sum()),0);self.assertGreater(float(g),0);self.assertGreater(float(c),0)
 def test_cumulative_state_changes(self):
  torch.manual_seed(95);r=PersistentUpdateRule([(3,2)],state_dim=4,width=8);m=torch.zeros(1,8);e=[dict(before=torch.ones(1,4),action=torch.tensor([1]),after=torch.ones(1,4),reward=torch.tensor([1.]),done=torch.tensor([0.]),feedback_change=torch.tensor([1.]))];m1=r.step_memory(m,e);m2=r.step_memory(m1,e);self.assertFalse(torch.equal(m1,m2))
if __name__=='__main__':unittest.main()
