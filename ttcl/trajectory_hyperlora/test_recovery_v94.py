import unittest
from types import SimpleNamespace
import torch
from ttcl.trajectory_hyperlora.direct_composition_pilot import GeneratedDownProjection
from ttcl.trajectory_hyperlora.recover_continual_xland_v94 import Runner

class MountTest(unittest.TestCase):
    def test_parameter_then_generated_tensor_keeps_gradients(self):
        adapter=GeneratedDownProjection(torch.nn.Linear(5,7),2)
        stub=SimpleNamespace(agent=SimpleNamespace(adapters=[adapter]))
        p=torch.nn.Parameter(torch.zeros(1,7,2))
        Runner.mount(stub,[p]);adapter(torch.randn(1,3,5)).sum().backward()
        self.assertIsNotNone(p.grad);self.assertNotIn('b',adapter._parameters)
        head=torch.nn.Linear(3,14)
        b=head(torch.ones(1,3)).reshape(1,7,2)
        Runner.mount(stub,[b]);adapter(torch.randn(1,3,5)).sum().backward()
        self.assertIsNotNone(head.weight.grad)
        adapter.b=None
        Runner.mount(stub,[p]);self.assertNotIn('b',adapter._parameters)

if __name__=='__main__':unittest.main()
