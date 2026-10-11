"""Tests of persistent updates, norm limits, and feedback-only target selection."""
import unittest
import torch
from ttcl.trajectory_hyperlora.continual_update_v94 import UpdateRule,bound,effective_norm,positive_targets

class Updates(unittest.TestCase):
    def test_effective_norm_and_bound(self):
        torch.manual_seed(94);a=torch.randn(4,17);b=torch.randn(1,13,4)*10
        expected=(b[0]@a/4).norm()
        torch.testing.assert_close(effective_norm(b,a),expected)
        self.assertLessEqual(float((bound(b,a,.1)[0]@a/4).norm()),.100001)
        z=torch.zeros_like(b,requires_grad=True);bound(z,a,.1).sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
    def test_unrolled_gradient_and_state_dependence(self):
        torch.manual_seed(94);rule=UpdateRule(8,[(7,4)]);aa=[torch.randn(4,9)];bb=[torch.zeros(1,7,4)]
        fields={'last':torch.randn(1,8),'mean':torch.randn(1,8)}
        out,_=rule(fields,bb,aa,[10]);torch.testing.assert_close(out[0],bb[0])
        with torch.no_grad():rule.heads[0].weight.normal_(std=.01)
        once,_=rule(fields,bb,aa,[10]);twice,_=rule(fields,once,aa,[10])
        self.assertFalse(torch.equal(once[0],twice[0]))
        twice[0].square().sum().backward()
        self.assertTrue(torch.isfinite(rule.heads[0].weight.grad).all())
        self.assertGreater(float(rule.encoder[1].weight.grad.abs().sum()),0)
        self.assertLessEqual(float(effective_norm(twice[0],aa[0])),1.50001)
    def test_failure_does_not_invent_action_labels(self):
        ep={'content_sha256':'abc','trajectory':[{'observation':[[[0,0]]],'action':i%6,'reward':0.0} for i in range(10)]}
        self.assertEqual(positive_targets([ep]),[])
        ep['trajectory'][6]['reward']=1
        result=positive_targets([ep]);self.assertEqual([r['step'] for r in result],[3,4,5,6])
        self.assertTrue(all(r['step']<=6 for r in result))

if __name__=='__main__':unittest.main()
