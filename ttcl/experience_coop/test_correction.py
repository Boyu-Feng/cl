import math
import unittest


class CorrectionTests(unittest.TestCase):
    def test_exact_behavior_reduces_to_original_objective(self):
        import torch
        from .learning import ppo_loss, rollout_correction
        old=torch.tensor([-1.,-2.]);current=torch.tensor([-.95,-2.1],requires_grad=True)
        weights,keep=rollout_correction(old,old)
        self.assertTrue(keep);self.assertFalse(weights.requires_grad)
        original=ppo_loss(current,old,old,.7)[0]
        corrected=ppo_loss(current,old,old,.7,correction=weights)[0]
        torch.testing.assert_close(original,corrected)

    def test_correction_is_detached_and_not_an_extra_ppo_ratio(self):
        import torch
        from .learning import ppo_loss, rollout_correction
        anchor=torch.tensor([-1.],requires_grad=True)
        behavior=torch.tensor([-1.-math.log(1.5)])
        weights,keep=rollout_correction(anchor,behavior)
        current=anchor.detach().clone().requires_grad_()
        loss,_,_,fraction=ppo_loss(current,anchor.detach(),anchor.detach(),1.,beta=0.,correction=weights)
        loss.backward()
        self.assertTrue(keep);self.assertEqual(fraction,0.)
        self.assertAlmostEqual(float(current.grad),-1.5,places=5)
        self.assertIsNone(anchor.grad)

    def test_weights_are_bounded_and_extreme_actions_are_rejected(self):
        import torch
        from .learning import rollout_correction
        anchor=torch.tensor([-1.,-2.]);behavior=anchor-torch.tensor([math.log(3.),0.])
        weights,keep=rollout_correction(anchor,behavior)
        self.assertTrue(keep);self.assertEqual(float(weights.max()),2.)
        _,keep=rollout_correction(anchor,anchor+torch.tensor([math.log(5.),0.]))
        self.assertFalse(keep)


if __name__=='__main__':unittest.main()
