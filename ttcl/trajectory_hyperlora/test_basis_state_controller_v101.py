import unittest
import torch

from .basis_state_controller_v101 import (BasisStateController, PersistentBasisMemory,
                                           FeatureBasisRouter, SuccessPrototypeMemory,
                                           AnchoredDeltaController, SignedBasisController)


class BasisStateControllerTest(unittest.TestCase):
    def test_shared_source_does_not_fall_back_to_direct_success(self):
        from .alfworld_mounted_basis_online_v102 import direct_write_source
        own = {'status': 'complete', 'reward': 1, 'trajectory': ['own']}
        failed = {'status': 'complete', 'reward': 0, 'trajectory': ['failed']}
        shared = {'status': 'complete', 'reward': 1, 'trajectory': ['shared']}
        self.assertIsNone(direct_write_source(own, failed, shared=True))
        self.assertIs(direct_write_source(failed, shared, shared=True), shared)
        self.assertIs(direct_write_source(own, shared, shared=False), own)

    def test_tied_signed_update_preserves_direction_and_matches_common_step(self):
        bases = [[torch.tensor([1., 0.])], [torch.tensor([0., 2.])]]
        full = SignedBasisController(bases, lr=.05)
        tied = SignedBasisController(bases, lr=.05, tied=True)
        loss = lambda bb: (bb[0] - torch.tensor([2., -1.])).square().sum()
        full.update_mounted(loss)
        tied.update_mounted(loss)
        self.assertTrue(torch.allclose(tied.coefficients, full.coefficients.mean().expand(2)))
        self.assertFalse(torch.allclose(full.coefficients[0], full.coefficients[1]))
        expected = tied.coefficients[0] * torch.tensor([1., 2.])
        self.assertTrue(torch.allclose(tied.matrices()[0], expected))

    def test_signed_direction_preserves_initial_l1_magnitude(self):
        c = SignedBasisController([[torch.tensor([1.])], [torch.tensor([-1.])]],
                                   lr=.5, preserve_l1=True)
        c.update_mounted(lambda bb: (bb[0] - 1).square().sum())
        self.assertAlmostEqual(float(c.coefficients.abs().sum()), .5, places=5)

    def test_appended_basis_contributes_to_mounted_matrix(self):
        c = BasisStateController([[torch.zeros(2)]])
        c.append_basis([torch.tensor([0., 4.])], initial_logit=0.)
        self.assertEqual(c.k, 2)
        self.assertTrue(torch.allclose(c.matrices()[0], torch.tensor([0., 1.])))
        audit = c.update_mounted(lambda bb: (bb[0][1] - 2).square())
        self.assertTrue(audit['matrix_changed'])

    def test_anchor_penalty_limits_state_drift(self):
        c = BasisStateController([[torch.tensor([1.])], [torch.tensor([-1.])]],
                                  lr=.5, anchor_strength=10.)
        before = c.matrices()[0].clone()
        c.update_mounted(lambda bb: (bb[0] - 5).square().sum())
        self.assertLess(float((c.matrices()[0] - before).abs()), 1.)

    def test_trajectory_tail_pairs_observations_and_normalizes_selected_steps(self):
        from unittest.mock import patch
        from . import alfworld_mounted_basis_online_v102 as runner
        episode = {'initial_observation': 'initial', 'trajectory': [
            {'turn': i, 'observation': f'after-{i}', 'command': f'act-{i}',
             'available_commands': [f'act-{i}']} for i in range(6)]}
        seen = []
        def loss(agent, tok, observation, commands, command, device, bb):
            seen.append((observation, command))
            return torch.tensor(2.)
        with patch.object(runner, 'loss_for_action', side_effect=loss):
            actual = runner.loss_for_trajectory(None, None, episode, 'cpu', [])
            self.assertEqual(seen, [(f'after-{i-1}', f'act-{i}') for i in range(2, 6)])
            self.assertAlmostEqual(float(actual), 2., places=5)
            seen.clear()
            actual = runner.loss_for_trajectory(None, None, episode, 'cpu', [], credit_steps=0)
        self.assertEqual(seen, [(('initial' if i == 0 else f'after-{i-1}'), f'act-{i}')
                                for i in range(6)])
        self.assertAlmostEqual(float(actual), 2., places=5)

    def test_mounted_nonlinear_actor_updates_B_without_changing_actor(self):
        from .direct_composition_pilot import GeneratedDownProjection
        torch.manual_seed(101)
        adapter = GeneratedDownProjection(torch.nn.Linear(3, 2), rank=1)
        adapter.requires_grad_(False)
        with torch.no_grad():
            adapter.a.fill_(1.)
        before = {n: p.clone() for n, p in adapter.named_parameters()}
        bases = [[torch.tensor([[[1.], [-1.]]])],
                 [torch.tensor([[[-1.], [1.]]])]]
        c = BasisStateController(bases, lr=.1)
        def loss_fn(bb):
            adapter.b = bb[0]
            logits = adapter(torch.ones(1, 1, 3)).tanh().reshape(1, 2)
            return torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        old_loss = float(loss_fn(c.matrices()))
        audit = c.update_mounted(loss_fn)
        self.assertTrue(audit['matrix_changed'])
        self.assertGreater(audit['coefficient_grad_norm'], 0)
        self.assertLess(float(loss_fn(c.matrices())), old_loss)
        for name, p in adapter.named_parameters():
            self.assertTrue(torch.equal(p, before[name]))
            self.assertIsNone(p.grad)
        for row, ref in zip(c.bases, bases):
            self.assertTrue(torch.equal(row[0], ref[0]))

    def test_zero_basis_cannot_be_reported_as_effective_learning(self):
        c = BasisStateController([[torch.zeros(1)], [torch.zeros(1)]])
        info = c.update_mounted(lambda bb: (bb[0] - 1).square().sum())
        self.assertFalse(info['matrix_changed'])
        self.assertEqual(info['coefficient_grad_norm'], 0.)

    def test_nonfinite_mounted_loss_does_not_commit_state(self):
        c = BasisStateController([[torch.ones(1)], [-torch.ones(1)]])
        before = c.state.coefficients.clone()
        with self.assertRaises(ValueError):
            c.update_mounted(lambda bb: bb[0].sum() * float('nan'))
        self.assertTrue(torch.equal(before, c.state.coefficients))

    def test_failure_mounted_uses_negative_evidence_without_new_label(self):
        c = BasisStateController([[torch.ones(1)], [-torch.ones(1)]], lr=.1)
        before = c.state.coefficients.clone()
        info = c.update_failure_mounted(lambda bb: (bb[0] - 1).square().sum())
        self.assertTrue(info['negative_update'])
        self.assertFalse(torch.equal(before, c.state.coefficients))

    def test_success_changes_only_small_state_and_failure_does_not_change_bases(self):
        bases = [[torch.randn(2, 3), torch.randn(4, 3)] for _ in range(3)]
        before = [[x.clone() for x in row] for row in bases]
        c = BasisStateController(bases)
        c.update_failure()
        c.update_success(torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.5, 0.5]]), 1)
        self.assertTrue(any(not torch.equal(a, b)
                           for a, b in zip(c.state.coefficients, torch.zeros(3))))
        for row, old in zip(c.bases, before):
            for a, b in zip(row, old):
                self.assertTrue(torch.equal(a, b))
        self.assertTrue(all(torch.isfinite(x).all() for x in c.matrices()))

    def test_persistent_memory_keeps_task_slots_separate(self):
        bases = [[torch.ones(1, 2, 1), -torch.ones(1, 2, 1)],
                 [-torch.ones(1, 2, 1), torch.ones(1, 2, 1)]]
        memory = PersistentBasisMemory(bases, capacity=2, temperature=.1, lr=.2)
        logits = torch.tensor([[2., -2.], [-2., 2.]])
        memory.write_success(torch.tensor([1., 0.]), logits, 1)
        memory.write_success(torch.tensor([0., 1.]), logits, 0)
        _, audit = memory.read(torch.tensor([1., 0.]))
        self.assertEqual(audit["entries"], 2)
        self.assertEqual(audit["selected"], 0)
        self.assertEqual(len(memory.entries), 2)

    def test_append_basis_preserves_old_vectors_and_adds_one_coefficient(self):
        first = [[torch.ones(1, 2, 1)], [[-torch.ones(1, 2, 1)][0]]]
        c = BasisStateController(first, lr=.1)
        old = [x.clone() for row in c.bases for x in row]
        c.append_basis([torch.full((1, 2, 1), 3.)])
        self.assertEqual(len(c.bases), 3)
        self.assertEqual(c.state.coefficients.numel(), 3)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(
            old, [x for row in c.bases[:2] for x in row])))

    def test_residual_basis_is_bounded_and_immutable(self):
        c = BasisStateController([[torch.ones(2)], [torch.ones(2)]])
        scale = c.append_residual_basis([torch.full((2,), 100.)], max_fraction=.25)
        self.assertEqual(len(c.bases), 3)
        self.assertGreater(scale, 0)
        residual_norm = c.bases[-1][0].norm()
        reference_norm = torch.stack([x[0] for x in c.bases[:2]]).mean(0).norm()
        self.assertLessEqual(float(residual_norm), float(reference_norm * .25 + 1e-6))

    def test_orthogonal_residual_removes_existing_direction(self):
        c = BasisStateController([[torch.tensor([1., 0.])]])
        c.append_residual_basis([torch.tensor([1., 1.])], max_fraction=1.,
                                orthogonalize=True)
        self.assertAlmostEqual(float(c.bases[-1][0][0]), 0., places=5)
        self.assertGreater(float(c.bases[-1][0][1]), 0.)

    def test_feature_router_updates_selection_without_mutating_bases(self):
        bases = [[torch.tensor([1., 0.])], [torch.tensor([0., 1.])]]
        router = FeatureBasisRouter(bases, key_dim=2, lr=.2)
        old = [[x.clone() for x in row] for row in router.bases]
        key = torch.tensor([1., 0.])
        before = router.matrices(key)[0].clone()
        audit = router.update_mounted(key, lambda bb: (bb[0][1] - 1).square().sum())
        after = router.matrices(key)[0]
        self.assertTrue(audit['router_grad_norm'] > 0)
        self.assertGreater(float(after[1]), float(before[1]))
        self.assertTrue(all(torch.equal(x, y) for row, oldrow in zip(router.bases, old)
                            for x, y in zip(row, oldrow)))

    def test_success_prototype_memory_retrieves_matching_key(self):
        memory = SuccessPrototypeMemory([torch.zeros(2)], threshold=.5)
        memory.add(torch.tensor([1., 0.]), [torch.tensor([2., 3.])])
        basis, audit = memory.read(torch.tensor([1., .01]))
        self.assertEqual(audit['selected'], 0)
        self.assertTrue(torch.equal(basis[0], torch.tensor([2., 3.])))

    def test_anchored_delta_keeps_initial_matrix_and_adds_bounded_residual(self):
        c = AnchoredDeltaController([torch.tensor([1., 0.])], lr=.1, max_fraction=.25)
        c.add_success_delta([torch.tensor([1., 4.])], initial_weight=1.)
        self.assertLessEqual(float((c.matrices()[0] - c.anchor[0]).norm()), .25 + 1e-5)
        old_anchor = c.anchor[0].clone()
        c.update_mounted(lambda bb: (bb[0][1] - 1).square().sum())
        self.assertTrue(torch.equal(c.anchor[0], old_anchor))

    def test_signed_composition_starts_at_frozen_mix_and_allows_negative_weights(self):
        c = SignedBasisController([[torch.tensor([1.])], [torch.tensor([-1.])]], lr=1.)
        self.assertAlmostEqual(float(c.matrices()[0]), 0., places=5)
        c.update_mounted(lambda bb: (bb[0] - 1).square().sum())
        self.assertLess(float(c.coefficients[1]), 0.5)
        self.assertTrue((c.coefficients < 0).any() or (c.coefficients > .5).any())

    def test_keyed_memory_routes_mounted_updates_to_slots(self):
        bases = [[torch.tensor([1.0])], [torch.tensor([-1.0])]]
        memory = PersistentBasisMemory(bases, capacity=2, lr=.1)
        loss_fn = lambda bb: (bb[0] - 1).square().sum()
        first = memory.write_mounted_success(torch.tensor([1., 0.]), loss_fn)
        second = memory.write_mounted_failure(torch.tensor([0., 1.]), loss_fn)
        self.assertEqual(first['slot'], 0)
        self.assertEqual(second['slot'], 1)
        self.assertEqual(len(memory.entries), 2)
        self.assertTrue(first['matrix_changed'])


if __name__ == "__main__":
    unittest.main()
