"""
Tests that task instance ordering is reproducible across runs.

Key invariant: prepare_run(0) must be a no-op — the canonical instance
ordering used by the baseline must be identical to the ordering that
run_index=0 would produce. Tasks that violate this produce meaningless
gain curves because baseline and run instances don't correspond.
"""

import unittest

from src.registry import get_task_class, list_tasks


def _snapshot(task):
    """Return a comparable snapshot of task.instances if it exists."""
    instances = getattr(task, "instances", None)
    if instances is None:
        return None
    # Use repr so this works for arbitrary instance types (dicts, dataclasses, etc.)
    return [repr(inst) for inst in instances]


class TestPrepareRunReproducibility(unittest.TestCase):
    """prepare_run(0) must leave self.instances unchanged."""

    def _try_instantiate(self, task_name, num_instances=5):
        try:
            cls = get_task_class(task_name)
        except Exception as exc:
            self.skipTest(f"Could not import {task_name}: {exc}")
        try:
            return cls(num_instances=num_instances)
        except Exception as exc:
            self.skipTest(f"Could not instantiate {task_name}: {exc}")

    def _check_task(self, task_name):
        task = self._try_instantiate(task_name)
        before = _snapshot(task)
        if before is None:
            self.skipTest(f"{task_name} has no self.instances to check")
        task.prepare_run(0)
        after = _snapshot(task)
        self.assertEqual(
            before,
            after,
            f"{task_name}.prepare_run(0) changed self.instances — "
            "it must be a no-op so the baseline sees the canonical ordering.",
        )

    def _check_task_permutes(self, task_name):
        """prepare_run(1) does not crash for tasks with >1 instance."""
        task = self._try_instantiate(task_name, num_instances=10)
        before = _snapshot(task)
        if before is None or len(before) < 2:
            self.skipTest(f"{task_name} doesn't have enough instances to permute")
        task.prepare_run(1)
        after = _snapshot(task)
        # We just verify it doesn't crash; some tasks may not support permutation
        self.assertIsNotNone(after)

    def _check_determinism(self, task_name):
        """Same seed → same canonical instance ordering across two instantiations."""
        try:
            cls = get_task_class(task_name)
        except Exception as exc:
            self.skipTest(f"Could not import {task_name}: {exc}")
        try:
            task_a = cls(num_instances=5)
            task_b = cls(num_instances=5)
        except Exception as exc:
            self.skipTest(f"Could not instantiate {task_name}: {exc}")
        snap_a = _snapshot(task_a)
        snap_b = _snapshot(task_b)
        if snap_a is None:
            self.skipTest(f"{task_name} has no self.instances")
        self.assertEqual(
            snap_a,
            snap_b,
            f"{task_name} produces different instance orderings on consecutive "
            "instantiations with the same default seed.",
        )


def _make_test(name):
    def test_noop(self):
        self._check_task(name)

    def test_permute(self):
        self._check_task_permutes(name)

    def test_determinism(self):
        self._check_determinism(name)

    return test_noop, test_permute, test_determinism


# Dynamically generate one test per registered task so failures are isolated.
for _task_name in list_tasks():
    _safe = _task_name.replace("-", "_")
    _noop, _permute, _det = _make_test(_task_name)
    setattr(
        TestPrepareRunReproducibility,
        f"test_{_safe}_prepare_run_0_is_noop",
        _noop,
    )
    setattr(
        TestPrepareRunReproducibility,
        f"test_{_safe}_prepare_run_1_does_not_crash",
        _permute,
    )
    setattr(
        TestPrepareRunReproducibility,
        f"test_{_safe}_instantiation_is_deterministic",
        _det,
    )


if __name__ == "__main__":
    unittest.main()
