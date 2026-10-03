"""Numerical projections abstain during source-episode burn-in."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from ttcl.icl_mem0_comparison.protocol import read, save
from .grounded_evidence_v25 import TypedGroundedMemory as V25Memory
from .grounded_evidence_v26 import (MIN_NUMERIC_SOURCES, TypedGroundedMemory,
                                    weak_numeric_projection)


class NumericSourceGateTest(TestCase):
    def test_only_weak_numeric_consensus_is_reverted(self):
        self.assertTrue(weak_numeric_projection(dict(
            operation='NUMERIC_CONSENSUS',
            candidate=dict(kind='numeric_consensus', sample_count=MIN_NUMERIC_SOURCES - 1))))
        self.assertFalse(weak_numeric_projection(dict(
            operation='NUMERIC_CONSENSUS',
            candidate=dict(kind='numeric_consensus', sample_count=MIN_NUMERIC_SOURCES))))
        self.assertFalse(weak_numeric_projection(dict(
            operation='RECURRING_RECORDS',
            candidate=dict(kind='recurring_records', sample_count=1))))
        self.assertFalse(weak_numeric_projection(dict(operation='KEEP')))

    def test_weak_projection_restores_actor_action_and_audit(self):
        class Schema:
            @staticmethod
            def model_validate(value):
                return value

        with TemporaryDirectory() as directory:
            system = SimpleNamespace(output=Path(directory), turn=1,
                                     last=('prompt', {'value': 1}))
            system.respond = lambda query: SimpleNamespace(
                action={'value': 1}, metadata={'source': 'actor'})
            path = system.output / 'typed_action_001.json'
            save(path, dict(actor={'value': 2}, final={'value': 1},
                            operation='NUMERIC_CONSENSUS',
                            candidate=dict(kind='numeric_consensus', sample_count=2)))
            memory = TypedGroundedMemory.__new__(TypedGroundedMemory)
            with patch.object(V25Memory, 'decorate_system'):
                memory.decorate_system(system)
            response = system.respond(SimpleNamespace(response_schema=Schema))
            self.assertEqual(response.action, {'value': 2})
            self.assertEqual(system.last, ('prompt', {'value': 2}))
            audit = read(path)
            self.assertEqual(audit['final'], {'value': 2})
            self.assertEqual(audit['operation'], 'KEEP')
            self.assertEqual(audit['rejected_candidate']['sample_count'], 2)
