from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ttcl.icl_mem0_comparison.protocol import save
from .contextual_utility import ContextualUtilityMemory, UtilityGate, retrieval_features


class ContextualUtilityTests(unittest.TestCase):
    def test_training_feature_order_and_word_gate(self):
        ret = {'ids':['m1','m2'], 'tokens':1024}
        md = {'m1':{'success':True,'q_value':.8},
              'm2':{'success':False,'q_value':-.4,'writer_token_limit_hit':True}}
        self.assertEqual(retrieval_features(ret, md), [2/3, .5, .5, .5, .2, .5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'gate.json'
            save(path, {'schema':'contextual_utility_ridge_v1',
                        'vocab':['clean','object'], 'weights':[0., .3, -.5, 0.,0.,0.,0.,0.,0.],
                        'threshold':-.2})
            gate = UtilityGate(path)
            self.assertAlmostEqual(gate.predict('Clean the object', [0.]*6), -.2)

    def test_suppressed_memory_has_no_credit_ids(self):
        obj = object.__new__(ContextualUtilityMemory)
        obj.gate = SimpleNamespace(threshold=-.2, predict=lambda q,z:-.3)
        obj.store = SimpleNamespace(get=lambda mid:SimpleNamespace(
            metadata=SimpleNamespace(model_dump=lambda:{'success':True,'q_value':.1})))
        base = dict(query='q', context='memory text', ids=['m1'], tokens=3,
                    dropped_whole_entries=[], context_sha256='old')
        with patch('ttcl.memrl_comparison.contextual_utility.Memory.retrieve', return_value=base):
            result = obj.retrieve('q')
        self.assertEqual(result['ids'], [])
        self.assertEqual(result['context'], '')
        self.assertEqual(result['dropped_whole_entries'], ['m1'])
        self.assertEqual(result['gate'], 'suppressed')


if __name__ == '__main__':
    unittest.main()
