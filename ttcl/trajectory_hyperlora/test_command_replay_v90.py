"""Budget, provenance and sampling checks without loading the actor."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import unittest
from collections import Counter


class SamplingTests(unittest.TestCase):
    def test_balanced_sampling_and_source_binding(self):
        source = Path(__file__).with_name('alfworld_cold_own_static_v90.py')
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('action_group', 'replay_items')]
        ns = {'random': random, 'digest': lambda x: hashlib.sha256(
            json.dumps(x, sort_keys=True).encode()).hexdigest()}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        actions = [{'target_action': f'go to room {i}', 'visible': i} for i in range(99)]
        actions.append({'target_action': 'take object', 'visible': 99})
        current = {'game': 'current', 'actions': actions}
        pool = [{'game': 'past', 'family': 'family', 'actions': copy.deepcopy(actions)}]
        before = copy.deepcopy((current, pool))
        counts = Counter()
        for index in range(200):
            items, schedule = ns['replay_items'](current, pool, index)
            self.assertEqual((items, schedule), ns['replay_items'](current, pool, index))
            self.assertEqual(len(items), 20)
            self.assertEqual(Counter(s['kind'] for s in schedule['sources']),
                             {'current': 10, 'past': 10})
            self.assertEqual(schedule['items_sha256'], ns['digest'](items))
            for item, src in zip(items, schedule['sources'], strict=True):
                owner = current if src['kind'] == 'current' else pool[0]
                self.assertEqual(src['game'], owner['game'])
                self.assertEqual(item, owner['actions'][src['action_index']])
                self.assertEqual(src['command_group'], item['target_action'].split()[0])
                counts[src['command_group']] += 1
        # A 99:1 action imbalance should not remain a 99:1 training imbalance.
        self.assertTrue(1800 < counts['take'] < 2200, counts)
        self.assertEqual((current, pool), before)
        with self.assertRaises(ValueError):
            ns['action_group']({'target_action': ' '})


if __name__ == '__main__':
    unittest.main()
