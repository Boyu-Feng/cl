import unittest
from .audit_causal_pairs_v161 import executable, prefix


class CausalPrefixTest(unittest.TestCase):
    def test_same_query_does_not_hide_different_feedback(self):
        a = [{'query': 'q', 'action': {'action': 'QUERY'}, 'public_feedback': 'table A'}]
        b = [{'query': 'q', 'action': {'action': 'QUERY'}, 'public_feedback': 'table B'}]
        self.assertNotEqual(prefix(a, 1), prefix(b, 1))

    def test_current_and_future_feedback_excluded(self):
        a = [{'query': 'q', 'action': {'action': 'ANSWER'}, 'public_feedback': 'CORRECT'}]
        b = [{'query': 'q', 'action': {'action': 'ANSWER'}, 'public_feedback': 'INCORRECT'}]
        self.assertEqual(prefix(a, 0), prefix(b, 0))

    def test_rationale_is_not_executable_action(self):
        self.assertEqual(executable({'action': 'RAISE', 'amount': 40, 'thinking': 'a'}),
                         executable({'action': 'RAISE', 'amount': 40, 'thinking': 'b'}))
        self.assertNotEqual(executable({'action': 'RAISE', 'amount': 40}),
                            executable({'action': 'RAISE', 'amount': 80}))


if __name__ == '__main__':
    unittest.main()
