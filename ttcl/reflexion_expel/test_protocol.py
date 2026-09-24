from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from .upstream import Upstream
from . import run


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Upstream(Path(__file__).resolve().parents[3]/'upstream')

    def test_native_rule_counters_and_rejected_truncation(self):
        rules, operations, _ = self.upstream.update([], 'ADD 1: Inspect the schema first.\nADD 2: incomplete')
        self.assertEqual(rules, [('Inspect the schema first.', 2)])
        rules, _, _ = self.upstream.update(rules, 'AGREE 1: Inspect the schema first.')
        self.assertEqual(rules[0][1], 3)
        rules, _, _ = self.upstream.update(rules, 'REMOVE 1: Inspect the schema first.')
        self.assertEqual(rules[0][1], 2)

    def test_feedback_allowlist(self):
        episode = {'steps': [{'action': 'x', 'public_feedback': 'failed'}],
                   'reward': -2, 'hidden_answer': 'SECRET', 'future_task': 'SECRET'}
        supplied = run.public_episode(episode, {'success': False, 'instance_id': 'SECRET'})
        self.assertEqual(supplied['reward'], -2)
        self.assertFalse(supplied['official_success'])
        self.assertNotIn('SECRET', str(supplied))

    def test_failure_reflection_and_first_success_stopping(self):
        rows = [{'reward': -1, 'success': False, 'actor_calls': 1,
                 'actor_input_tokens': 2, 'actor_output_tokens': 3},
                {'reward': .1, 'success': True, 'actor_calls': 1,
                 'actor_input_tokens': 2, 'actor_output_tokens': 3}]
        eps = [{'reward': -1, 'steps': ['failed action']}, {'reward': .1, 'steps': []}]
        seen = []
        def actor(*args):
            seen.append(args[4])
            i = len(seen)-1
            return rows[i], eps[i]
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'actor_episode', side_effect=actor), \
                patch.object(run, 'llm_call', return_value={'raw_response': 'Recheck the action.'}) as llm:
            got, _ = run.run_trials(None, self.upstream, SimpleNamespace(task='test'),
                                    0, Path(d), 303)
        self.assertEqual(len(got), 2)
        self.assertEqual(seen[0], '')
        self.assertIn('Recheck the action.', seen[1])
        self.assertIn('"reward": -1', llm.call_args.args[1][0]['content'])

    def test_no_reflection_control(self):
        row = {'reward': -1, 'success': False, 'actor_calls': 1,
               'actor_input_tokens': 2, 'actor_output_tokens': 3}
        with tempfile.TemporaryDirectory() as d, \
                patch.object(run, 'actor_episode', return_value=(row, {})) as actor, \
                patch.object(run, 'llm_call') as llm:
            got, _ = run.run_trials(None, self.upstream, SimpleNamespace(task='test'),
                                    0, Path(d), 303, reflection=False)
        self.assertEqual(len(got), 3)
        llm.assert_not_called()
        self.assertTrue(all(call.args[4] == '' for call in actor.call_args_list))

    def test_expel_contrasts_only_success_and_failure_of_same_task(self):
        client = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s, **kw: s.split()))
        def item(index, rewards):
            return {'index': index,
                    'rows': [{'reward': r, 'success': r > 0} for r in rewards],
                    'episodes': [{'reward': r, 'initial_public_query': f'Task {index}',
                                  'steps': [f'trajectory {index}']} for r in rewards]}
        gathered = [item(0, [-1, 1]), item(1, [-1, -1]), item(2, [1])]
        with tempfile.TemporaryDirectory() as d, patch.object(run, 'llm_call',
                return_value={'raw_response': 'ADD 1: Inspect feedback before choosing an action.'}) as llm:
            state = run.extract_rules(client, self.upstream, gathered, Path(d), 'test', 303)
        self.assertEqual(llm.call_count, 2)
        contrast = llm.call_args_list[0].args[1][1]['content']
        self.assertIn('Task 0', contrast)
        self.assertNotIn('Task 1', contrast)
        self.assertNotIn('Task 2', contrast)
        self.assertEqual([x['index'] for x in state['successful_examples']], [0, 2])
        self.assertFalse(state['test_feedback_used'])

    def test_empty_bank_has_no_actor_context(self):
        context, audit = run.retrieve_context({'rules': [], 'successful_examples': []},
                                              'target', None, None)
        self.assertEqual(context, '')
        self.assertEqual(audit['selected'], [])

    def test_retrieval_excludes_identical_task(self):
        retriever = SimpleNamespace(rank=lambda q, docs: [(0, 1), (1, .5)])
        tokenizer = SimpleNamespace(encode=lambda s, **kw: s.split())
        state = {'rules': [], 'successful_examples': [
            {'index': 0, 'query': 'target', 'trajectory': 'DO NOT RETRIEVE'},
            {'index': 1, 'query': 'other', 'trajectory': 'useful past interaction'}]}
        context, audit = run.retrieve_context(state, 'target', retriever, tokenizer)
        self.assertNotIn('DO NOT RETRIEVE', context)
        self.assertEqual(audit['selected'][0]['training_index'], 1)


if __name__ == '__main__':
    unittest.main()
