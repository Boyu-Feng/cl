import unittest
import tempfile
from pathlib import Path
from .posteval import next_action, final_adapters, read_optional


class AutomaticEvaluationTests(unittest.TestCase):
    def test_never_evaluate_partial_training(self):
        self.assertEqual(next_action({'phase':'ppo_update'},False,None),'wait')
        self.assertEqual(next_action({'phase':'complete'},False,None),'failed')

    def test_original_development_and_test_are_allowed_to_finish(self):
        for phase in ['development','final_comparison']:
            self.assertEqual(next_action({'phase':phase},True,None),'wait')

    def test_completed_original_test_is_not_duplicated(self):
        self.assertEqual(next_action({'phase':'complete'},True,{'complete':True}),'reuse')

    def test_full_test_runs_even_when_development_gate_does_not_pass(self):
        self.assertEqual(next_action({'phase':'complete','qualified':[],
            'outcome':'no_dual_benchmark_development_gain'},True,None),'run_full_test')

    def test_missing_last_batch_is_a_wait_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(final_adapters(Path(directory),
                {'blocks':[{'id':'block_015'}],'arms':['writer_only','dual']}))
            self.assertIsNone(read_optional(Path(directory)/'not-yet-created.json'))

    def test_summary_waits_for_final_integrity_checks(self):
        self.assertEqual(next_action({'phase':'final_comparison'},True,{'complete':True}),'wait')

    def test_original_infrastructure_failure_remains_failure(self):
        self.assertEqual(next_action({'phase':'failed'},True,None),'failed')


if __name__ == '__main__':
    unittest.main()
