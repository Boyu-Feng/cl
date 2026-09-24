"""Check original-agent lifecycle isolation and evaluator-only scoring."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

from ttcl.generic_agent.run_benchmark import ROOT, run_mode, sandbox_command, seed_for


class LocalBackendTest(unittest.TestCase):
    def test_genericagent_content_blocks_reach_qwen_as_text(self):
        sys.path.insert(0, str(ROOT / 'current_work/GenericAgent'))
        from local_qwen import text_messages
        from llmcore import _msgs_claude2oai

        original = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Actual scan 53.7 MHz'}]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Previous answer'}]}]
        output = text_messages(_msgs_claude2oai(original))
        self.assertEqual(output[0]['content'], 'Actual scan 53.7 MHz')
        self.assertEqual(output[1]['content'], 'Previous answer')
        self.assertIsInstance(original[0]['content'], list)
        with self.assertRaises(ValueError):
            text_messages([{'role': 'user', 'content': [{'type': 'image_url'}]}])


class FakeWorker:
    instances = []

    def __init__(self, args, runtime):
        self.prompts = []
        self.closed = False
        self.instances.append(self)

    def ask(self, prompt, seed, timeout):
        self.prompts.append(prompt)
        return {"response": '{"transmitters":[]}', "usage": [{"output_tokens": 4}],
                "tools": [], "working": {}, "exit_reason": "CURRENT_TASK_DONE"}

    def close(self):
        self.closed = True


class ProtocolTest(unittest.TestCase):
    def test_tool_report_survives_final_ack_and_latest_revision_wins(self):
        from ttcl.generic_agent.worker import delivered_report

        first = '{"transmitters":[]}'
        later = '{"transmitters":[{"center_freq":12}]}'
        self.assertEqual(delivered_report([first, 'Done'], 'Done', True), first)
        self.assertEqual(delivered_report([first, later, 'Done'], 'Done', True), later)
        self.assertEqual(delivered_report([first], 'Stopped', False), '')

    def test_stateful_keeps_original_agent_stateless_recreates_it(self):
        from ttcl.common.bsm import BENCH

        for mode, count in (("stateful", 1), ("stateless", 2)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                FakeWorker.instances = []
                args = SimpleNamespace(output_dir=Path(tmp), num_scans=2, seed=42, timeout=30,
                                       data_path=BENCH / 'data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl')
                metrics = run_mode(args, mode, FakeWorker)
                self.assertEqual(len(FakeWorker.instances), count)
                self.assertTrue(all(w.closed for w in FakeWorker.instances))
                self.assertEqual(metrics['reward_calls'], 2)
                self.assertEqual(metrics['model_calls'], 2)
                self.assertFalse(metrics['reward_supplied_to_agent'])
                prompts = [p for w in FakeWorker.instances for p in w.prompts]
                self.assertTrue(all('Past scalar-reward experience' not in p for p in prompts))
                self.assertTrue(all('Historical public scan evidence' not in p for p in prompts))
                saved = json.loads((Path(tmp) / mode / 'metrics.json').read_text())
                self.assertEqual(metrics, saved)

    def test_sandbox_does_not_mount_benchmark_or_result_tree(self):
        args = SimpleNamespace(model=Path('/models/qwen'))
        command = sandbox_command(args, Path('/runs/private/runtime'))
        self.assertIn('--unshare-net', command)
        self.assertIn('--unshare-pid', command)
        self.assertNotIn('/home/fengboyu/cl', command)
        self.assertNotIn('/runs/private', command)
        self.assertEqual(seed_for(42, 'scan1'), seed_for(42, 'scan1'))
        self.assertNotEqual(seed_for(42, 'scan1'), seed_for(42, 'scan2'))


if __name__ == '__main__':
    unittest.main()
