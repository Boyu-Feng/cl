from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from pydantic import BaseModel, create_model

from .improved_cl import ImprovedCLMemory
from .test_protocol import Client, Embeddings


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plan = dict(upstream=str(Path(__file__).resolve().parents[2] / 'current_work/MemRL'),
                         writer_max_tokens=768, memory_tokens=2048,
                         rl_config=dict(epsilon=0., alpha=.3, gamma=0., topk=3))
        self.client = Client()

    def make(self, task):
        return ImprovedCLMemory(self.plan, self.client, Path(self.tmp.name) / task,
                                dict(threshold=1.1, mean=.9, std=.1), task, Embeddings())

    def test_scan_state_uses_only_prior_public_peaks_and_survives_restore(self):
        memory = self.make('blind_spectrum_monitoring')
        scan = '| freq: 43.8 MHz | power: -47.8 dBm | width: 5.7 MHz'
        self.assertEqual(memory.retrieve(scan)['context'], '')
        memory.update(scan, '[]', .2, False, memory.retrieve(scan), {})
        self.assertEqual(memory.retrieve(scan)['context'], '')
        # Two close detections in one scan are not evidence of recurrence.
        same_scan = scan + '\n| freq: 44.1 MHz | power: -45.0 dBm | width: 5.8 MHz'
        memory._record_scan(same_scan)
        self.assertEqual(max(x['count'] for x in memory.scan_registry), 2)
        memory.update(scan, '[]', .2, False, memory.retrieve(scan), {})
        result = memory.retrieve('A different current scan')
        self.assertIn('43.8 MHz', result['context'])
        self.assertIn('seen 3 times', result['context'])
        self.assertEqual(result['ids'], [])
        path = Path(self.tmp.name) / 'state.json'
        memory.snapshot(path)
        restored = ImprovedCLMemory(self.plan, self.client, Path(self.tmp.name) / 'restored',
                                    dict(threshold=1.1, mean=.9, std=.1),
                                    'blind_spectrum_monitoring', Embeddings())
        restored.restore(path)
        self.assertEqual(restored.retrieve('A different current scan')['context'], result['context'])

    def test_cohort_cross_stage_fallback_keeps_public_tool_measurements(self):
        memory = self.make('cohort_studies')
        query = '## Study 1/20: HERALD Panel\n**Region(s) in this dataset:** Rural'
        trace = json.dumps([dict(action=dict(tool_call=dict(tool='estimate_survival_by_group',
                               group_expression="CASE WHEN age > 50 THEN 'older' ELSE 'younger' END")),
                                 public_feedback='=== Per-Group Survival ===\n'
                                 '  older: n=30 (30.0%)  S(12m)=0.8000  S(24m)=0.6000  S(36m)=0.4000')])
        memory.update(query, trace, .1, True, memory.retrieve(query), {})
        later = '## Study 5/20: MERIDIAN Registry\n**Region(s) in this dataset:** Urban'
        result = memory.retrieve(later)
        self.assertTrue(result['candidates'])
        self.assertIn('HERALD Panel', result['context'])
        self.assertIn('0.8000/0.6000/0.4000', result['context'])
        self.assertIn('submit all 36', result['context'])
        self.assertLessEqual(result['tokens'], memory.plan['memory_tokens'])

    def test_spectrum_action_uses_prior_scans_and_current_peaks(self):
        class Transmitter(BaseModel):
            center_freq: float
            bandwidth: float
            currently_active: bool
            estimated_power: float

        class Report(BaseModel):
            transmitters: list[Transmitter]

        memory = self.make('blind_spectrum_monitoring')
        old = '| freq: 43.8 MHz | power: -47.8 dBm | width: 5.7 MHz'
        memory._record_scan(old)
        memory._record_scan(old)
        current = '| freq: 103.5 MHz | power: -34.0 dBm | width: 14.8 MHz'
        query = SimpleNamespace(prompt=current, response_schema=Report)
        system = SimpleNamespace(output=Path(self.tmp.name), last=None)

        def actor(q):
            report = Report(transmitters=[])
            system.last = (q.prompt, report.model_dump())
            return SimpleNamespace(action=report, metadata=None)

        system.respond = actor
        memory.decorate_system(system)
        response = system.respond(query)
        centers = [x.center_freq for x in response.action.transmitters]
        self.assertEqual(centers, [43.8, 103.5])
        self.assertFalse(response.action.transmitters[0].currently_active)
        self.assertTrue(response.action.transmitters[1].currently_active)
        self.assertEqual(system.last[1], response.action.model_dump())
        audit = json.loads((Path(self.tmp.name) / 'policy_action.json').read_text())
        self.assertEqual(len(audit['added_history']), 1)

    def test_cohort_submission_averages_prior_raw_reports(self):
        keys = ('group__s12', 'group__s24', 'group__s36')
        Report = create_model('Report', **{key: (float, ...) for key in keys})
        memory = self.make('cohort_studies')
        memory.cohort_reports = [dict(zip(keys, (.9, .7, .5))),
                                 dict(zip(keys, (.7, .5, .3)))]
        prompt = '## Study 5/20: MERIDIAN Registry\n**Region(s) in this dataset:** Urban'
        query = SimpleNamespace(prompt=prompt, response_schema=Report)
        system = SimpleNamespace(output=Path(self.tmp.name), last=None)

        def actor(q):
            action = Report.model_validate(dict(zip(keys, (.2, .2, .2))))
            system.last = (q.prompt, action.model_dump())
            return SimpleNamespace(action=action, metadata=None)

        system.respond = actor
        memory.decorate_system(system)
        response = system.respond(query)
        self.assertAlmostEqual(response.action.group__s12, .8)
        self.assertAlmostEqual(response.action.group__s24, .6)
        self.assertAlmostEqual(response.action.group__s36, .4)
        self.assertEqual(system.last[1], response.action.model_dump())
        memory.update(prompt, '[]', .1, True, memory.retrieve(prompt), {})
        self.assertEqual(memory.cohort_reports[-1], dict(zip(keys, (.2, .2, .2))))


if __name__ == '__main__':
    unittest.main()
