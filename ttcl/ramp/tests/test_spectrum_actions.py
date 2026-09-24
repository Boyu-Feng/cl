"""Causal catalogs and strict include-list decoding, without benchmark answers."""

import json
import unittest

from ttcl.ramp.spectrum_actions import (
    build_candidate_catalog,
    decode_selection,
    render_selection_prompt,
)
from ttcl.ramp.spectrum_memory import SpectrumMemory
from ttcl.ramp.tests.test_spectrum_memory import observation


class SpectrumActionsTest(unittest.TestCase):
    def test_catalog_includes_current_without_mutating_history(self):
        memory = SpectrumMemory()
        memory.observe(observation(1, [(40.0, 10.0), (80.0, 3.0)]))
        before = memory.state_dict()
        current = observation(2, [(42.0, 12.0), (120.0, 5.0)])
        current["detected_peaks"][0]["power_dbm"] = -31.0
        catalog = build_candidate_catalog(memory, current)
        self.assertEqual(memory.state_dict(), before)
        self.assertEqual(catalog.scan_number, 2)
        by_id = {row.candidate_id: row for row in catalog.candidates}
        self.assertEqual(by_id[1].center_freq, 41.0)
        self.assertEqual(by_id[1].bandwidth, 11.0)
        self.assertEqual(by_id[1].estimated_power, -33.0)
        self.assertEqual(by_id[1].scan_count, 2)
        self.assertTrue(by_id[1].currently_active)
        self.assertFalse(by_id[2].currently_active)
        self.assertTrue(by_id[3].currently_active)
        self.assertEqual(catalog, build_candidate_catalog(memory, current))
        memory.observe(current)
        self.assertEqual(
            [row["candidate_id"] for row in memory.state_dict()["candidates"]],
            sorted(by_id),
        )
        # A previously constructed prompt cannot gain evidence from later scans.
        old_prompt = render_selection_prompt(catalog)
        memory.observe(observation(3, [(180.0, 10.0)]))
        self.assertEqual(render_selection_prompt(catalog), old_prompt)
        self.assertNotIn("180.00", old_prompt)

    def test_selection_maps_only_selected_public_measurements(self):
        memory = SpectrumMemory()
        memory.observe(observation(1, [(40.0, 10.0), (80.0, 3.0)]))
        catalog = build_candidate_catalog(memory, observation(2, [(41.0, 12.0)]))
        decoded = decode_selection('{"include":[2,1]}', catalog)
        self.assertEqual(decoded["transmitters"], [
            {"center_freq": 80.0, "bandwidth": 3.0,
             "currently_active": False, "estimated_power": -35.0},
            {"center_freq": 40.5, "bandwidth": 11.0,
             "currently_active": True, "estimated_power": -35.0},
        ])
        self.assertEqual(decode_selection('{"include":[]}', catalog), {"transmitters": []})

    def test_invalid_actions_and_unknown_ids_fail_closed(self):
        catalog = build_candidate_catalog(SpectrumMemory(), observation(1, [(40, 10), (80, 3)]))
        invalid = [
            "not json", '{"include":[1]} trailing', "[]", "null",
            '{}', '{"include":1}', '{"include":[1],"other":2}',
            '{"include":[true]}', '{"include":[1.0]}', '{"include":["1"]}',
            '{"include":[0]}', '{"include":[-1]}', '{"include":[1,1]}',
            '{"include":[99]}', '{"include":[1,2,3]}',
            '{"include":[1],"include":[2]}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_selection(raw, catalog)

    def test_one_entire_markdown_fence_is_optional_without_relaxing_validation(self):
        catalog = build_candidate_catalog(
            SpectrumMemory(), observation(1, [(40, 10), (80, 3)])
        )
        expected = decode_selection('{"include":[1]}', catalog)
        for raw in [
            '```json\n{"include":[1]}\n```',
            '  ```\n{"include":[1]}\n```  ',
            '```json\r\n{"include":[1]}\r\n```',
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(decode_selection(raw, catalog), expected)
        for raw in [
            'Here is the answer:\n```json\n{"include":[1]}\n```',
            '```json\n{"include":[1]}\n```\nExplanation.',
            '```json\n{"include":[1]} {"include":[2]}\n```',
            '```json\n{"include":[1,1]}\n```',
            '```json\n{"include":[99]}\n```',
            '```json\n{"include":[true]}\n```',
            '```json\n{"include":[1],"include":[2]}\n```',
            '```json\n{"include":[1]}\n```\n```json\n{"include":[2]}\n```',
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_selection(raw, catalog)

    def test_capacity_reserves_current_and_prioritizes_repeated_history(self):
        memory = SpectrumMemory(max_candidates=2, max_tracks=6)
        memory.observe(observation(1, [(40, 10), (80, 3)]))
        memory.observe(observation(2, [(40, 10)]))
        catalog = build_candidate_catalog(memory, observation(3, [(120, 5)]))
        self.assertEqual({row.candidate_id for row in catalog.candidates}, {1, 3})
        self.assertEqual(catalog.omitted_candidates, 1)
        self.assertEqual(catalog.omitted_current_candidates, 0)
        self.assertIn("1 candidates omitted", render_selection_prompt(catalog))
        with self.assertRaises(ValueError):
            decode_selection('{"include":[2]}', catalog)

    def test_full_tracker_does_not_silently_drop_current_catalog_candidates(self):
        memory = SpectrumMemory(max_candidates=1, max_tracks=1)
        memory.observe(observation(1, [(40, 10)]))
        memory.observe(observation(2, [(40, 10)]))
        before = memory.state_dict()
        catalog = build_candidate_catalog(memory, observation(3, [(120, 5)]))
        self.assertEqual(len(catalog.candidates), 1)
        self.assertEqual(catalog.candidates[0].center_freq, 120)
        self.assertTrue(catalog.candidates[0].currently_active)
        self.assertEqual(catalog.omitted_candidates, 1)
        self.assertEqual(memory.state_dict(), before)

    def test_current_overflow_is_bounded_deterministic_and_disclosed(self):
        memory = SpectrumMemory(max_candidates=1)
        current = observation(1, [(40, 10), (80, 3)])
        current["detected_peaks"][1]["power_dbm"] = -20
        catalog = build_candidate_catalog(memory, current)
        self.assertEqual(len(catalog.candidates), 1)
        self.assertEqual(catalog.candidates[0].candidate_id, 2)
        self.assertEqual(catalog.omitted_current_candidates, 1)
        self.assertIn("(1 current)", render_selection_prompt(catalog))

    def test_public_allowlist_and_empty_catalog(self):
        current = observation(1, [])
        current["reward"] = 0.91
        current["latent"] = "SECRET-ANSWER"
        catalog = build_candidate_catalog(SpectrumMemory(), current)
        rendered = render_selection_prompt(catalog) + json.dumps(catalog.state_dict())
        self.assertNotIn("SECRET-ANSWER", rendered)
        self.assertNotIn('"reward"', rendered)
        self.assertEqual(decode_selection('{"include":[]}', catalog), {"transmitters": []})
        with self.assertRaises(ValueError):
            decode_selection('{"include":[1]}', catalog)

    def test_public_prompt_interface_and_schema_validation(self):
        from ttcl.common.bsm import BENCH  # Sets benchmark import path.
        from src.tasks.blind_spectrum_monitoring.task import ScanReport

        self.assertTrue(BENCH.is_dir())
        prompt = """--- Scan 1/90 ---
scan_id: scan-one
Detected peaks:
  - peak_id: p1 | freq: 40.0 MHz | power: -35.0 dBm | width: 10.0 MHz
Band: 0.0-200.0 MHz
"""
        catalog = build_candidate_catalog(SpectrumMemory(), prompt, "instance-one")
        report = decode_selection('{"include":[1]}', catalog, ScanReport)
        self.assertIsInstance(report, ScanReport)
        self.assertEqual(report.transmitters[0].center_freq, 40)


if __name__ == "__main__":
    unittest.main()
