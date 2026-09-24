"""Public observation memory is causal, numerical, and independent of rewards."""

import json
import unittest

from ttcl.ramp.spectrum_memory import SpectrumMemory


def observation(scan, peaks):
    return {
        "scan_number": scan,
        "scan_id": f"scan-{scan}",
        "instance_id": f"instance-{scan}",
        "band_mhz": [0.0, 200.0],
        "detected_peaks": [
            {
                "peak_id": f"peak-{index}",
                "freq_mhz": center,
                "width_mhz": width,
                "power_dbm": -35.0,
            }
            for index, (center, width) in enumerate(peaks)
        ],
    }


class SpectrumMemoryTest(unittest.TestCase):
    def test_history_is_immutable_prompt_snapshot_and_survives_silence(self):
        memory = SpectrumMemory()
        self.assertEqual(memory.context(), "")
        memory.observe(observation(1, [(40.0, 10.0)]))
        first_context = memory.context()
        memory.observe(observation(2, [(42.0, 12.0)]))
        memory.observe(observation(3, []))
        candidate = memory.state_dict()["candidates"][0]
        self.assertEqual(candidate["scan_count"], 2)
        self.assertEqual(candidate["last_seen"], 2)
        self.assertEqual(candidate["center_freq_mhz"], 41.0)
        self.assertEqual(candidate["center_range_mhz"], [40.0, 42.0])
        self.assertEqual(candidate["bandwidth_mhz"], 11.0)
        self.assertEqual(candidate["status"], "repeated_candidate")
        self.assertIn("40.00", first_context)
        self.assertNotIn("42.00", first_context)
        self.assertIn("last=2", memory.context())

    def test_same_scan_peaks_do_not_count_as_repeated_evidence(self):
        memory = SpectrumMemory()
        memory.observe(observation(1, [(40.0, 10.0), (41.0, 10.5)]))
        candidates = memory.state_dict()["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(c["scan_count"] == 1 for c in candidates))
        self.assertTrue(all(c["status"] == "uncertain_single_scan" for c in candidates))
        memory.observe(observation(2, [(40.2, 10.0), (41.2, 10.5)]))
        self.assertEqual([c["scan_count"] for c in memory.state_dict()["candidates"]], [2, 2])

    def test_different_width_modes_are_not_blended(self):
        memory = SpectrumMemory()
        memory.observe(observation(1, [(40.0, 3.0)]))
        memory.observe(observation(2, [(40.1, 15.0)]))
        memory.observe(observation(3, [(40.2, 3.2)]))
        candidates = memory.state_dict()["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(sorted(c["scan_count"] for c in candidates), [1, 2])

    def test_duplicate_or_out_of_order_scan_cannot_contaminate_counts(self):
        memory = SpectrumMemory()
        memory.observe(observation(2, [(40.0, 10.0)]))
        before = memory.state_dict()
        for scan in [2, 1]:
            with self.assertRaises(ValueError):
                memory.observe(observation(scan, [(40.0, 10.0)]))
        self.assertEqual(memory.state_dict(), before)

    def test_only_public_numeric_fields_are_retained(self):
        memory = SpectrumMemory()
        public = observation(1, [(40.0, 10.0)])
        public["latent"] = "secret-answer"
        public["reward"] = 0.9
        public["detected_peaks"][0]["label"] = "secret-answer"
        memory.observe(public)
        public["detected_peaks"][0]["freq_mhz"] = 999.0
        rendered = memory.context() + json.dumps(memory.state_dict())
        self.assertNotIn("secret-answer", rendered)
        self.assertNotIn("999", rendered)
        self.assertNotIn('"reward"', rendered)

    def test_bounded_summary_keeps_repeated_old_candidate(self):
        memory = SpectrumMemory(max_candidates=2, max_tracks=3)
        memory.observe(observation(1, [(40.0, 10.0)]))
        memory.observe(observation(2, [(40.2, 10.0)]))
        for scan, center in enumerate([70.0, 90.0, 110.0, 130.0], start=3):
            memory.observe(observation(scan, [(center, 3.0)]))
        self.assertEqual(len(memory.state_dict()["candidates"]), 3)
        context = memory.context()
        self.assertEqual(len([line for line in context.splitlines() if line.startswith("- ")]), 2)
        self.assertIn("40.10", context)
        self.assertIn("1 tracked candidates omitted", context)

    def test_public_prompt_interface_and_atomic_validation(self):
        memory = SpectrumMemory()
        memory.observe("""--- Scan 1/90 ---
scan_id: scan-one
Detected peaks:
  - peak_id: p1 | freq: 40.0 MHz | power: -35.0 dBm | width: 10.0 MHz
Band: 0.0-200.0 MHz
""", "instance-one")
        before = memory.state_dict()
        invalid = observation(2, [(40.0, 10.0), (70.0, float("nan"))])
        with self.assertRaises(ValueError):
            memory.observe(invalid)
        self.assertEqual(memory.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
