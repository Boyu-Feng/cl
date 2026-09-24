"""CPU checks for ordering, held-out feedback and protocol accounting."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ttcl.seal.run_seal_benchmark import (
    BENCH,
    build_memory_examples,
    parse_report,
    parse_scan_observation,
    run,
    score_recall,
    token_chunks,
    validate_memories,
)


class FakeMemory:
    def __init__(self, args):
        self.updates = 0
        self.answers = 0

    def respond(self, query):
        self.answers += 1
        return "invalid" if self.answers == 2 else '{"transmitters": []}'

    def adapt(self, history, output_dir):
        assert all("detected_peaks" in item and "prompt" not in item for item in history)
        assert len(history) <= 1
        self.updates += 1
        return {"update": self.updates}


class ProtocolTest(unittest.TestCase):
    def test_online_order_and_invalid_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=tmp,
                data_path=str(
                    BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl"
                ),
                num_scans=3,
                seed=42,
                memory_window=1,
                update_every=1,
                mode="ttt",
                model="fake",
            )
            result = run(args, memory_factory=FakeMemory)
            records = [
                json.loads(line)
                for line in (Path(tmp) / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["updates_before_answer"] for r in records], [0, 1, 2])
            self.assertEqual(records[1]["reward"], 0)
            self.assertEqual(result["num_updates"], 2)  # No useless final update.
            self.assertEqual(result["invalid_reports"], 1)
            with self.assertRaises(ValueError):
                run(args, memory_factory=FakeMemory)

    def test_material_chunks_preserve_tokens(self):
        ids = list(range(13))
        chunks = list(token_chunks(ids, 5, 99))
        self.assertTrue(all(len(chunk) <= 5 for chunk in chunks))
        self.assertEqual([x for chunk in chunks for x in chunk[:-1]], ids)

    def test_frozen_never_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_dir=tmp,
                data_path=str(
                    BENCH / "data/blind_spectrum_monitoring/mixed_grid_lifecycle.jsonl"
                ),
                num_scans=2,
                seed=42,
                memory_window=1,
                update_every=1,
                mode="frozen",
                model="fake",
            )
            result = run(args, memory_factory=FakeMemory)
            self.assertEqual(result["num_updates"], 0)
            self.assertFalse((Path(tmp) / "updates.jsonl").exists())

    def test_report_with_wrapping_and_bad_object(self):
        from src.tasks.blind_spectrum_monitoring.task import ScanReport

        report = parse_report(
            'note {"other": 1}\n```json\n{"transmitters": []}\n```', ScanReport
        )
        self.assertEqual(report.transmitters, [])
        with self.assertRaises(ValueError):
            parse_report('{"transmitters": [', ScanReport)

    def test_observation_excludes_repeated_instructions(self):
        prompt = """Ignore this task rule and schema.
--- Scan 3/90 ---
Scan metadata:
  scan_id: scan-0003
  estimated_noise_floor_dbm: -60.0
Detected peaks:
  - peak_id: p1 | freq: 43.1 MHz | power: -38.0 dBm | width: 4.2 MHz
Band: 0.0-168.0 MHz
Submit your report."""
        result = parse_scan_observation(prompt, "instance-3")
        self.assertEqual(result["scan_number"], 3)
        self.assertEqual(result["detected_peaks"][0]["freq_mhz"], 43.1)
        self.assertNotIn("prompt", result)

    def test_numeric_material_is_grounded_and_keeps_uncertainty(self):
        observations = [
            {
                "instance_id": "one",
                "scan_number": 1,
                "scan_id": "scan-1",
                "noise_floor_dbm": -60.0,
                "band_mhz": [0.0, 168.0],
                "detected_peaks": [
                    {
                        "peak_id": "p1",
                        "freq_mhz": 43.1,
                        "width_mhz": 4.2,
                        "power_dbm": -38.0,
                    }
                ],
            }
        ]
        raw = json.dumps(
            {
                "memories": [
                    {
                        "center_freq_mhz": 43.1,
                        "bandwidth_mhz": 4.2,
                        "evidence": [
                            {
                                "scan_number": 1,
                                "observed_freq_mhz": 43.1,
                                "observed_width_mhz": 4.2,
                            }
                        ],
                        "confidence": "high",
                    },
                    {
                        "center_freq_mhz": 99.0,
                        "bandwidth_mhz": 8.0,
                        "evidence": [
                            {
                                "scan_number": 1,
                                "observed_freq_mhz": 99.0,
                                "observed_width_mhz": 8.0,
                            }
                        ],
                    },
                ]
            }
        )
        memories, stats = validate_memories(raw, observations)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["confidence"], "low")
        self.assertEqual(stats["source_peak_coverage"], 1.0)
        example = build_memory_examples(memories)[0]
        self.assertIn("43.1 MHz", example["question"])
        self.assertIn("remains uncertain", example["answer"])
        recall = score_recall(
            '{"candidates":[{"center_freq_mhz":43.0,"bandwidth_mhz":4.3}]}',
            memories,
        )
        self.assertEqual(recall["f1"], 1.0)
        natural_recall = score_recall(
            "43.0 MHz / 4.3 MHz / low confidence", memories
        )
        self.assertEqual(natural_recall["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
