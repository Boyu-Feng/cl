"""Pure-data checks for the four-relation slot LoRA protocol."""

import random
import unittest

from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import (
    DEV_RULES, TEST_RULES, TRAIN_RULES, oracle_bits, source_records,
)
from ttcl.trajectory_hyperlora.slot_lora_raw_readout import unique_bit_accuracy


class SlotTransferProtocolTest(unittest.TestCase):
    def test_whole_rule_splits_are_disjoint(self):
        self.assertEqual(len(set(TRAIN_RULES) | set(DEV_RULES) | set(TEST_RULES)), 16)
        self.assertFalse(set(TRAIN_RULES) & set(DEV_RULES))
        self.assertFalse(set(TRAIN_RULES) & set(TEST_RULES))
        self.assertFalse(set(DEV_RULES) & set(TEST_RULES))
        self.assertTrue(all(rule.bit_count() % 2 == 0 for rule in TRAIN_RULES))
        self.assertTrue(all(rule.bit_count() % 2 == 1
                            for rule in (*DEV_RULES, *TEST_RULES)))

    def test_same_source_observations_opposite_actions(self):
        for rule in range(16):
            source = source_records(rule, random.Random(1234), test=True)
            wrong = source_records(rule ^ 15, random.Random(1234), test=True)
            self.assertEqual(oracle_bits(source),
                             tuple((rule >> cue) & 1 for cue in range(4)))
            for left, right in zip(source, wrong, strict=True):
                self.assertEqual(left["observation"], right["observation"])
                self.assertEqual(left["feedback"], right["feedback"])
                self.assertNotEqual(left["action"], right["action"])

    def test_bit_accuracy_counts_unique_sources(self):
        row = {"source_sha256": "a" * 64, "readout_bits": (1, 0, 1, 0),
               "oracle_bits": (1, 0, 1, 0)}
        self.assertEqual(unique_bit_accuracy([row, row]),
                         {"correct": 4, "total": 4,
                          "exact_sources": 1, "sources": 1})


if __name__ == "__main__":
    unittest.main()
