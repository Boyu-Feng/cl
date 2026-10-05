"""Small mechanism checks for the slot-free trajectory relation pipeline."""

import random
import unittest

import torch
from torch.nn import functional as F

from ttcl.trajectory_hyperlora.feedback_relation_pretrain import (
    FeedbackRelationMemory, add_corrections,
)
from ttcl.trajectory_hyperlora.probe_unseen_marker_vocab import (
    NEW_CUES, changed_source,
)
from ttcl.trajectory_hyperlora.slot_lora_oracle_pilot import source_records


class TinyBatch:
    def __init__(self, ids, mask):
        self.input_ids = ids
        self.attention_mask = mask

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class TinyTokenizer:
    def __call__(self, strings, **_):
        encoded = [[ord(char) % 255 + 1 for char in value] for value in strings]
        width = max(map(len, encoded))
        ids = torch.tensor([row + [0] * (width - len(row)) for row in encoded])
        mask = (ids != 0).long()
        return TinyBatch(ids, mask)


class GenericRelationTest(unittest.TestCase):
    def test_correction_keeps_observation_and_reverses_action(self):
        clean = source_records(3, random.Random(7), test=False)
        corrected = add_corrections(clean, test=False)
        self.assertEqual(len(corrected), 2 * len(clean))
        for index, original in enumerate(clean):
            rejected, accepted = corrected[2 * index:2 * index + 2]
            self.assertEqual(rejected["observation"], original["observation"])
            self.assertEqual(accepted, original)
            self.assertNotEqual(rejected["action"], accepted["action"])
            self.assertEqual(rejected["feedback"], "rejected")

    def test_future_action_loss_reaches_feedback_gate(self):
        embedding = torch.nn.Embedding(256, 12)
        memory = FeedbackRelationMemory(embedding, width=8)
        tokenizer = TinyTokenizer()
        source = add_corrections(source_records(3, random.Random(4), test=False),
                                 test=False)
        logits, relation = memory(
            tokenizer, source,
            ["Choose ALPHA", "Choose BETA", "Choose GAMMA"], "cpu")
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(relation.shape), (8, 8))
        loss = F.cross_entropy(logits, torch.tensor([1, 1, 0]))
        loss.backward()
        self.assertIsNotNone(memory.feedback[1].weight.grad)
        self.assertGreater(float(memory.feedback[1].weight.grad.abs().sum()), 0)
        self.assertIsNone(embedding.weight.grad)

    def test_unseen_vocabulary_swap_preserves_episode_pairing(self):
        source = changed_source(3, 19, corrected=True)
        wrong = changed_source(3 ^ 15, 19, corrected=True)
        self.assertEqual(len(source), 24)
        self.assertTrue(all(any(cue in row["observation"] for cue in NEW_CUES)
                            for row in source))
        for left, right in zip(source, wrong, strict=True):
            self.assertEqual(left["observation"], right["observation"])
            self.assertEqual(left["feedback"], right["feedback"])
            self.assertNotEqual(left["action"], right["action"])


if __name__ == "__main__":
    unittest.main()
