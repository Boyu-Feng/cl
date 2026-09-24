import unittest

from ttcl.experience_training.core import comparisons, select


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = {"a": {"accepted": True}, "b": {"accepted": True}}
        self.scores = {
            "keep": {"1": 0.2, "2": 0.3},
            "a": {"1": 0.4, "2": 0.5},
            "b": {"1": 0.3, "2": 0.3},
        }

    def test_positive_both(self):
        self.assertEqual(select(self.candidates, self.scores, [1, 2])["selected"], "a")

    def test_missing_control_not_zero(self):
        self.scores["keep"]["2"] = None
        self.assertIsNone(select(self.candidates, self.scores, [1, 2])["selected"])

    def test_mixed_sign_not_reliable(self):
        self.scores["a"]["2"] = 0.1
        self.scores["b"] = self.scores["keep"].copy()
        self.assertIsNone(select(self.candidates, self.scores, [1, 2])["selected"])

    def test_all_tie_not_positive_or_keep_label(self):
        for c in self.candidates:
            self.scores[c] = self.scores["keep"].copy()
        self.assertIsNone(select(self.candidates, self.scores, [1, 2])["selected"])

    def test_harmful_candidates_teach_keep(self):
        for c in self.candidates:
            self.scores[c] = {"1": 0.1, "2": 0.2}
        self.assertEqual(
            select(self.candidates, self.scores, [1, 2])["selected"], "keep"
        )

    def test_invalid_not_used_to_infer_keep(self):
        self.candidates["a"]["accepted"] = False
        self.scores["b"] = {"1": 0.1, "2": 0.2}
        self.assertIsNone(select(self.candidates, self.scores, [1, 2])["selected"])

    def test_incomplete_pairs_excluded_for_all_arms(self):
        rows = [
            dict(
                task="x",
                repeat=1,
                canonical_index=0,
                instance_id="one",
                status="complete",
                arm=a,
                reward=r,
            )
            for a, r in [("none", 0.2), ("trained", 0.4)]
        ]
        rows.append(
            dict(
                task="x",
                repeat=1,
                canonical_index=1,
                instance_id="two",
                status="complete",
                arm="trained",
                reward=1.0,
            )
        )
        result = comparisons(rows, ["none", "trained"], 2)
        self.assertEqual(result["paired_count"], 1)
        self.assertFalse(result["complete"])
        self.assertAlmostEqual(result["arms"]["trained"]["delta_vs_none"], 0.2)


if __name__ == "__main__":
    unittest.main()
