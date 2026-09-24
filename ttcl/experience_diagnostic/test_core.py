import unittest
from ttcl.experience_diagnostic.core import ARMS, extract_raw, paired_summary


class Controls(unittest.TestCase):
    def test_missing_not_zero_and_identity(self):
        rows = [dict(source_episode=13, canonical_index=13, repeat=505, arm=a,
                     status="complete", reward=.2, instance_id="a", actor_calls=2) for a in ARMS]
        self.assertEqual(paired_summary(rows)["paired_count"], 1)
        rows[-1]["status"] = "failed"
        rows[-1]["reward"] = None
        self.assertEqual(paired_summary(rows)["paired_count"], 0)
        rows[-1].update(status="complete", reward=.1, instance_id="b")
        with self.assertRaises(ValueError):
            paired_summary(rows)

    def test_excerpts_must_be_public_verbatim(self):
        ep = {"episode": 13, "steps": [{"step": 2, "public_feedback": "a real observation"}]}
        text, audit = extract_raw(ep, {"entries": []}, [{"step": 2, "text": "real observation"}], len, 2048)
        self.assertIn("real observation", text)
        with self.assertRaises(ValueError):
            extract_raw(ep, {"entries": []}, [{"step": 2, "text": "invented observation"}], len)
        with self.assertRaises(ValueError):
            extract_raw(ep, {"entries": []}, [{"step": 2, "text": "real observation"}], len, 1)


if __name__ == "__main__":
    unittest.main()
