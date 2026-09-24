import json
import unittest

from ttcl.experience_evolution.clbench import summarize, writer_messages


class TransferTests(unittest.TestCase):
    def test_writer_only_gets_completed_public_information(self):
        episode = {
            "public_task_brief": "brief",
            "initial_public_query": "query",
            "response_schemas": [],
            "steps": [{"action": "SQL", "public_feedback": "visible"}],
            "reward": 0.5,
            "completed": True,
            "format_failures": [],
            "hidden_answer": "SECRET",
            "next_query": "FUTURE",
            "metadata": {"truth": 99},
        }
        messages = writer_messages("old", episode)
        text = messages[1]["content"]
        self.assertNotIn("SECRET", text)
        self.assertNotIn("FUTURE", text)
        self.assertNotIn("metadata", text)
        payload = json.loads(text)
        self.assertEqual(payload["previous_experience"], "old")
        self.assertEqual(
            payload["completed_interaction"]["trajectory"], episode["steps"]
        )
        self.assertEqual(payload["completed_interaction"]["reward"], 0.5)

    def test_missing_scores_excluded_and_negative_rewards_retained(self):
        rows = [
            dict(
                arm=a,
                repeat=303,
                canonical_index=i,
                instance_id=str(i),
                status=status,
                reward=r,
            )
            for a, i, status, r in [
                ("none", 12, "complete", 0.0),
                ("delta", 12, "complete", -0.2),
                ("none", 13, "complete", 0.7),
                ("delta", 13, "failed", None),
            ]
        ]
        result = summarize(rows, ["none", "delta"])
        self.assertEqual(result["paired_count"], 1)
        self.assertEqual(result["arms"]["delta"]["delta_vs_none"], -0.2)
        self.assertEqual(result["arms"]["delta"]["losses"], 1)


if __name__ == "__main__":
    unittest.main()
