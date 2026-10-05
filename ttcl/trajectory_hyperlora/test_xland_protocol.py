"""Pure split checks; no XLand dependency required in the mainline environment."""

import unittest

from ttcl.trajectory_hyperlora.prepare_xland_ruleset_split import select_split


class XLandProtocolTest(unittest.TestCase):
    def test_split_is_deterministic_and_content_disjoint(self):
        digests = [f"digest-{i}" for i in range(100)]
        first = select_split(digests, 20261005, (12, 4, 4))
        self.assertEqual(first, select_split(digests, 20261005, (12, 4, 4)))
        all_rows = sum(first.values(), [])
        self.assertEqual(len(all_rows), 20)
        self.assertEqual(len({row["ruleset_id"] for row in all_rows}), 20)
        self.assertEqual(len({row["content_sha256"] for row in all_rows}), 20)

    def test_duplicate_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_split(["same", "same", "different"], 1, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
