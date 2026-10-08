"""A numeric gate must abstain on same-schema categorical evidence."""

import unittest

import numpy as np

from ttcl.trajectory_hyperlora.typed_evidence_v20 import (
    extract, features, fit, proposals, route,
)


class TypeSeparatedGateTest(unittest.TestCase):
    def test_numeric_training_cannot_authorize_string_copy(self):
        number = extract({"value": 2.}, source_index=0,
            trajectory_sha256="t", feedback="ok", reward=1., query="q")[0]
        numeric = next(x for x in proposals({"value": 4.}, number)
                       if x.operator == "scalar_copy")
        feature = features(numeric, {"value": 4.}, target_index=1,
                           target_query="q")
        model = fit([("scalar_copy", feature + np.ones(len(feature))*i*.001,
                      .5) for i in range(8)], min_examples=8)
        text = extract({"value": "RIGHT"}, source_index=0,
            trajectory_sha256="t", feedback="ok", reward=1., query="q")[0]
        candidates = proposals({"value": "LEFT"}, text)
        self.assertEqual(candidates[0].operator, "categorical_copy")
        self.assertEqual(route(candidates, {"value": "LEFT"}, model,
            target_index=1, target_query="q")[0], "lora")


if __name__ == "__main__":
    unittest.main()
