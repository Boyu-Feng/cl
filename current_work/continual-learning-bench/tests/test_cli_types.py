import unittest

from src.cli import convert_type
from src.registry import get_class_params
from src.systems.ace import ACESystem
from src.tasks.exploitable_poker.task import Poker


class CLITypesTests(unittest.TestCase):
    def test_get_class_params_resolves_future_annotations(self):
        params = get_class_params(ACESystem)
        self.assertIs(params["curate_every_n_updates"]["type"], int)
        self.assertIs(params["generator_model_kwargs_json"]["type"], str)

    def test_convert_type_handles_task_numeric_annotations(self):
        params = get_class_params(Poker)
        self.assertEqual(convert_type("25", params["num_instances"]["type"]), 25)
        self.assertEqual(convert_type("15", params["big_blind"]["type"]), 15)

    def test_get_class_params_includes_variant_parameter(self):
        params = get_class_params(Poker)
        self.assertEqual(
            convert_type("calling_station", params["variant"]["type"]),
            "calling_station",
        )

    def test_get_class_params_includes_schedule_parameter(self):
        params = get_class_params(Poker)
        self.assertEqual(
            convert_type("quick_test", params["schedule"]["type"]),
            "quick_test",
        )


if __name__ == "__main__":
    unittest.main()
